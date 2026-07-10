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
const Derive = @import("symbols/Derive.zig");
pub const DeriveRecipe = Derive.Derive;
const Composite = @import("symbols/Composite.zig");
const Infer = @import("symbols/Infer.zig");
const derive_synth = @import("types/derive_synth.zig");
pub const coherence = @import("types/coherence.zig");
const reify = @import("types/reify.zig");
const register = @import("symbols/register.zig");
const prelude_reg = @import("symbols/Prelude.zig");
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
pub const Kind = @import("layout/Type.zig").Kind;
pub const Type = @import("layout/Type.zig").Type;
pub const IntDesc = @import("layout/Type.zig").IntDesc;
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
pub const type_names = std.StaticStringMap(Type).initComptime(.{
    .{ "int", Type.int },
    .{ "uint", Type.uint },
    .{ "int8", Type.int8 },
    .{ "int16", Type.int16 },
    .{ "int32", Type.int32 },
    .{ "int64", Type.int64 },
    .{ "uint8", Type.uint8 },
    .{ "uint16", Type.uint16 },
    .{ "uint32", Type.uint32 },
    .{ "uint64", Type.uint64 },
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
        // `ctor_is_enum`, so render an enum-App off `enumSyms`, a struct-App off
        // `structSyms`.
        if (ty.kind == .app) {
            const e = self.composite.at(ty.appIdx());
            if (e.ctor_is_enum) {
                if (e.ctor < self.enumSyms().len) return self.enumSyms()[e.ctor].name;
            } else {
                if (e.ctor < self.structSyms().len) return self.structSyms()[e.ctor].name;
            }
        }
        // An integer type renders its sign/width spelling (`int8`/`uint`/…), not the
        // bare `int` kind tag — the single source both passes share.
        if (ty.kind == .int) return ty.intName();
        return @tagName(ty.kind);
    }

    /// Map a type-reference node (an `identifier`, a `literal_unit` for `()`, or in
    /// graph mode a qualified `mod.Type` `field_access`) to a `Type`.
    pub fn typeFromNode(self: anytype, type_node: Ast.Index) Type {
        if (type_node == Ast.none) return Type.unit;
        const tn = self.tree.nodes[type_node.int()];
        if (tn.tag == .literal_unit) return Type.unit; // explicit `-> ()` / `p: ()`
        // A type application `Box[int]` (in type position or a struct-construction lhs)
        // resolves to a composite `App`: the base is a generic struct ctor, each
        // arg is recursively resolved (subst-aware), interned to one composite index.
        if (tn.tag == .type_app) return refs.typeFromTypeApp(self, tn);
        // A qualified cross-module type-ref `mod.Type` parses as a field_access whose
        // receiver binds to a `.module`. Resolve it against the owning module's tables.
        if (tn.tag == .field_access) return refs.typeFromQualified(self, type_node, tn);
        const tok = tn.main_token;
        const name = refs.nameText(self, tok);
        if (type_names.get(name)) |b| return b;
        if (self.activeAliasMap().get(name)) |ty| return ty;
        if (self.activeStructMap().get(name)) |id| {
            // A generic struct named WITHOUT type args (`x: Box`) is not a value type —
            // it needs its args. Diagnose rather than mis-resolve it to `structT`.
            if (id < self.structSyms().len and self.structSyms()[id].is_generic) {
                self.sink.emitFmtCode(.T0001, refs.byteOf(self, tok), "generic type '{s}' requires type arguments, e.g. {s}[int]", .{ name, name }) catch {};
                return .invalid;
            }
            return Type.structT(id);
        }
        if (self.activeEnumMap().get(name)) |id| {
            // A generic enum named WITHOUT type args (`x: Either`) is not a value type —
            // it needs its args. Diagnose rather than mis-resolve it to `enumT`
            // (mirror the generic-struct gate above).
            if (id < self.enumSyms().len and self.enumSyms()[id].is_generic) {
                self.sink.emitFmtCode(.T0001, refs.byteOf(self, tok), "generic type '{s}' requires type arguments, e.g. {s}[int]", .{ name, name }) catch {};
                return .invalid;
            }
            return Type.enumT(id);
        }
        // Inside an inherent method, `Self` names the receiver type. Resolved via
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
    /// a composite `App`. The base must resolve to a GENERIC STRUCT or GENERIC
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
    /// For an inherent method: the receiver type-ref node in the owning
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
    /// Monomorphized generic instances, in canonical order. Empty for a
    /// program with no reachable generic instances. Owned.
    instances: []Mono.Instance = &.{},
    /// The program-wide inherent-method table. Each entry's `name` is BORROWED
    /// from the sibling tree/source (never freed here); only the slice is owned.
    methods: []Method = &.{},
    /// The generic-type method TEMPLATE table. Each entry's `name` is BORROWED from
    /// the sibling tree/source (never freed here); only the slice is owned. Consumed during
    /// checking (dispatch + the mono reification tail); exposed for introspection/tests.
    templates: []TemplateMethod = &.{},
    /// The authorized structural auto-derive recipes, in canonical order. Empty
    /// for a program with no derived `Eq` use. Each recipe's `name`, `params`, and
    /// `field_witnesses` OUTER slice are OWNED here; the `FieldWitness` name slices +
    /// `protocol_name` are BORROWED (sibling derive/instance/fn names + prelude/source
    /// protocol names — all outlive codegen).
    derives: []DeriveRecipe = &.{},
    /// The prelude protocol ids, snapshotted off the checker so codegen resolves
    /// each `==`/`<`/`+`/`.hash()`/`print`/`?`-widen witness by its SPECIFIC protocol (a
    /// sibling protocol reusing the name is excluded). Threaded into every job's `Frozen`
    /// / lower `Inputs`; lower and the fingerprint fold read the SAME bundle.
    prelude_ids: PreludeProtocolIds = .{},
    /// The compiler-provided `char` struct id (see `Prelude.char_struct`), threaded into
    /// lower `Inputs` so the conversion recognizer + char-literal lowering key off it.
    /// Null only for a prelude-less internal caller.
    char_struct: ?u32 = null,

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
            gpa.free(@constCast(inst.name.?));
            // Each conformance's `witness_syms` + `protocol_args` OUTER slices +
            // the vector itself are owned (elements are borrowed source / `gph_fn_names` /
            // PODs).
            for (inst.conformances) |rc| {
                gpa.free(@constCast(rc.witness_syms));
                if (rc.protocol_args.len > 0) gpa.free(@constCast(rc.protocol_args));
            }
            gpa.free(@constCast(inst.conformances));
        }
        gpa.free(self.instances);
        // Method `name`s are borrowed from source (like `Sig.name`); each entry may
        // own a `protocol_args` dupe (freed here), then the backing array.
        freeMethodEntries(gpa, self.methods);
        gpa.free(self.methods);
        gpa.free(self.templates);
        // Derive recipes: free each unit's minted `name`, its `params`, and its
        // `field_witnesses` OUTER slice (the FieldWitness name slices are borrowed), then the
        // backing array.
        freeDeriveEntries(gpa, self.derives);
        gpa.free(self.derives);
        self.* = undefined;
    }
};

/// Free the per-recipe owned data of a derive table: each `name`, `params`, and
/// `field_witnesses` outer slice. The `FieldWitness` name slices + `protocol_name` stay
/// borrowed (sibling derive/instance/fn names + source protocol names).
pub fn freeDeriveEntries(gpa: std.mem.Allocator, derives: []const DeriveRecipe) void {
    for (derives) |d| {
        gpa.free(@constCast(d.name.?));
        if (d.params.len > 0) gpa.free(@constCast(d.params));
        if (d.field_witnesses.len > 0) gpa.free(@constCast(d.field_witnesses));
    }
}

/// One program-wide inherent-method table entry: the receiver `Type`, the
/// SOURCE method name, and the global fn id the method desugared to. Built serially
/// in Pass A (`decodeFnSig`), frozen onto the `Model` before the parallel body pass,
/// and read read-only by `BodyChecker` dispatch, the fingerprint fold, and `lower`.
pub const Method = struct {
    recv: Type,
    name: []const u8,
    fn_id: u32,
    /// `mut self`: the receiver is passed BY ADDRESS and mutated in place, so a
    /// call dispatch must enforce the receiver is a mutable place and `lower` must
    /// pass its address instead of a by-value copy. Type-checking is unaffected —
    /// the receiver type in the `Sig` stays the struct/enum (by-value-logical).
    mut_self: bool = false,
    /// A REIFIED-DISPATCH entry: appended in the mono tail for each reachable
    /// `(instance, method)` of a generic-type `impl`. `recv` is the reified concrete
    /// `structT`/`enumT` and `instance` indexes `GraphResult.instances`, so post-typecheck
    /// consumers (`lower`, `CallVisitor`) resolve the call to the instance's mangled symbol
    /// via `findMethod` on the reified receiver. `null` for an inherent / conformance /
    /// derive entry. `recv` is ALWAYS a concrete type here — generic `impl Box[T]` templates
    /// live in the separate `TemplateMethod` table (which never carries a `recv:Type`), so
    /// `findMethod` post-typecheck can never deref a stale check-time `App`.
    instance: ?u32 = null,
    /// Generic-protocol conformance stamp: the protocol id this method witnesses
    /// (`impl <recv> has P[args]`), or `null` for an inherent method / a reified generic
    /// dispatch entry / a method not registered through `checkCoherence`. Set IN PLACE by
    /// `checkCoherence` (before the Model snapshot), so a doubly-conforming `(recv, name)`
    /// pair carries distinct `(protocol_id, protocol_args)` per conformance and the
    /// multi-conformance resolver can disambiguate them.
    protocol_id: ?u32 = null,
    /// The protocol type-args of the conformance this method witnesses, an OWNED
    /// dupe (freed per-entry via `freeMethodEntries`). Empty for a non-generic protocol or
    /// a non-conformance entry. Folded into the multi-conformance disambiguation key.
    protocol_args: []const Type = &.{},
    /// A SOURCE-LESS auto-derive witness: the index into `GraphResult.derives` of
    /// the synthetic unit this method resolves to, or `null` for a real fn / instance /
    /// builtin entry. Appended in the synthesis barrier for each authorized derive so
    /// `resolveConformanceMethod(T,"eq")` returns `.one` and lower/`foldWitness` route to
    /// the synthetic unit's mangled name (checked BEFORE `instance`/`fn_id`). Grows
    /// `Method` (never `Type`), which is never content-cache-memcpy'd, so `@sizeOf(Type)`
    /// is unaffected.
    derive: ?u32 = null,
};

/// A method TEMPLATE registered for a generic-type `impl Box[T] { .. }`. Kept in
/// its OWN table (never `Method`) precisely because it carries NO `recv: Type`: a template's
/// receiver would be a check-time `App` whose composite index is freed post-typecheck, so
/// making it un-representable here turns the old prose-only "never deref a template's App
/// recv" rule into a type-level guarantee. Dispatch keys off `(recv_ctor, recv_is_enum)`
/// (plain ints, never a composite deref) via `findGenericMethod`, so it is deterministic at
/// any `-jN` and safe while the composite is alive (Pass A / Pass C / the mono tail). Read
/// only during checking; the mono tail turns each reachable `(instance, template)` into a
/// concrete reified `Method`. `name` is BORROWED from source; the table owns no heap data.
pub const TemplateMethod = struct {
    recv_ctor: u32,
    recv_is_enum: bool,
    name: []const u8,
    fn_id: u32,
    mut_self: bool = false,
};

/// Structural equality of two `Type` vectors: same length and pairwise `Type.eql`.
/// Used to compare a conformance's protocol type-args against a use site's explicit args.
pub fn eqlTypeVec(a: []const Type, b: []const Type) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (!Type.eql(x, y)) return false;
    return true;
}

/// The outcome of resolving a method call to a conformance witness. `.one` is the
/// selected `Method`; `.ambiguous` means the receiver conforms to one generic protocol
/// MULTIPLE times and the use site must supply explicit type-args (T0025); `.none` means
/// no method matched (the caller falls back to builtin-scalar / T0018).
pub const MethodPick = union(enum) {
    none,
    one: Method,
    ambiguous,
};

/// The prelude protocol ids an operator/derive desugaring resolves its witness against,
/// bundled so lower / `AstWalk` / `Codegen` thread ONE value from the checker to
/// every witness site. Each `==`/`<`/`+`/`.hash()`/`print`/`?`-widen site resolves by the
/// SPECIFIC id here, so a SIBLING protocol reusing `eq`/`cmp`/… on the same type is never
/// selected — otherwise two same-named conformance methods make the resolver `.ambiguous`,
/// which the checker (keying off conformance EXISTENCE) never sees, and codegen aborts a
/// program `toy check` accepted. Lower and the fingerprint fold MUST pass the IDENTICAL id
/// or warm-cache/`-jN` determinism breaks. A null id (a prelude-less internal caller)
/// degrades to name-only resolution — safe, since no prelude conformance can exist then.
pub const PreludeProtocolIds = struct {
    eq: ?u32 = null,
    ord: ?u32 = null,
    add: ?u32 = null,
    sub: ?u32 = null,
    mul: ?u32 = null,
    div: ?u32 = null,
    hash: ?u32 = null,
    display: ?u32 = null,
    from: ?u32 = null,
    into: ?u32 = null,
    try_into: ?u32 = null,
};

/// Snapshot the prelude protocol ids off a `Typecheck`/`Model` into a `PreludeProtocolIds`
/// (all-null for a prelude-less caller, which denies every witness resolution downstream).
pub fn gatherPreludeIds(src: anytype) PreludeProtocolIds {
    return if (src.prelude) |p| p.protocols else .{};
}

/// The prelude protocol id a desugaring witness method name resolves against (`eq`→Eq,
/// `cmp`→Ord, `add`/`sub`/`mul`/`div`→arith, `hash`→Hash, `display`→Display, `from`→From),
/// or null when the name is not a prelude witness. Centralizes the name→protocol map that
/// the arithmetic operator (its method name computed from the token) and the AstWalk operator
/// fold share, so lower and the fingerprint fold pass the IDENTICAL id.
pub fn witnessProtocolId(ids: PreludeProtocolIds, name: []const u8) ?u32 {
    if (std.mem.eql(u8, name, "eq")) return ids.eq;
    if (std.mem.eql(u8, name, "cmp")) return ids.ord;
    if (std.mem.eql(u8, name, "add")) return ids.add;
    if (std.mem.eql(u8, name, "sub")) return ids.sub;
    if (std.mem.eql(u8, name, "mul")) return ids.mul;
    if (std.mem.eql(u8, name, "div")) return ids.div;
    if (std.mem.eql(u8, name, "hash")) return ids.hash;
    if (std.mem.eql(u8, name, "display")) return ids.display;
    if (std.mem.eql(u8, name, "from")) return ids.from;
    if (std.mem.eql(u8, name, "into")) return ids.into;
    if (std.mem.eql(u8, name, "try_into")) return ids.try_into;
    return null;
}

/// The prelude protocol + enum ids the checker discovers in `registerPrelude`, in fixed
/// append order (Eq=0..From=8; `Ordering`/`Option`/`Result` enums appended after every
/// user type). Populated incrementally during registration, then frozen onto the `Model`.
/// The whole aggregate is `?Prelude` on `Typecheck`/`Model`: `null` for a prelude-less
/// internal caller (one that skips `registerPrelude`), so every operator/derive/native path
/// keys off `null` and degrades to "no conformance" (-> T0026/27/28/…) rather than
/// miscompiling. The nine desugaring-witness protocol ids live in an embedded
/// `PreludeProtocolIds` — the SAME bundle codegen consumes (`gatherPreludeIds`) — so lower /
/// `AstWalk` and the checker never carry divergent copies.
pub const Prelude = struct {
    protocols: PreludeProtocolIds = .{},
    /// The prelude `Ordering{lt,eq,gt}` enum id; the type `Ord.cmp` returns.
    ordering_enum: ?u32 = null,
    /// The prelude `Option[T]`/`Result[T,E]` template enum ids; key native
    /// inherent-method recognition on an `.app` receiver via `optResultFamily`.
    option_enum: ?u32 = null,
    result_enum: ?u32 = null,
    /// The checker-internal `ConvErr{out_of_range}` enum id — the error payload of a
    /// `try_into`'s synthesized `Result[Dst, ConvErr]`. Never named in source.
    conv_err_enum: ?u32 = null,

    /// The compiler-provided `char` tuple struct id (`struct char(uint32)`), appended to
    /// the struct table after every user struct so its id is a pure function of source.
    /// A `char` literal types to `structT(char_struct)`; the `.into()`/`.try_into()`
    /// conversion recognizer keys the char cases off it. Ord/Eq/Hash derive over its inner
    /// `uint32` through the ordinary structural path.
    char_struct: ?u32 = null,

    /// The native-enum family of an `App` ctor: `.option`/`.result` when `ctor` is the
    /// prelude Option/Result TEMPLATE id, else `.none`. A user `enum Option` shadow has its
    /// own distinct id (never the prelude template id), so the native path stays silent on it.
    pub fn optResultFamily(p: Prelude, ctor: u32) LayoutEngine.NativeEnumFamily {
        if (p.option_enum) |oid| if (oid == ctor) return .option;
        if (p.result_enum) |rid| if (rid == ctor) return .result;
        return .none;
    }

    /// Whether `pid` is one of the four structurally-derivable prelude protocols
    /// (Eq/Ord/Hash/Display) — the only protocols a struct/enum can satisfy WITHOUT an
    /// explicit `impl`. A custom/parameterized protocol always requires an impl.
    pub fn isDerivable(p: Prelude, pid: u32) bool {
        inline for (.{ p.protocols.eq, p.protocols.ord, p.protocols.hash, p.protocols.display }) |maybe| {
            if (maybe) |x| if (x == pid) return true;
        }
        return false;
    }
};

/// The ONE disambiguation the three method-dispatch consumers (BodyChecker types, lower
/// symbols, AstWalk fingerprint) share, so all three select the IDENTICAL witness (a
/// divergence would be a miscompile or a `-jN` determinism break). A linear scan over
/// `Type.eql` + name (table = source fn-id order → deterministic):
///   * with `explicit_args`: pick the conformance method whose `protocol_args` match
///     (a generic protocol's conformances are coherence-deduped, so at most one matches),
///     additionally filtered to `witness_pid` when given;
///   * without explicit args, `witness_pid == null` (general `v.m(..)` dispatch) and
///     `<= 1` matching `(recv, name)`: BYTE-IDENTICAL to `findMethod` (the legacy path —
///     inherent, single conformance, prelude, generic instance) so no existing program's
///     dispatch/fingerprint changes;
///   * without explicit args and `>= 2` CONFORMANCE methods match: `.ambiguous`.
/// `witness_pid` (operator/derive desugaring: `==`/`<`/`+`/`.hash()`/`print`/`?`-widen) keeps
/// ONLY methods witnessing that SPECIFIC protocol — so a same-named inherent method (no
/// `protocol_id`) AND a sibling protocol reusing the name are both excluded, and the operator
/// binds its own protocol's witness or nothing. Those methods stay callable through the
/// general `v.m(..)` path (`witness_pid == null`).
pub fn resolveConformanceMethod(methods: []const Method, recv: Type, name: []const u8, witness_pid: ?u32, explicit_args: ?[]const Type) MethodPick {
    if (explicit_args) |ea| {
        for (methods) |m| {
            if (Type.eql(m.recv, recv) and std.mem.eql(u8, m.name, name) and
                m.protocol_id != null and eqlTypeVec(m.protocol_args, ea))
            {
                if (witness_pid) |want| if (m.protocol_id.? != want) continue;
                return .{ .one = m };
            }
        }
        return .none;
    }
    var first: ?Method = null;
    var total: usize = 0;
    var conform_count: usize = 0;
    for (methods) |m| {
        if (Type.eql(m.recv, recv) and std.mem.eql(u8, m.name, name)) {
            if (witness_pid) |want| {
                const got = m.protocol_id orelse continue;
                if (got != want) continue;
            }
            if (first == null) first = m;
            total += 1;
            if (m.protocol_id != null) conform_count += 1;
        }
    }
    if (total == 0) return .none;
    if (total == 1) return .{ .one = first.? };
    if (conform_count >= 2) return .ambiguous;
    return .{ .one = first.? };
}

/// The conformance methods matching `(recv, name)` in table (source fn-id) order — used
/// ONLY to build the T0025 ambiguity message deterministically (protocol id + args per
/// conflicting conformance). `out` is caller-owned; each entry borrows the `Method`.
pub fn conformancesFor(methods: []const Method, recv: Type, name: []const u8, gpa: std.mem.Allocator, out: *std.ArrayList(Method)) !void {
    for (methods) |m| {
        if (m.protocol_id != null and Type.eql(m.recv, recv) and std.mem.eql(u8, m.name, name))
            try out.append(gpa, m);
    }
}

/// The witness `Method` for a SPECIFIC conformance `(recv, name, protocol_id, args)` —
/// how `buildConformances` picks the right witness under multi-conformance (a bare
/// `findMethod` would take the first `(recv, name)`, aliasing a sibling conformance).
/// Null when no such conformance method exists (e.g. a builtin-scalar witness, which has
/// no `Method` entry — the caller then falls back to the bare method name).
pub fn findConformanceMethod(methods: []const Method, recv: Type, name: []const u8, pid: u32, args: []const Type) ?Method {
    for (methods) |m| {
        if (m.protocol_id == pid and Type.eql(m.recv, recv) and std.mem.eql(u8, m.name, name) and eqlTypeVec(m.protocol_args, args))
            return m;
    }
    return null;
}

/// Free the per-entry owned `protocol_args` of a method table. The `name` slices
/// stay borrowed (source); only `protocol_args` is owned (a `checkCoherence` dupe). A
/// reified generic-dispatch entry defaults to `&.{}`, so freeing it is a no-op.
pub fn freeMethodEntries(gpa: std.mem.Allocator, methods: []const Method) void {
    for (methods) |m| if (m.protocol_args.len > 0) gpa.free(@constCast(m.protocol_args));
}

/// Look up an inherent method by receiver type + source name. A linear scan over
/// `Type.eql` (pure content comparison of kind+id) + name equality — no hashmap /
/// thread order, so every consumer selects the SAME method at any `-jN`. Every entry's
/// `recv` is a concrete type (generic `impl Box[T]` templates live in `TemplateMethod`,
/// keyed by ctor int), so this never dereferences a stale check-time `App` — use
/// `findGenericMethod` for the check-time generic-receiver dispatch instead.
pub fn findMethod(methods: []const Method, recv: Type, name: []const u8) ?Method {
    for (methods) |mth| {
        if (Type.eql(mth.recv, recv) and std.mem.eql(u8, mth.name, name)) return mth;
    }
    return null;
}

/// Look up a generic-type method TEMPLATE entry by its receiver type-constructor.
/// Keys off `recv_ctor`/`recv_is_enum` (plain ints, never a composite deref)
/// so it is safe while the composite is alive (Pass A / Pass C / the mono tail) and
/// deterministic at any `-jN`. Returns the first matching template entry; a concrete
/// receiver's `(ctor, is_enum)` uniquely picks the owning `impl`'s method set (one
/// `impl Box[T]` per type until coherence).
pub fn findGenericMethod(templates: []const TemplateMethod, ctor: u32, is_enum: bool, name: []const u8) ?TemplateMethod {
    for (templates) |tmpl| {
        if (tmpl.recv_ctor == ctor and tmpl.recv_is_enum == is_enum and std.mem.eql(u8, tmpl.name, name)) return tmpl;
    }
    return null;
}

/// A builtin scalar protocol-method recognizer: the pure, table-free source
/// of truth for the compiler-registered `Eq` conformance on builtin scalars. Returns the
/// method's return `Type` for a recognized `(recv, name)`, else null. Shared by all
/// three method-dispatch consumers — BodyChecker (types the call), lower (emits an
/// inline machine op), and the fingerprint fold (a fixed sentinel sig) — so a builtin
/// scalar method is NEVER a phantom `t.fns`/`t.methods` entry (that would desync the
/// `names`/`sigs` parallel arrays and churn the method-count unit tests). All four scalars
/// ship `eq` -> bool: `int`/`bool` lower to an inline `icmp`; `str` to a heap-free
/// byte-compare (`load_byte` loop); `unit` to a trivially-true `bconst` (the two
/// `()` values are always equal). None gets a `t.fns` entry — each lowers to a machine op.
pub fn builtinScalarMethod(recv: Type, name: []const u8) ?struct { ret: Type, arity: usize } {
    if (!recv.isScalar()) return null;
    // `eq(self, other) -> bool` is 2-ary (arity counts the non-self args = 1); `hash
    // (self) -> int` is 1-ary (0 non-self args). The `arity` lets the shared
    // method-dispatch consumers gate arg count without hardcoding a per-method constant.
    if (std.mem.eql(u8, name, "eq")) return .{ .ret = Type.bool, .arity = 1 };
    if (std.mem.eql(u8, name, "hash")) return .{ .ret = Type.int, .arity = 0 };
    return null;
}

/// The kind of a conversion the `Into`/`TryInto` recognizer accepts. `widen`/`narrow`
/// are the int↔int cases; the four `char_*`/`*_char` cases are the char surface.
pub const ConvKind = enum { widen, narrow, char_to_int, byte_to_char, int_to_char, char_to_byte };

/// Whether `t` is the compiler-provided `char` struct (`char_id` from the prelude). The
/// single source of the "is this char" predicate; the checker and lower delegate here.
pub fn isCharTy(t: Type, char_id: ?u32) bool {
    return t.kind == .@"struct" and char_id != null and t.struct_id == char_id.?;
}

/// The pure, table-free recognizer for the target-directed `.into()` (lossless
/// widening/identity) and `.try_into()` (fallible any-int-to-int) conversion surface.
/// Shared by the checker (types the call) and lower (emits the inline op) so a
/// conversion is NEVER a phantom `t.methods` row (which would desync `names`/`sigs`) and
/// the two consumers cannot diverge at any `-jN`. `recv` is the receiver type, `expected`
/// the target inferred from context (a let-annotation / return / arg slot / the `T` of a
/// chained `unwrap`). Returns null unless BOTH are integers, so a struct/enum receiver
/// leaves the existing `.ambiguous`/T0025/T0018 dispatch untouched.
///   * `into`: same-signedness AND `expected` at least as wide as `recv` ⇒ `.widen`
///     targeting `expected`; narrowing/sign-change via `into` is null (→ T0018).
///   * `try_into`: any integer `expected` ⇒ `.narrow` targeting `expected` (the desired
///     payload `T`). A widening/identity `try_into` is accepted and always yields `Ok`.
pub fn builtinConvMethod(recv: Type, expected: Type, member: []const u8, char_id: ?u32) ?struct { kind: ConvKind, target: Type } {
    const recv_char = isCharTy(recv, char_id);
    const exp_char = isCharTy(expected, char_id);
    if (std.mem.eql(u8, member, "into")) {
        // char -> int: a codepoint (≤ 21 bits) fits any integer ≥ 32 bits, either sign,
        // losslessly (a char is always non-negative and small).
        if (recv_char and expected.isInteger() and expected.intBits() >= 32)
            return .{ .kind = .char_to_int, .target = expected };
        // byte -> char: a `uint8` (0..=255) is always a valid Unicode scalar.
        if (Type.eql(recv, Type.uint8) and exp_char)
            return .{ .kind = .byte_to_char, .target = expected };
        // int -> int widen (same-sign, non-narrowing).
        if (recv.isInteger() and expected.isInteger() and
            recv.isSigned() == expected.isSigned() and expected.intBits() >= recv.intBits())
            return .{ .kind = .widen, .target = expected };
        return null;
    }
    if (std.mem.eql(u8, member, "try_into")) {
        // int -> char: range-check a valid Unicode scalar (surrogate / > 0x10FFFF fail).
        if (recv.isInteger() and exp_char) return .{ .kind = .int_to_char, .target = expected };
        // char -> byte: range-check the codepoint fits `uint8` (≤ 0xFF).
        if (recv_char and Type.eql(expected, Type.uint8)) return .{ .kind = .char_to_byte, .target = expected };
        // int -> int narrow (fallible).
        if (recv.isInteger() and expected.isInteger()) return .{ .kind = .narrow, .target = expected };
        return null;
    }
    return null;
}

/// A compiler-provided inherent method on the prelude `Option`/`Result` enums:
/// `is_tag0`/`is_tag1` are the two predicates (typed `bool`), `unwrap`/`unwrap_or` yield
/// the payload type-arg (variant 0's payload). Like `builtinScalarMethod`, this is a pure
/// recognizer with NO `t.methods`/`t.fns` entry — the checker types the call and `lower`
/// inlines the tag test / payload load, so a native method never becomes a phantom table
/// row (which would desync `names`/`sigs` and churn the method-count tests). The family's
/// predicate NAMES differ (`is_some`/`is_none` vs `is_ok`/`is_err`) but both map to the
/// same tag-0/tag-1 test; `unwrap`/`unwrap_or` are shared. Variant order is FIXED by
/// `registerPrelude` (payload = index 0, absence/error = index 1).
pub const OptResultMethod = enum { is_tag0, is_tag1, unwrap, unwrap_or };

pub fn optionResultMethod(family: LayoutEngine.NativeEnumFamily, name: []const u8) ?OptResultMethod {
    switch (family) {
        .none => return null,
        .option => {
            if (std.mem.eql(u8, name, "is_some")) return .is_tag0;
            if (std.mem.eql(u8, name, "is_none")) return .is_tag1;
        },
        .result => {
            if (std.mem.eql(u8, name, "is_ok")) return .is_tag0;
            if (std.mem.eql(u8, name, "is_err")) return .is_tag1;
        },
    }
    if (std.mem.eql(u8, name, "unwrap")) return .unwrap;
    if (std.mem.eql(u8, name, "unwrap_or")) return .unwrap_or;
    return null;
}

/// The native-enum family of an `App` ctor: `.option`/`.result` when `ctor` is the
/// prelude Option/Result TEMPLATE id, else `.none`. The checker keys native-method
/// recognition off this on an `.app` receiver during Pass C; a user `enum Option` shadow
/// has its own distinct id (never equals the prelude template id), so the native path
/// stays silent on it.
pub fn optResultFamilyOf(model: *const Model, ctor: u32) LayoutEngine.NativeEnumFamily {
    return if (model.prelude) |p| p.optResultFamily(ctor) else .none;
}

/// Whether `pid` is one of the four structurally-derivable prelude protocols
/// (Eq/Ord/Hash/Display) — the only protocols a struct/enum can satisfy WITHOUT an
/// explicit `impl`. A custom/parameterized protocol always requires an impl, so a
/// bound on one is never discharged structurally. Gates the structural bound-resolution
/// fallback in `enqueueInstance` (a `[T has Ord]` bound satisfied by a derive-only struct).
fn isDerivableProtocol(model: *const Model, pid: u32) bool {
    return if (model.prelude) |p| p.isDerivable(pid) else false;
}

/// A recorded derive request: "type `conform_ty` should structurally derive
/// protocol `protocol_id`". A trivially-copyable POD (no owned data) so a per-checker
/// list moves out by value and the fn-id-ordered merge is a plain concat + dedup — the
/// determinism basis for the synthesized set (see the synthesis barrier).
pub const DeriveReq = struct {
    protocol_id: u32,
    conform_ty: Type,
};

/// A struct field that blocks a structural derive: its declared name + type.
/// Named (not an anonymous struct) so `firstNonConformingField` and the BodyChecker's
/// use site share ONE nominal type.
pub const NonConformingField = struct { name: []const u8, ty: Type };

/// The conformance DECISION module: existence, the type_var bound-as-axiom rule, the
/// composed operator verdict (`classify`), the recursive structural engine (`structural`),
/// and the T0029/30/31 derive-blocker leaf (`firstNonConformingField`) — the single owner
/// of "does `T` conform to `P`, via which tier". The witness PICK stays here as
/// `resolveConformanceMethod` (the leaf lower + the fingerprint fold already share).
pub const conform = @import("types/conform.zig");

/// A declared protocol: a signature-only bundle of method NAMES. Registered
/// SERIALLY in Phase 0c (`registerProtocols`) in module-then-decl order (its global id
/// is its index in `t.protocols`), then frozen onto the `Model`. Only method-NAME
/// completeness is checked here (signature compatibility is a later tier).
pub const ProtocolSym = struct {
    /// Protocol name (borrowed source slice).
    name: []const u8,
    /// Owning module id.
    mod: u32,
    /// Whether the `protocol` decl is `pub` (exported to importers).
    pub_export: bool,
    /// The `protocol_decl` node in its owning module's tree.
    decl_node: Ast.Index,
    /// The protocol's generic type-params, in declaration order (borrowed source
    /// slices), the associated-type replacement: `protocol Into[U]` -> `["U"]`.
    /// Empty for a non-generic protocol. The OUTER array is owned by `t.protocols`. When
    /// a method sig is decoded these live at type-var ordinals 1.. (ordinal 0 is `Self`),
    /// so `-> U` decodes to `type_var(1)`; see `registerProtocols`.
    generic_params: []const []const u8 = &.{},
    /// The protocol's required method names, in declaration order (borrowed source
    /// slices). The OUTER array is owned by `t.protocols` (freed at teardown).
    methods: []const []const u8,
    /// The decoded method SIGNATURES, parallel to `methods` (same declaration
    /// order). `method_params[i]` is method i's param types INCLUDING the leading
    /// synthetic `self` at index 0 (forced to `Type.typeVar(0)` = `Self`); any other
    /// `Self`-typed param also decodes to `type_var(0)`. `method_rets[i]` is the return
    /// (`.unit` for an omitted `-> R`). Consumed by bound-as-axiom body checking (to
    /// type `v.m()` on a `type_var` receiver) and by the T0024 coherence signature
    /// check. The OUTER arrays + each inner `method_params` array are OWNED by
    /// `t.protocols` (freed at teardown); scalar/struct element Types are PODs.
    method_params: []const []const Type = &.{},
    method_rets: []const Type = &.{},
};

/// One recorded conformance: `impl <recv> has <protocol>`. Frozen onto the
/// `Model` (read-only; no Pass-C consumer — it sets up the conformance query).
/// `recv` is the byte-foldable receiver `Type` (a `structT`/`enumT`), stored verbatim
/// so a future `Type.eql` conformance lookup is a one-liner.
pub const Conformance = struct {
    protocol: u32,
    recv: Type,
    /// The protocol type-args this conformance is keyed on: `impl P has Into[int]`
    /// records `[int]`. Two conformances `Into[int]`/`Into[bool]` on ONE type differ ONLY
    /// here — so the coherence key and `findConformance` fold this vector structurally (a
    /// bare `(protocol, recv)` match would alias them). Restricted to concrete non-`App`
    /// value types. Empty for a non-generic protocol. OWNED by `t.conformances`.
    protocol_args: []const Type = &.{},
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
    /// Ordered generic-parameter NAMES for a generic template `fn f[T,U](..)`
    /// (borrowed source slices; the outer array is owned by `t.fns`). Empty for a
    /// non-generic fn. `params`/`ret` of a template carry `Type.typeVar(ord)` where
    /// `ord` indexes this list; the mono tail substitutes them to concrete types.
    generic_params: []const []const u8 = &.{},
    /// The resolved bound protocol id per generic param (`[T has P]`), parallel
    /// to `generic_params` (ordinal-indexed); `null` for an unbounded param. Empty for
    /// a non-generic fn. Populated in `decodeFnSig` via `protocolIdFromNode` (an
    /// undeclared bound protocol -> T0021). Owned by `t.fns` (freed at teardown).
    generic_bounds: []const ?u32 = &.{},
    /// The bound's protocol type-args per generic param (`[T has P[args]]`), parallel
    /// to `generic_bounds`: `generic_bound_args[i]` is the decoded arg vector for param i's
    /// bound (empty for an unbounded param or a non-generic protocol). An arg may be a
    /// `type_var` referencing ANOTHER generic param (`[T has Convert[U]]`), substituted
    /// through the instance args at the mono worklist. Both the inner vectors AND the outer
    /// array are OWNED by `t.fns` (freed at teardown).
    generic_bound_args: []const []const Type = &.{},
    /// For an inherent method: the RESOLVED receiver `Type` (params[0] is the
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

/// The program-wide inherent-method table, built SERIALLY in Phase A
/// (`decodeFnSig`) in global fn-id order, frozen onto the `Model` before the
/// parallel body pass. Transferred into `GraphResult.methods` by `checkGraph`.
methods: std.ArrayList(Method) = .empty,

/// The program-wide generic-type method TEMPLATE table, built SERIALLY in Phase A
/// (`decodeFnSig`) alongside `methods`. Read-only during Pass C / the mono tail; the tail
/// turns each reachable `(instance, template)` into a concrete reified `methods` entry.
templates: std.ArrayList(TemplateMethod) = .empty,

/// The program-wide protocol table, built SERIALLY in Phase 0c
/// (`registerProtocols`) in module-then-decl order (a protocol's global id is its
/// index here). Frozen onto the `Model`; freed at teardown (each entry's `methods`
/// outer array is owned, its name slices borrowed).
protocols: std.ArrayList(ProtocolSym) = .empty,

/// The recorded conformances, filled by the SERIAL `checkCoherence` phase in
/// module-then-decl order. Frozen onto the `Model`; freed at teardown.
/// Pre-seeds the builtin scalar conformances (`(Eq,int)`/`(Eq,bool)`) here in
/// `registerPrelude` BEFORE any user impl, so `checkCoherence` collides a duplicate.
conformances: std.ArrayList(Conformance) = .empty,

/// The prelude protocol/enum ids, populated incrementally by `registerPrelude`
/// in fixed append order (Eq=0..From=8; `Ordering`/`Option`/`Result` enums after every user
/// type — a pure function of source). Each id doubles as the "universal, no import" fallback
/// in `protocolIdFromNode` for a bare `Eq`/`Ord`/… that names no module protocol. `null` for
/// a single-file internal caller that skips `registerPrelude`, so every operator/derive/native
/// path denies conformance (-> T0026/27/28/…) rather than miscompiling. See `Prelude`.
prelude: ?Prelude = null,

/// The receiver `Type` of the method currently being decoded/checked, so a `Self`
/// type-ref resolves to it (via `refs.typeFromNode`'s `selfType` hook). Set around
/// each method's `decodeFnSig` (Pass A); null otherwise (non-method decoding is
/// byte-identical). The body pass sets its own copy on the `BodyChecker`.
cur_self_type: ?Type = null,

/// Monomorphization instances discovered by the serial mono tail. Transferred
/// whole into `GraphResult.instances` by `checkGraph`; the leftover (on an error
/// path) is freed by `checkGraph`'s defer.
mono: std.ArrayList(Mono.Instance) = .empty,

/// Structural derive REQUESTS collected from Pass C: "module fn X used `==` on
/// a derivable type T with no impl". Merged in fn-id order (deterministic) by
/// `checkBodies` from each `BodyResult`, then consumed by the synthesis barrier (deduped
/// + canonical-sorted into `derives`). PODs, no owned data; freed at teardown.
derive_reqs: std.ArrayList(DeriveReq) = .empty,

/// The concrete `Result[char, ConvErr]` / `Result[byte, ConvErr]` a `try_into` call
/// site typed, captured (gate + exact enum layout) so the synthesis barrier can reify
/// the fallible char conversions as shared source-less witnesses instead of inlining
/// their ~24/~12-cell validate+Result-build at every site (the frame-overflow fix).
/// `null` when a program uses no such conversion (then no recipe is appended — the
/// non-conversion path stays byte-identical). OR-merged from every `BodyResult` /
/// mono re-check; all values are the same interned enum, so the merge is order-free.
conv_int_char_result: ?Type = null,
conv_char_byte_result: ?Type = null,

/// The authorized structural auto-derive recipes, built by the synthesis barrier
/// at the end of `monomorphize` in canonical order. Transferred whole into
/// `GraphResult.derives` by `checkGraph`; leftover (error path) freed by the teardown.
derives: std.ArrayList(DeriveRecipe) = .empty,

/// The composite (`App`) intern table. Heap-allocated in `checkGraph` (stable
/// address across the run, so every `BodyChecker` can borrow it), freed at teardown.
/// `App` types are interned here during checking and REIFIED away before the layout
/// snapshot, so nothing in `GraphResult` references it.
composite: *Composite = undefined,

/// Maps a composite (`App`) index -> the concrete `Type` it reified to in the
/// monomorphization tail: a `structT(fresh_struct_id)` for a struct-App, an
/// `enumT(fresh_enum_id)` for an enum-App. Memoizes the `reifyAppTo` dispatcher so each
/// ground `App` reifies to exactly one concrete type, and so `rewriteApp` can restore
/// the decided type without re-dispatching. Deinit'd at teardown.
reify_map: std.AutoHashMapUnmanaged(u32, Type) = .empty,

/// The current `reifyAppTo` recursion depth (termination guard). An unbounded
/// generic-enum whose variant payload re-applies itself with a strictly-growing arg
/// (`enum L[T] { cons(T, L[Box[T]]) }`) recurses forever through
/// `reifyAppToEnum -> substReify -> reifyAppTo`; each level is a fresh distinct enum id
/// (args are grounded to `structT`, so `Composite.appDepth` stays 1 and CANNOT catch
/// it). This true-nesting counter latches T0017 and bails past `max_instantiation_depth`
/// — covering the same latent hazard for `reifyAppToStruct` for free.
reify_depth: u32 = 0,

/// OWNS the mangled names minted for reified generic-struct instances. A reified
/// `StructSym.name` is a view into one of these; `snapshotLayouts` dupes it, so these
/// are freed wholesale at teardown (the struct teardown frees `field_*`/`offsets` but
/// NOT `name`, which for a source struct is a borrowed source slice).
reified_names: std.ArrayList([]u8) = .empty,

/// One-shot latch for the T0017 instantiation-depth diagnostic: the guard fires
/// at the FIRST too-deep instantiation in the (serial) worklist and then goes quiet,
/// so an unbounded `f[T] -> f[Box[T]]` yields exactly one diagnostic at `-j1`/`-jN`.
mono_depth_capped: bool = false,

/// Per-global-fn-id poison flags: true for a BOUNDED generic template whose
/// bound-as-axiom body check (`bodyUnit`) emitted at least one diagnostic. The mono
/// tail (`enqueueInstance`) skips instantiating a poisoned template so a definition
/// error is reported ONCE at the template, never re-cascaded per instance. Allocated
/// serially in `checkBodies` (post-join) in fn-id order; freed at teardown. Empty
/// (`&.{}`) until then, so the guard is inert on any pre-checkBodies path.
bound_poisoned: []bool = &.{},

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
        /// Bare protocol name → GLOBAL protocol id (this module's own protocol decls).
        /// Populated in Phase 0c (`registerProtocols`).
        protocol_ids: std.StringHashMapUnmanaged(u32) = .empty,
        /// Import namespace name → imported module id (graph module id).
        namespaces: std.StringHashMapUnmanaged(u32) = .empty,
        /// Bare alias name -> RESOLVED target `Type` (this module's `type X = Y` decls
        /// + the if-absent-seeded prelude `byte = uint8`). A pure lookup in
        /// `typeFromNode`; all cycle risk is confined to the one-shot resolve pass that
        /// fills this. Aliases mint no id, so this map is never read by lower/codegen/fp.
        alias_ids: std.StringHashMapUnmanaged(Type) = .empty,
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
    /// The program-wide inherent-method table, frozen from Pass A. Read-only
    /// during the parallel body pass; drives `BodyChecker` method dispatch.
    methods: []const Method,
    /// The generic-type method TEMPLATE table, frozen from Pass A. Read-only during
    /// the parallel body pass; drives `BodyChecker` generic-receiver dispatch.
    templates: []const TemplateMethod,
    /// The program-wide protocol table + recorded conformances, frozen before
    /// the parallel body pass. Read-only; no Pass-C consumer (sets up the conformance query).
    protocols: []const ProtocolSym,
    conformances: []const Conformance,
    /// The prelude protocol/enum ids, frozen from the checker, or `null` if
    /// `registerPrelude` never ran (a narrow internal caller). Every `==`/`<`/arith/
    /// `.hash()`/`print`/`?`/native-method path keys off it; `null` denies conformance
    /// (-> T0026/27/28/…) rather than miscompiling. See `Prelude`; the embedded protocol
    /// bundle is fetched via `preludeProtocols`.
    prelude: ?Prelude,

    /// The nine desugaring-witness protocol ids (all-null for a prelude-less caller). The
    /// operator/derive typing reads each id through this so a missing prelude denies
    /// conformance rather than early-returning.
    pub fn preludeProtocols(model: *const Model) PreludeProtocolIds {
        return if (model.prelude) |p| p.protocols else .{};
    }
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
        .templates = t.templates.items,
        .protocols = t.protocols.items,
        .conformances = t.conformances.items,
        .prelude = t.prelude,
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
    // A method: put its receiver type in scope so a `Self` type-ref in a body
    // annotation resolves to it. `.invalid` ⟺ not a method (leave the hook null).
    if (f.self_type.kind != .invalid) bc.cur_self_type = f.self_type;
    // The per-generic-param bound protocol ids, so a bounded template's body can
    // dispatch `v.m()` on a `type_var` receiver via its bound protocol (bound-as-axiom).
    // Empty for a non-generic/unbounded fn (inert); in the per-instance re-check the
    // receiver is grounded so the `type_var` dispatch branch never fires.
    bc.bound_protocols = f.generic_bounds;
    bc.bound_protocol_args = f.generic_bound_args;
    // The generic-param names, so an innermost-failure diagnostic can render an
    // abstract `App`'s failing `type_var` as its source param name.
    bc.gph_generic_params = f.generic_params;
    return bc;
}

/// Switch the active tree/tokens/source/resolutions + bare-name maps to module
/// `mod`. Returns the previous active module so the caller can restore it (layout
/// recursion crosses module boundaries).
pub fn gphSelect(t: *Typecheck, mod: u32) u32 {
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
pub fn activeStructMap(t: *Typecheck) *std.StringHashMapUnmanaged(u32) {
    return &t.graph.mods[t.graph_mod].struct_ids;
}

/// The active bare-name → global-enum-id map: the current module's table.
pub fn activeEnumMap(t: *Typecheck) *std.StringHashMapUnmanaged(u32) {
    return &t.graph.mods[t.graph_mod].enum_ids;
}

/// The active bare-name -> resolved alias `Type` map: the current module's table.
pub fn activeAliasMap(t: *Typecheck) *std.StringHashMapUnmanaged(Type) {
    return &t.graph.mods[t.graph_mod].alias_ids;
}

/// The active bare-name → global-protocol-id map: the current module's table.
pub fn activeProtocolMap(t: *Typecheck) *std.StringHashMapUnmanaged(u32) {
    return &t.graph.mods[t.graph_mod].protocol_ids;
}

/// The layout engine's view of this checker: its tables + the per-module accessors
/// the layout recursion needs, wired to the existing methods. The `emit*` thunks
/// forward to `sink.emitFmt` with the SAME literal format strings the layout code
/// used in-line, so the emitted diagnostics stay byte-identical.
pub fn layoutEnv(t: *Typecheck) LayoutEngine.Env {
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
            return reify.reifyAppTo(tc, app_idx);
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

    // The composite (`App`) intern table. Heap-allocated so its address is stable
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
            // The per-param bound protocol-id array (owned; borrows nothing).
            if (f.generic_bounds.len > 0) gpa.free(@constCast(f.generic_bounds));
            // The per-param bound protocol-args (inner vectors + outer array owned).
            for (f.generic_bound_args) |ba| if (ba.len > 0) gpa.free(@constCast(ba));
            if (f.generic_bound_args.len > 0) gpa.free(@constCast(f.generic_bound_args));
        }
        t.fns.deinit(gpa);
        // Any instances not transferred into the result (an error path) are freed
        // here; the success path empties `t.mono` via `toOwnedSlice` first.
        for (t.mono.items) |*inst| {
            gpa.free(@constCast(inst.args));
            gpa.free(inst.node_types);
            gpa.free(@constCast(inst.params));
            freeInstanceConformances(gpa, inst.conformances);
        }
        t.mono.deinit(gpa);
        // Derive requests (PODs, no owned data) + any un-transferred derive recipes
        // (success path empties `t.derives` via `toOwnedSlice`; an error path frees them).
        t.derive_reqs.deinit(gpa);
        freeDeriveEntries(gpa, t.derives.items);
        t.derives.deinit(gpa);
        // The per-fn bound-poison flags (owned; empty until checkBodies ran).
        if (t.bound_poisoned.len > 0) gpa.free(t.bound_poisoned);
        // The method table's backing array (entries' names are borrowed source
        // slices). Each entry may own a `protocol_args` dupe (freed here). On success
        // `toOwnedSlice` empties it, so both are a no-op there (the result frees them).
        freeMethodEntries(gpa, t.methods.items);
        t.methods.deinit(gpa);
        // The template table owns no per-entry heap data (names are borrowed source);
        // on success `toOwnedSlice` empties it, so this frees only the backing array.
        t.templates.deinit(gpa);
        // The protocol table: each entry owns its `methods` outer array (the
        // name slices are borrowed source); the conformance list owns only its array.
        // Each entry also owns its decoded `method_params` (inner + outer) +
        // `method_rets` arrays (the element Types are PODs). The `generic_params`
        // outer array (name slices borrowed).
        for (t.protocols.items) |p| {
            gpa.free(@constCast(p.methods));
            if (p.generic_params.len > 0) gpa.free(@constCast(p.generic_params));
            for (p.method_params) |mp| gpa.free(@constCast(mp));
            if (p.method_params.len > 0) gpa.free(@constCast(p.method_params));
            if (p.method_rets.len > 0) gpa.free(@constCast(p.method_rets));
        }
        t.protocols.deinit(gpa);
        // Each conformance may own a `protocol_args` dupe.
        for (t.conformances.items) |c| if (c.protocol_args.len > 0) gpa.free(@constCast(c.protocol_args));
        t.conformances.deinit(gpa);
        for (t.enums.items) |e| {
            for (e.variants) |v| {
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(e.variants);
            // A generic enum template's `generic_params` array is owned; a reified /
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
            // A generic template's `generic_params` array is owned; a reified /
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

    // Transfer the monomorphization instances out of the live table into the
    // result before the layout snapshot. `toOwnedSlice` empties `t.mono` so the
    // teardown defer no longer sees them; an errdefer frees them (incl. their minted
    // names) if a later snapshot fails.
    const instances = try t.mono.toOwnedSlice(gpa);
    errdefer {
        for (instances) |*inst| {
            gpa.free(@constCast(inst.args));
            gpa.free(inst.node_types);
            gpa.free(@constCast(inst.params));
            gpa.free(@constCast(inst.name.?));
            freeInstanceConformances(gpa, inst.conformances);
        }
        gpa.free(instances);
    }

    // Transfer the method table out of the live list before the teardown defer
    // sees it. `toOwnedSlice` empties `t.methods`; the entries' names stay borrowed.
    // Each entry may own a `protocol_args` dupe, freed on the error path here (the
    // window after transfer, before the result owns it) — else it would leak.
    const methods = try t.methods.toOwnedSlice(gpa);
    errdefer {
        freeMethodEntries(gpa, methods);
        gpa.free(methods);
    }

    // Transfer the template table out of the live list (mirrors methods); it owns no
    // per-entry heap data, so the error path frees only the backing array.
    const templates_out = try t.templates.toOwnedSlice(gpa);
    errdefer gpa.free(templates_out);

    // Transfer the derive recipes out of the live list before the teardown defer
    // sees them (mirrors instances/methods). `toOwnedSlice` empties `t.derives`; on the
    // error path here the recipes' owned data is freed (else it would leak).
    const derives = try t.derives.toOwnedSlice(gpa);
    errdefer {
        freeDeriveEntries(gpa, derives);
        gpa.free(derives);
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
        .instances = instances,
        .methods = methods,
        .templates = templates_out,
        .derives = derives,
        .prelude_ids = gatherPreludeIds(t),
        .char_struct = if (t.prelude) |p| p.char_struct else null,
    };
}

/// The graph driver: register all types globally, lay them out, decode all fn
/// sigs, check pub-signature coherence, then check every fn body.
fn runGraph(t: *Typecheck, mods: []const GraphModuleInput, fns: []const GraphFnInput, entry_mod: u32) !void {
    // Generics gate. Generic syntax PARSES but has no semantics yet, so a serial
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
        try register.registerStructs(t, Ast.rangeSlice(t.tree, prog.lhs.int()), mod);
    }
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;
        try register.registerEnums(t, Ast.rangeSlice(t.tree, prog.lhs.int()), mod);
    }

    // Phase 0-alias: register each module's transparent `type X = Y` aliases and
    // seed the prelude `byte = uint8` (if-absent). AFTER structs+enums (a target may be
    // either), BEFORE layout (0b) / fn-sig decode (A) so field/param/ret refs resolve
    // through a full alias map. BEFORE prelude, so a `type X = Ordering` (a prelude enum)
    // is out of scope — the corpus never needs it.
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;
        try register.registerAliases(t, Ast.rangeSlice(t.tree, prog.lhs.int()));
    }

    // Prelude: native-register `Eq` (+ its builtin scalar conformances) BEFORE the
    // per-module protocol loop, so `Eq` is global id 0 and is bare-nameable everywhere
    // with no import (the `print` precedent — no module-graph/fingerprint surface).
    t.prelude = try prelude_reg.register(t.gpa, &t.protocols, &t.conformances, &t.enums, &t.structs, t.graph.mods);

    // Phase 0c: register every module's `protocol` decls into ONE global id
    // space (module-id order, then decl order — same determinism as structs/enums).
    // Each module's `protocol_ids` (bare name -> global id) is filled so a bare or
    // qualified protocol reference resolves later. Must run before `checkCoherence`.
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;
        try register.registerProtocols(t, Ast.rangeSlice(t.tree, prog.lhs.int()), mod);
    }

    // Phase 0a: decode each generic struct TEMPLATE's field types as PATTERNS
    // (`v: T` -> `type_var(0)`; `b: Box[T]` -> `App(Box, [type_var 0])`) into its
    // `field_names`/`field_types`, with the template's generic params in scope so a
    // `T` ref decodes to a `type_var`. The mono tail's `substReify` grounds these
    // patterns per instance. Marked `.done` so nothing accidentally lays out a
    // template's type-var fields (Phase 0b skips it anyway).
    try register.decodeTemplateFields(t);
    try register.decodeTemplateVariants(t);

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

    // WHOLE-PROGRAM COHERENCE: walk every `impl .. has ..`, reject a duplicate
    // (protocol, receiver-ctor) conformance (T0020) — including a retroactive one in a
    // sibling module — and an incomplete/undeclared conformance (T0021). SERIAL, in
    // module-then-decl order, BEFORE the parallel Pass-C fan-out — so the emit order is
    // a pure function of source (byte-identical at any `-jN`).
    try coherence.checkCoherence(t, mods);

    // Ord-refines-Eq: after coherence, append exactly one `(Eq, recv)` conformance
    // per `Ord` receiver that lacks an existing `(Eq, recv)` entry — so `==`/`!=` on an
    // Ord-only type routes through `cmp` (no `eq` method is added) and structural
    // derive sees the `Eq` slot filled and never double-fires. Explicit `Eq` is
    // authoritative: a genuine `impl T has Eq` alongside `impl T has Ord` is skipped here
    // (no T0020). Writes ONLY `t.conformances` (never the freed coherence seen-set), serial
    // and deterministic (collect-then-append over a frozen prefix in insertion order).
    try coherence.deriveEqFromOrd(t);

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

    // Monomorphization tail: a SERIAL pass on the LIVE tables, after the
    // per-fn Pass-C fan-out has joined and BEFORE `checkGraph` snapshots/frees them.
    // It discovers every reachable `(template, concrete-args)` instance to a
    // fixpoint, re-checks each instance body into its own `node_types`, and mints a
    // canonical mangled name — a pure function of source, so `-jN` stays identical.
    try t.monomorphize(&model);
}

/// A worklist entry: a `(template gid, concrete args)` to instantiate. `args` is
/// OWNED for the duration of the fixpoint (freed with the worklist).
const Pending = struct { gid: u32, args: []Type };

/// A generous ceiling on the number of monomorphized instances. UNREACHABLE in practice
/// (type-args must already be concrete — there is no `App`, so the instance set is
/// a finite closure of the source's explicit AND bare-inferred call sites); the
/// depth cap below fires first, so this stays the belt-and-suspenders backstop.
const mono_instance_cap: usize = 10_000;

/// The termination guard (T0017): the max `App`-nesting depth of a type-arg before
/// an instantiation is rejected. An unbounded `f[T]` transitively instantiating
/// `f[Box[T]]` forms `App`s of strictly-growing depth (`Box[Box[..[int]..]]`); this
/// cap makes the serial worklist reject it deterministically (never hang/OOM) while a
/// legitimately deep-but-finite generic program (nesting well under 64) still compiles.
pub const max_instantiation_depth: u32 = 64;

/// True when `ty` is a concrete value type usable as a monomorphization type-arg.
/// Admits a ground `App` (a generic-struct instance like `Box[int]` used as a
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

/// Substitute a template type through a concrete arg tuple: a `type_var(ord)`
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

    // Reification: mint a fresh concrete `struct_id` for every reachable ground
    // `App` (in all node_types + instance sigs + non-generic fn sigs) and REWRITE
    // every `.app` to that `structT`. Serial, BEFORE the sort/mangle (so instance
    // args are plain `structT` for `Mono.mangle`) and BEFORE the layout snapshot (so
    // codegen/fingerprint/cache see only concrete structs). Ids are assigned in an
    // index-INDEPENDENT structural-key order, so they are a pure function of source.
    try reify.reifyApps(t, nts);

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
            for (t.mono.items[i + 1 ..]) |b| std.debug.assert(!std.mem.eql(u8, a.name.?, b.name.?));
        }
    }

    // Append a REIFIED-DISPATCH `Method` entry for each reachable
    // (generic-type instance, method). This MUST run at the very end — after the
    // fixpoint, reify, sort, and naming — because the Pass-C `Model` borrows
    // `t.methods.items` (captured before Pass C) and `scanCalls` read it throughout
    // the fixpoint; appending here (once nothing reads `model.methods` again) is the
    // only realloc-safe point. Templates live in the separate `t.templates` array,
    // so appending reified entries to `t.methods` cannot dangle the template iterator.
    // Iterate the already-canonically-sorted `t.mono`, so the reified entries are a
    // pure function of source; each entry's `recv` is the reified concrete
    // `structT`/`enumT` (the instance's substituted-then-reified self, `inst.params[0]`),
    // which is what `findMethod` keys off in lower / `CallVisitor`.
    if (t.templates.items.len > 0) {
        for (t.mono.items, 0..) |inst, i| {
            if (inst.params.len == 0) continue;
            for (t.templates.items) |tmpl| {
                if (tmpl.fn_id != inst.template_gid) continue;
                try t.methods.append(t.gpa, .{
                    .recv = inst.params[0],
                    .name = tmpl.name,
                    .fn_id = tmpl.fn_id,
                    .mut_self = tmpl.mut_self,
                    .instance = @intCast(i),
                });
                break;
            }
        }
    }

    // Auto-derive SYNTHESIS BARRIER. Same realloc-safe point as the append
    // above (nothing reads `model.methods` again): turn Pass C's derive requests into
    // canonical source-less recipes + their synthetic method-table entries.
    try derive_synth.synthesizeDerives(t);

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
            // Explicit-args `id[int](..)`: the type-args are the type-app's
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
            try t.enqueueInstance(model, gid, args, worklist, seen, mc.tokens[n.main_token].start, mod);
        } else if (callee.tag == .identifier) {
            // Bare inferred `id(7)`: re-run the SHARED matcher over the value-arg
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
            try t.enqueueInstance(model, gid, out, worklist, seen, mc.tokens[n.main_token].start, mod);
        } else if (callee.tag == .field_access) {
            // A method call on a generic-type instance `b.get()`: the receiver
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
            const m = findGenericMethod(model.templates, re.ctor, re.ctor_is_enum, member) orelse continue;
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
            try t.enqueueInstance(model, m.fn_id, out, worklist, seen, mc.tokens[n.main_token].start, mod);
        }
    }
}

/// Enqueue `(gid, args)` for instantiation, deduped on the canonical key. `args` is
/// BORROWED (the caller keeps its scratch); a fresh owned copy is stored on the
/// worklist. Shared by the explicit and inferred `scanCalls` branches so their dedup
/// path is byte-identical. `at_byte`/`mod` anchor the T0017 termination diagnostic.
fn enqueueInstance(t: *Typecheck, model: *const Model, gid: u32, args: []const Type, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void), at_byte: u32, mod: u32) !void {
    // A BOUNDED template poisoned by its bound-as-axiom body check (a definition
    // error) is never instantiated — the error was reported once at the template; a
    // per-instance re-check would only re-cascade it.
    if (gid < t.bound_poisoned.len and t.bound_poisoned[gid]) return;
    // Termination guard: reject an instantiation whose type-args nest generic
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

    // Resolve each `[T has P]` bound against the ground type-arg via the frozen
    // conformance table. A MISSING conformance is a use-site T0023 (naming type +
    // protocol at the call byte) and SKIPS this instantiation — no worklist entry, so
    // no per-instance cascade. The `seen` dedup above already fired once per distinct
    // `(gid, args)`, so the error is emitted once per distinct bad instantiation at any
    // `-jN` (the worklist is serial). A satisfied bound's `conform_ty` is concrete, so
    // building the per-instance `ResolvedConformance` in `recheck` is sound.
    const f = model.fns[gid];
    var cmemo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer cmemo.deinit(t.gpa);
    for (f.generic_bounds, 0..) |maybe_pid, ord| {
        const pid = maybe_pid orelse continue;
        if (ord >= args.len) continue;
        // The bound's protocol type-args are substituted THROUGH the instance args
        // (so `[T has Convert[U]]` grounds `U` to the concrete arg bound to `U`), then the
        // conformance is looked up keyed on `(protocol, recv, protocol-args)`.
        const bargs = if (ord < f.generic_bound_args.len) f.generic_bound_args[ord] else &.{};
        var subst_buf: []Type = &.{};
        defer if (subst_buf.len > 0) t.gpa.free(subst_buf);
        if (bargs.len > 0) {
            subst_buf = try t.gpa.alloc(Type, bargs.len);
            for (bargs, 0..) |ba, k| subst_buf[k] = t.substType(ba, args);
        }
        if (conform.existence(model, pid, args[ord], subst_buf)) continue;
        // The frozen table holds only EXPLICIT/prelude conformances. A struct/enum that
        // merely DERIVES Eq/Ord/Hash structurally (no `impl`) satisfies a `[T has Ord]`
        // bound too — the same relation the `<`/`==`/`.hash()` operator predicates
        // use. Fall back to the recursive `conforms` query, gated to a zero-arg DERIVABLE
        // protocol (structural conformance is undefined for a parameterized/custom one, so
        // a custom-protocol bound with no impl still correctly emits T0023). The per-
        // instance re-check records the ground derive request, so the witness is synthesized.
        if (subst_buf.len == 0 and isDerivableProtocol(model, pid) and
            try conform.structural(model.structs, model.enums, model.conformances, args[ord], pid, &cmemo, t.gpa, t.composite, &.{}))
            continue;
        _ = t.gphSelect(mod);
        try t.sink.emitFmtCode(.T0023, at_byte, "type '{s}' does not conform to protocol '{s}'", .{ t.typeName(args[ord]), model.protocols[pid].name });
        return;
    }

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
    // A derivable-struct `==` reached ONLY from an unbounded generic body is seen for the
    // first time here (Pass C skips unbounded templates), so drain the structural-Eq
    // requests recorded during this re-check; `synthesizeDerives` dedups+sorts afterward.
    try t.derive_reqs.appendSlice(t.gpa, bc.derive_reqs.items);
    // A char conversion reached ONLY from an unbounded generic body is first seen here;
    // OR-merge its captured Result so the synthesis barrier still reifies the witness.
    if (t.conv_int_char_result == null) t.conv_int_char_result = bc.conv_int_char_result;
    if (t.conv_char_byte_result == null) t.conv_char_byte_result = bc.conv_char_byte_result;

    // Build this instance's resolved bound conformances (all satisfied — enqueue
    // gated them, so a bound's `conform_ty` is a concrete `structT`/`enumT`/scalar the
    // later `reifyApps` never rewrites). The (e) fingerprint fold folds their structural
    // identity so a sibling-conformance edit invalidates exactly this instance.
    const confs = try t.buildConformances(model, f, args);
    errdefer freeInstanceConformances(t.gpa, confs);

    return .{
        .template_gid = gid,
        .args = args_owned,
        .node_types = inst_nt,
        .params = params,
        .ret = ret,
        .name = null, // minted in canonical order after the sort
        .mod = f.mod,
        .decl_node = f.decl_node,
        .conformances = confs,
    };
}

/// Build a monomorphized instance's resolved `[T has P]` bound conformances,
/// one per bounded generic param in generic-param order. Each witness SymName is resolved
/// in protocol-DECLARED order via `findConformanceMethod` on the ground conforming type,
/// keyed on the bound's `(protocol, protocol-args)` (substituted through the instance
/// args) so multi-conformance picks the RIGHT witness (a bare `findMethod` would take the
/// first `(recv, name)`, aliasing a sibling conformance): the mangled name from
/// `gph_fn_names` when a real impl method exists, else the bare method name (a
/// builtin-scalar witness — deterministic, forward-looking). `protocol_name`
/// borrows the source-backed `ProtocolSym.name`; `witness_syms` borrow `gph_fn_names`
/// (both outlive codegen). `protocol_args` is an OWNED dupe (folded into the (e) fp). Only
/// the outer slice + each `witness_syms`/`protocol_args` outer slice are OWNED. Empty for
/// an unbounded template.
fn buildConformances(t: *Typecheck, model: *const Model, f: FnSym, args: []const Type) ![]const Mono.ResolvedConformance {
    var count: usize = 0;
    for (f.generic_bounds) |b| {
        if (b != null) count += 1;
    }
    if (count == 0) return &.{};
    const list = try t.gpa.alloc(Mono.ResolvedConformance, count);
    var built: usize = 0;
    errdefer {
        for (list[0..built]) |rc| {
            t.gpa.free(@constCast(rc.witness_syms));
            if (rc.protocol_args.len > 0) t.gpa.free(@constCast(rc.protocol_args));
        }
        t.gpa.free(list);
    }
    for (f.generic_bounds, 0..) |maybe_pid, ord| {
        const pid = maybe_pid orelse continue;
        const recv = args[ord];
        const prot = model.protocols[pid];
        // The bound's protocol type-args, substituted through the instance args, so
        // `[T has Convert[U]]` resolves against the concrete arg bound to `U`. OWNED (rides
        // the ResolvedConformance into the (e) fp fold).
        const bargs = if (ord < f.generic_bound_args.len) f.generic_bound_args[ord] else &.{};
        var pargs: []Type = &.{};
        errdefer if (pargs.len > 0) t.gpa.free(pargs);
        if (bargs.len > 0) {
            pargs = try t.gpa.alloc(Type, bargs.len);
            for (bargs, 0..) |ba, k| pargs[k] = t.substType(ba, args);
        }
        const witness = try t.gpa.alloc([]const u8, prot.methods.len);
        for (prot.methods, 0..) |mname, k| {
            witness[k] = if (findConformanceMethod(model.methods, recv, mname, pid, pargs)) |m|
                (if (t.gph_fn_names) |fns| fns[m.fn_id] else mname)
            else
                mname;
        }
        list[built] = .{ .protocol_name = prot.name, .conform_ty = recv, .witness_syms = witness, .protocol_args = pargs };
        built += 1;
    }
    return list;
}

/// Free an instance's owned conformance vector: each entry's `witness_syms` +
/// `protocol_args` OUTER slices + the vector itself. Element slices are borrowed (source /
/// `gph_fn_names`) or PODs (Types) and never freed here.
fn freeInstanceConformances(gpa: std.mem.Allocator, confs: []const Mono.ResolvedConformance) void {
    for (confs) |rc| {
        gpa.free(@constCast(rc.witness_syms));
        if (rc.protocol_args.len > 0) gpa.free(@constCast(rc.protocol_args));
    }
    gpa.free(@constCast(confs));
}

/// The generics gate, now INERT. Generic FUNCTIONS, generic STRUCTS,
/// and generic ENUMS — plus `type_app` in both type and call position — are all
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
    /// True for a BOUNDED generic template whose bound-as-axiom body check
    /// emitted a diagnostic — the mono tail must not instantiate it (no per-instance
    /// cascade of a definition-site error).
    poisoned: bool = false,
    /// Structural derive requests this fn recorded: a `==`/`!=` on a derivable
    /// type with no impl. OWNED (moved out of the `BodyChecker`); merged into
    /// `t.derive_reqs` in fn-id order by `checkBodies`, then freed. PODs (no owned data).
    derive_reqs: []DeriveReq = &.{},
    /// The concrete `Result[char/byte, ConvErr]` this fn's `try_into` sites typed, OR-merged
    /// into `t.conv_*_result` by `checkBodies` (gate for the shared-witness synthesis).
    conv_int_char_result: ?Type = null,
    conv_char_byte_result: ?Type = null,
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
    // Each slot owns a derive-request slice moved out of its BodyChecker; free it
    // once here (safe on every path — an un-merged / errored slot keeps the empty
    // default, and freeing a zero-length slice is a no-op). The merge below copies the
    // PODs into `t.derive_reqs`, so freeing the source afterward is correct.
    defer for (slots) |*s| gpa.free(s.derive_reqs);

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

    // Snapshot the per-fn poison flags in fn-id order for the mono tail, once the
    // units have joined (serial, deterministic). A poisoned BOUNDED template is not
    // instantiated (see `enqueueInstance`). Freed at teardown.
    t.bound_poisoned = try gpa.alloc(bool, n);
    for (slots, 0..) |s, i| t.bound_poisoned[i] = s.poisoned;

    // Merge each fn's derive requests into the program-wide list in FN-ID ORDER (the
    // deterministic collection order the synthesis barrier's dedup+sort depends on —
    // mirrors the poison snapshot above). PODs, so a plain concat.
    for (slots) |s| try t.derive_reqs.appendSlice(gpa, s.derive_reqs);

    // OR-merge the captured conv Result types: every site interned the SAME
    // `Result[char/byte, ConvErr]`, so first-writer-wins is order-free.
    for (slots) |s| {
        if (t.conv_int_char_result == null) t.conv_int_char_result = s.conv_int_char_result;
        if (t.conv_char_byte_result == null) t.conv_char_byte_result = s.conv_char_byte_result;
    }

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
    // A generic TEMPLATE is normally checked only through its concrete instances (the
    // mono tail re-checks each instance body with a substitution) — its `type_var`-typed
    // params/locals have no ABI, so its node_types stay `.invalid`, never lowered. A
    // BOUNDED template (`[T has P]`) is the exception: it is checked ONCE against
    // the bound (bound-as-axiom), so a method-on-`T` error is reported at the definition
    // and the body-fp stays instantiation-independent.
    var any_bound = false;
    for (f.generic_bounds) |b| if (b != null) {
        any_bound = true;
        break;
    };
    if (f.isGeneric() and !any_bound) return; // unbounded template: skip as before
    var bc = t.bodyCheckerFor(model, f);
    defer bc.deinit();
    // A bounded template's `type_var`-typed nodes must NOT land in the shared module
    // node_types (reify/codegen would choke on an ungrounded `type_var`). Type it into a
    // SCRATCH buffer so the module's slots stay `.invalid` for template nodes exactly as
    // before; the scratch is discarded (a template is never a codegen unit — only its
    // reified instances are, and each instance re-check writes its OWN node_types).
    var scratch: []Type = &.{};
    if (f.isGeneric()) {
        scratch = t.gpa.alloc(Type, t.graph.mods[f.mod].tree.nodes.len) catch {
            out.err = error.OutOfMemory;
            return;
        };
        @memset(scratch, .invalid);
        bc.node_types = scratch;
    }
    defer if (scratch.len > 0) t.gpa.free(scratch);
    bc.checkBody(fid, f) catch |e| {
        out.err = e;
        return;
    };
    // Transfer the finished sink into the slot; leave bc holding a fresh empty sink
    // so the `defer bc.deinit()` frees nothing it no longer owns.
    out.sink.deinit();
    out.sink = bc.sink;
    bc.sink = DiagnosticSink.init(bc.gpa);
    // Poison a bounded template whose bound-as-axiom check reported anything, so the
    // mono tail skips instantiating it (a definition error fires once, not per instance).
    out.poisoned = f.isGeneric() and out.sink.count() > 0;
    // Move this fn's derive requests into the slot (merged fn-id-ordered by
    // checkBodies). `toOwnedSlice` empties bc's list so the deferred `bc.deinit` frees
    // nothing it no longer owns.
    out.derive_reqs = bc.derive_reqs.toOwnedSlice(bc.gpa) catch |e| {
        out.err = e;
        return;
    };
    out.conv_int_char_result = bc.conv_int_char_result;
    out.conv_char_byte_result = bc.conv_char_byte_result;
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

/// Resolve a (bare or qualified) protocol-reference node to a GLOBAL protocol id, or
/// null if it names no protocol. NON-EMITTING: `checkCoherence` turns a null into
/// a single T0021 at the reference site (Phase A never touches a protocol ref, so there
/// is no prior emit to double). A bare `identifier` resolves against the active module's
/// protocol map; a qualified `mod.P` `field_access` resolves the receiver namespace then
/// the owning module's protocol table, gated on `pub_export`.
pub fn protocolIdFromNode(t: *Typecheck, ref_idx: Ast.Index) ?u32 {
    if (ref_idx == Ast.none) return null;
    // A generic protocol reference `P[int]` / `mod.P[int]` parses to a `type_app`;
    // resolve its BASE name node (the args are read separately by the caller).
    const node_idx = Ast.protocolRefBase(t.tree, ref_idx);
    const n = t.tree.nodes[node_idx.int()];
    if (n.tag == .identifier) {
        const name = t.nameText(n.main_token);
        // A user protocol of the same name shadows the prelude (active map is consulted
        // first); a bare `Eq` that names no module protocol falls back to the prelude id.
        if (t.activeProtocolMap().get(name)) |id| return id;
        if (t.prelude) |p| {
            const pr = p.protocols;
            if (pr.eq) |id| if (std.mem.eql(u8, name, "Eq")) return id;
            if (pr.ord) |id| if (std.mem.eql(u8, name, "Ord")) return id;
            if (pr.add) |id| if (std.mem.eql(u8, name, "Add")) return id;
            if (pr.sub) |id| if (std.mem.eql(u8, name, "Sub")) return id;
            if (pr.mul) |id| if (std.mem.eql(u8, name, "Mul")) return id;
            if (pr.div) |id| if (std.mem.eql(u8, name, "Div")) return id;
            if (pr.hash) |id| if (std.mem.eql(u8, name, "Hash")) return id;
            if (pr.display) |id| if (std.mem.eql(u8, name, "Display")) return id;
            if (pr.from) |id| if (std.mem.eql(u8, name, "From")) return id;
            if (pr.into) |id| if (std.mem.eql(u8, name, "Into")) return id;
            if (pr.try_into) |id| if (std.mem.eql(u8, name, "TryInto")) return id;
        }
        return null;
    }
    if (n.tag == .field_access) {
        const recv = t.tree.nodes[n.lhs.int()];
        if (recv.tag != .identifier) return null;
        const target = t.graph.namespaceOfIn(t.graph_mod, t.nameText(recv.main_token)) orelse return null;
        if (t.graph.mods[target].protocol_ids.get(t.nameText(n.main_token))) |id| {
            if (t.protocols.items[id].pub_export) return id;
        }
        return null;
    }
    return null;
}

/// Resolve an `impl ... has` receiver type-ref node to its `Type` (a `structT`/`enumT`)
/// WITHOUT emitting. Phase A `decodeFnSig` already resolved the same node (and
/// emitted any T0001/T0002/T0003 for a bad receiver), so re-resolving here must stay
/// silent to avoid a double-emit. A bare `identifier` resolves against the active
/// struct/enum maps; a qualified `mod.T` `field_access` resolves the receiver namespace
/// then the owning module's tables. Null on any miss (Phase A already reported it).
pub fn receiverTypeFromNode(t: *Typecheck, node_idx: Ast.Index) ?Type {
    if (node_idx == Ast.none) return null;
    const n = t.tree.nodes[node_idx.int()];
    if (n.tag == .identifier) {
        const name = t.nameText(n.main_token);
        // A builtin scalar receiver (`impl int has P`) resolves via the same static map
        // Phase A used, so a user impl on int/bool/str keys into the multi-space
        // coherence check. A struct/enum shadow is impossible (T0011 rejects shadowing).
        if (type_names.get(name)) |ty| return ty;
        if (t.activeStructMap().get(name)) |id| return Type.structT(id);
        if (t.activeEnumMap().get(name)) |id| return Type.enumT(id);
        return null;
    }
    if (n.tag == .field_access) {
        const recv = t.tree.nodes[n.lhs.int()];
        if (recv.tag != .identifier) return null;
        const target = t.graph.namespaceOfIn(t.graph_mod, t.nameText(recv.main_token)) orelse return null;
        const member = t.nameText(n.main_token);
        if (t.graph.mods[target].struct_ids.get(member)) |id| return Type.structT(id);
        if (t.graph.mods[target].enum_ids.get(member)) |id| return Type.enumT(id);
        return null;
    }
    return null;
}

/// Ground a protocol-signature type-var to a conformance's concrete types. A
/// protocol's decoded sig uses `type_var(0)` for the `self` slot + any `Self`-typed
/// param/return, and `type_var(k>=1)` for the protocol's generic params (`protocol
/// Into[U]` -> `U == type_var(1)`, ordinal-offset so `Self` keeps 0). Ground `tv(0)` to
/// `recv` and `tv(k>=1)` to `protocol_args[k-1]`; anything else (a concrete type) passes
/// through. The `k-1 < len` guard is defensive against an arity-mismatched impl (already
/// rejected before this is trusted for a witness).
pub fn groundProtoType(ty: Type, recv: Type, protocol_args: []const Type) Type {
    if (!ty.isTypeVar()) return ty;
    const ord = ty.typeVarOrd();
    if (ord == 0) return recv;
    return if (ord - 1 < protocol_args.len) protocol_args[ord - 1] else ty;
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
    // The resolved bound protocol id per generic param (`[T has P]`), parallel to
    // `gnames`; null = unbounded. An undeclared bound protocol is T0021 at the bound
    // ref (protocols are registered in Phase 0c, before this Phase A). `protocolIdFromNode`
    // handles a bare `P` (active module map + `Eq` prelude fallback) and a qualified `mod.P`.
    var gbounds: []?u32 = &.{};
    errdefer if (gbounds.len > 0) t.gpa.free(gbounds);
    // The bound's protocol type-args per param (`[T has P[args]]`), parallel to
    // `gbounds`; inner + outer OWNED by `t.fns`. Decoded WITH the fn's generic params in
    // scope (below) so a bound arg spelled as another param (`[T has Convert[U]]`) becomes
    // a `type_var` substituted at the mono worklist.
    var gbound_args: [][]const Type = &.{};
    errdefer if (gbound_args.len > 0) {
        for (gbound_args) |ba| if (ba.len > 0) t.gpa.free(@constCast(ba));
        t.gpa.free(gbound_args);
    };
    if (proto.generic_params.len > 0) {
        gnames = try t.gpa.alloc([]const u8, proto.generic_params.len);
        for (proto.generic_params, 0..) |gp, i| gnames[i] = t.nameText(t.tree.nodes[gp.int()].main_token);
    }
    // Put the fn's generic params in scope BEFORE decoding the bound args, so a bound arg
    // referencing another param resolves to a `type_var` rather than an unknown type.
    t.cur_generic_params = gnames;
    defer t.cur_generic_params = &.{};
    if (proto.generic_params.len > 0) {
        gbounds = try t.gpa.alloc(?u32, proto.generic_params.len);
        gbound_args = try t.gpa.alloc([]const Type, proto.generic_params.len);
        for (gbound_args) |*ba| ba.* = &.{};
        for (proto.generic_params, 0..) |gp, i| {
            const bound = Ast.genericParamBound(t.tree, gp);
            gbounds[i] = if (bound) |bn| (t.protocolIdFromNode(bn) orelse blk: {
                const ref_tok = t.tree.nodes[Ast.protocolRefBase(t.tree, bn).int()].main_token;
                try t.sink.emitFmtCode(.T0021, t.byteOf(ref_tok), "'{s}' is not a declared protocol", .{t.nameText(ref_tok)});
                break :blk null;
            }) else null;
            // Decode the bound's protocol type-args. A composite `App` arg is out of
            // scope (its check-time index is run-order-dependent — would break the coherence
            // key + fp determinism); reject it and drop the arg. A `type_var` (another
            // param) is fine — it grounds at the mono worklist.
            if (bound) |bn| {
                const arg_nodes = Ast.protocolRefArgs(t.tree, bn);
                if (arg_nodes.len > 0) {
                    const av = try t.gpa.alloc(Type, arg_nodes.len);
                    for (arg_nodes, 0..) |an, k| {
                        const aty = t.typeFromNode(an);
                        if (aty.isApp()) {
                            try t.sink.emit(t.byteOf(t.tree.nodes[an.int()].main_token), "a generic-protocol bound argument must be a concrete non-composite type (composite protocol args are not yet supported)");
                            av[k] = .invalid;
                        } else av[k] = aty;
                    }
                    gbound_args[i] = av;
                }
            }
        }
    }

    // Inherent method: resolve the receiver type and put it in scope so the
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
    try t.fns.append(t.gpa, .{ .decl_node = fn_idx, .kind = .user_fn, .params = params, .ret = ret, .mod = mod, .generic_params = gnames, .generic_bounds = gbounds, .generic_bound_args = gbound_args, .self_type = self_ty });
    // Register the method into the program-wide table (SERIAL, fn-id order). The name
    // is BORROWED from source (like `Sig.name`); the table is frozen before Pass C.
    if (recv_type != Ast.none) {
        const name = t.nameText(decl.main_token);
        const mut_self = proto.params.len > 0 and Ast.isMutParam(t.tree, t.tokens, proto.params[0]);
        if (self_ty.isApp()) {
            // A generic receiver `impl Box[T]`: `self_ty` is a check-time `App`.
            // Record its ctor into the TEMPLATE table so `findGenericMethod` dispatches
            // off a concrete receiver `App`'s ctor without dereferencing the
            // (post-typecheck-freed) composite. A fully-GROUND receiver App (no `type_var`
            // arg) would be a concrete-type inherent impl (`impl Box[int]`), which is
            // coherence territory — reject it cleanly rather than mint an
            // uninstantiable template.
            const e = t.composite.at(self_ty.appIdx());
            var any_var = false;
            for (e.args) |a| if (a.isTypeVar()) {
                any_var = true;
            };
            if (!any_var) {
                try t.sink.emit(t.byteOf(decl.main_token), "an inherent 'impl' on a concrete type instance is not yet supported (use 'impl Box[T]')");
                return;
            }
            try t.templates.append(t.gpa, .{
                .recv_ctor = e.ctor,
                .recv_is_enum = e.ctor_is_enum,
                .name = name,
                .fn_id = gid,
                .mut_self = mut_self,
            });
        } else {
            try t.methods.append(t.gpa, .{
                .recv = self_ty,
                .name = name,
                .fn_id = gid,
                .mut_self = mut_self,
            });
        }
    }
}

/// Append the synthetic bodyless `print(str) -> ()` builtin to the fn table.
fn appendPrint(t: *Typecheck) !void {
    const params = try t.gpa.dupe(Type, &.{.str});
    try t.fns.append(t.gpa, .{ .decl_node = Ast.none, .kind = .builtin, .params = params, .ret = .unit });
}

pub fn typeFromNode(t: *Typecheck, type_node: Ast.Index) Type {
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

/// The receiver type when decoding a method signature, so a `Self` type-ref
/// resolves to it via `refs.typeFromNode`. Null outside a method (byte-identical).
pub fn selfType(t: *const Typecheck) ?Type {
    return t.cur_self_type;
}

/// Intern a composite `App(ctor, args)` to its table index. The `refs`
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

pub fn nameText(t: *const Typecheck, tok: u32) []const u8 {
    return refs.nameText(t, tok);
}

pub fn byteOf(t: *const Typecheck, tok: u32) u32 {
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

test "integer widths — width spelling + builtin scalar-method gate" {
    // The conformance-agreement half of this test (existence + structural both accept every
    // int width) moved to `conform.zig`'s keystone (which additionally pins the witness leaf).
    // What remains is the two facts that live in THIS module: `type_names` maps each width
    // spelling to its constant, and `builtinScalarMethod` gates on `kind == int` (so a width
    // receiver still gets its inline `eq`/`hash`).
    try testing.expect(builtinScalarMethod(Type.int8, "eq") != null);
    try testing.expect(builtinScalarMethod(Type.uint64, "hash") != null);

    try testing.expect(Type.eql(type_names.get("uint8").?, Type.uint8));
    try testing.expect(Type.eql(type_names.get("int16").?, Type.int16));
    try testing.expect(Type.eql(type_names.get("uint").?, Type.uint));
}

test "structural conformance on a recursive generic template terminates" {
    // `next: Node[T]` re-interns to the same App index, so without a coinductive
    // in-progress marker the conformance walk recurses until the stack overflows.
    // Reaching the assertions at all proves termination; the counts pin that a
    // recursive template behaves like its non-recursive analog (clean).
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct Node[T]{val:T,next:Node[T]}
        \\fn f[T has Ord](a:Node[T],b:Node[T])->bool{return a<b}
        \\fn main()->int{return 0}
    ));
    // Recursive field FIRST exercises the `deepestNonConforming` error-path walk,
    // which independently re-descends the self-referential App.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct Node[T]{next:Node[T],val:T}
        \\fn f[T](a:Node[T],b:Node[T])->bool{return a<b}
        \\fn main()->int{return 0}
    ));
}

test "R1: conformance across a multi-node template cycle is expression-order-independent" {
    // `A[T]`↔`B[T]` mutually recurse; neither is `Ord`. Both `x<x` and `y<y` must fire
    // T0027 regardless of which is checked first: the coinductive assumption that closes
    // the cycle must never settle a poisoned `true` for the non-entry member on the shared
    // per-fn memo, or the count would depend on statement order (a non-pure verdict).
    const first_x =
        \\protocol Foo{fn foo(self)->int}
        \\struct A[T]{b:B[T],val:T}
        \\struct B[T]{a:A[T]}
        \\fn f[T has Foo](x:A[T],y:B[T])->int{
        \\  p:=x<x
        \\  q:=y<y
        \\  return 0
        \\}
        \\fn main()->int{return 0}
    ;
    const first_y =
        \\protocol Foo{fn foo(self)->int}
        \\struct A[T]{b:B[T],val:T}
        \\struct B[T]{a:A[T]}
        \\fn f[T has Foo](x:A[T],y:B[T])->int{
        \\  q:=y<y
        \\  p:=x<x
        \\  return 0
        \\}
        \\fn main()->int{return 0}
    ;
    const nx = try checkDiagCount(first_x);
    try testing.expectEqual(nx, try checkDiagCount(first_y));
    try testing.expect(nx != 0);
}
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

test "the generic-fn demo typechecks clean and monomorphizes one instance per type-arg" {
    const gpa = testing.allocator;
    // The earlier gate is NARROWED here: a generic FN + an explicit-args CALL now compile.
    var c = try checkSource("fn id[T](x: T) -> T { x }\nstruct P { x: int, y: int }\nfn main() -> int {\n a := id[int](7)\n p := id[P](P{ x: 20, y: 15 })\n return a + p.x + p.y\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Exactly two instances (id$int, id$P) in canonical order (scalar kind < struct
    // kind), each with a distinct mangled name and a fully-concrete substituted sig.
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
    // The template name is module-qualified (`main.id`) for cross-module uniqueness.
    try testing.expectEqualStrings("main.id$int", c.result.instances[0].name.?);
    try testing.expect(std.mem.startsWith(u8, c.result.instances[1].name.?, "main.id$s"));
    try testing.expectEqual(Kind.int, c.result.instances[0].ret.kind);
    try testing.expectEqual(Kind.@"struct", c.result.instances[1].ret.kind);
    // The substituted params/ret are concrete — no `type_var` survives the mono tail.
    for (c.result.instances) |inst| {
        try testing.expect(!inst.ret.isTypeVar());
        for (inst.params) |p| try testing.expect(!p.isTypeVar());
    }
}

test "repeated call sites of one (template,args) monomorphize to ONE instance (dedup)" {
    const gpa = testing.allocator;
    var c = try checkSource("fn id[T](x: T) -> T { x }\nfn main() -> int {\n a := id[int](1)\n b := id[int](2)\n c := id[int](3)\n return a + b + c\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.id$int", c.result.instances[0].name.?);
}

test "an uninstantiated generic enum (and struct) is clean and reifies NOTHING" {
    const gpa = testing.allocator;
    // A generic enum decl is NO LONGER gated; UNinstantiated it is
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
    // The un-gated generic STRUCT case stays clean too (regression guard).
    try testing.expectEqual(@as(usize, 0), try checkDiagCount("struct Box[T] { v: T }\nfn main() -> int { return 0 }\n"));
}

test "an inherent method typechecks clean; the call types to the method return; one method-table entry" {
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

test "a call to a missing method emits exactly one T0018 naming the receiver + method" {
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

test "`Self` in a method signature resolves to the receiver type" {
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

test "decodeFnSig sets Method.mut_self for `mut self` but not plain `self`; the Sig stays by-value" {
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

test "a mut-self method on a temporary emits T0019; on a local / a field of a local it does not" {
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

test "a mut-self method on a builtin scalar receiver is rejected (T0022), even on a mutable local" {
    const gpa = testing.allocator;
    // On a mutable local `int` place: the place check would pass, but the by-address
    // self ABI has no write-back path for a scalar, so exactly one T0022 fires (not
    // T0019) and the call still types (to `mf.ret`) so nothing cascades.
    {
        var c = try checkSource(
            \\impl int { fn bump(mut self) -> int { return self + 1 } }
            \\fn main() -> int {
            \\ x := 41
            \\ return x.bump()
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0022, c.result.diags[0].code);
    }
    // A by-value `self` method on the same scalar is fine (no by-address ABI).
    {
        var c = try checkSource(
            \\impl int { fn inc(self) -> int { return self + 1 } }
            \\fn main() -> int {
            \\ x := 41
            \\ return x.inc()
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    }
}

test "a generic-type method typechecks clean; one template + one reified entry; call types to T" {
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

    // Exactly one generic TEMPLATE entry (keyed by ctor int, no recv Type) and exactly
    // one REIFIED-DISPATCH entry (recv a concrete struct, carrying the mono instance idx).
    try testing.expectEqual(@as(usize, 1), c.result.templates.len);
    try testing.expectEqualStrings("get", c.result.templates[0].name);
    var reified: usize = 0;
    for (c.result.methods) |m| {
        if (m.instance != null) {
            reified += 1;
            try testing.expectEqual(Kind.@"struct", m.recv.kind);
        }
    }
    try testing.expectEqual(@as(usize, 1), reified);

    // One monomorphized instance: `<path>.Box.get$int`, returning int.
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expect(std.mem.indexOf(u8, c.result.instances[0].name.?, "get$int") != null);
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

test "an UNCALLED generic-type method emits zero instances and zero reified-dispatch entries" {
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
    try testing.expectEqual(@as(usize, 1), c.result.templates.len);
    var reified: usize = 0;
    for (c.result.methods) |m| {
        if (m.instance != null) reified += 1;
    }
    try testing.expectEqual(@as(usize, 0), reified);
}

test "Box[int] and Box[Point] .get() lower to two DISTINCT instances; an uncalled sibling method emits none" {
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
    try testing.expect(!std.mem.eql(u8, c.result.instances[0].name.?, c.result.instances[1].name.?));
    for (c.result.instances) |inst| {
        try testing.expect(std.mem.indexOf(u8, inst.name.?, "same") == null);
        try testing.expect(std.mem.indexOf(u8, inst.name.?, "get$") != null);
    }
}

test "a mut-self generic-type method carries mut_self on the template AND the reified entry" {
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

test "calling a missing method on a generic-type instance emits T0018" {
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

test "an inherent impl on a CONCRETE type instance (impl Box[int]) is rejected, not monomorphized" {
    // Coherence territory: `impl Box[int]` has a fully-ground receiver App (no
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
    try testing.expectEqual(@as(usize, 0), c.result.templates.len);
}

test "builtinScalarMethod recognizes `eq` on all four scalars (pure, table-free)" {
    try testing.expect(builtinScalarMethod(Type.int, "eq") != null);
    try testing.expectEqual(Kind.bool, builtinScalarMethod(Type.int, "eq").?.ret.kind);
    try testing.expect(builtinScalarMethod(Type.bool, "eq") != null);
    try testing.expectEqual(Kind.bool, builtinScalarMethod(Type.bool, "eq").?.ret.kind);
    // str (heap-free byte-compare) and unit (trivially true) now recognize `eq` -> bool.
    try testing.expect(builtinScalarMethod(Type.str, "eq") != null);
    try testing.expectEqual(Kind.bool, builtinScalarMethod(Type.str, "eq").?.ret.kind);
    try testing.expect(builtinScalarMethod(Type.unit, "eq") != null);
    try testing.expectEqual(Kind.bool, builtinScalarMethod(Type.unit, "eq").?.ret.kind);
    // A non-`eq` name and a nominal receiver still recognize nothing.
    try testing.expect(builtinScalarMethod(Type.int, "foo") == null);
    try testing.expect(builtinScalarMethod(Type.structT(0), "eq") == null);
    // Arity: `eq` is 1-ary (one non-self arg).
    try testing.expectEqual(@as(usize, 1), builtinScalarMethod(Type.int, "eq").?.arity);
}

test "resolveConformanceMethod: an operator witness binds only its own protocol; a sibling makes general dispatch ambiguous" {
    // Both the `impl P { fn eq(self, n: int) }` inherent-shadow bug AND the sibling-protocol
    // bug: an inherent `eq` (protocol_id == null) and a `Weird` protocol's `eq` (a DIFFERENT
    // protocol_id) both share the name with the Eq witness. Resolving the Eq OPERATOR witness
    // (witness_pid == the Eq id) must bind ONLY the Eq conformance — the inherent would skip
    // the operator's arg typing, and the sibling would (without the id) make the resolver
    // `.ambiguous`, aborting codegen on a program the checker accepted. Each sibling is still
    // reachable by ITS id, and general `v.eq(..)` dispatch over two same-named conformances is
    // `.ambiguous` (T0025) — exactly the outcome the operator sites dodge by passing the id.
    const P = Type.structT(0);
    const eq_id: u32 = 7;
    const weird_id: u32 = 9;
    const methods = [_]Method{
        .{ .recv = P, .name = "eq", .fn_id = 1 }, // inherent: protocol_id defaults to null
        .{ .recv = P, .name = "eq", .fn_id = 2, .protocol_id = eq_id }, // the Eq witness
        .{ .recv = P, .name = "eq", .fn_id = 3, .protocol_id = weird_id }, // a sibling protocol
    };
    switch (resolveConformanceMethod(&methods, P, "eq", eq_id, null)) {
        .one => |m| {
            try testing.expectEqual(@as(?u32, eq_id), m.protocol_id);
            try testing.expectEqual(@as(u32, 2), m.fn_id);
        },
        .none, .ambiguous => return error.TestUnexpectedResult,
    }
    switch (resolveConformanceMethod(&methods, P, "eq", weird_id, null)) {
        .one => |m| try testing.expectEqual(@as(u32, 3), m.fn_id),
        .none, .ambiguous => return error.TestUnexpectedResult,
    }
    // General dispatch (null pid) over two sibling conformances is ambiguous — the checker
    // reports T0025; the operator path never sees this because it filters by protocol id.
    try testing.expect(resolveConformanceMethod(&methods, P, "eq", null, null) == .ambiguous);

    // With a SINGLE conformance, general dispatch reaches the first same-named method (the
    // inherent) — the legacy `findMethod` behavior a plain `v.eq(7)` call still relies on.
    const single = [_]Method{
        .{ .recv = P, .name = "eq", .fn_id = 1 },
        .{ .recv = P, .name = "eq", .fn_id = 2, .protocol_id = eq_id },
    };
    switch (resolveConformanceMethod(&single, P, "eq", null, null)) {
        .one => |m| try testing.expectEqual(@as(u32, 1), m.fn_id),
        .none, .ambiguous => return error.TestUnexpectedResult,
    }
}

test "builtinScalarMethod recognizes `hash` -> int (arity 0) on all four scalars" {
    inline for (.{ Type.int, Type.bool, Type.str, Type.unit }) |sc| {
        const bm = builtinScalarMethod(sc, "hash") orelse return error.TestUnexpectedResult;
        try testing.expectEqual(Kind.int, bm.ret.kind);
        try testing.expectEqual(@as(usize, 0), bm.arity); // `hash(self)` takes no non-self args
    }
    // A nominal receiver has no builtin `hash` (it derives structurally instead).
    try testing.expect(builtinScalarMethod(Type.structT(0), "hash") == null);
    try testing.expect(builtinScalarMethod(Type.enumT(0), "hash") == null);
}

test "a.eq(b) on int types the call to bool, zero diags, and pollutes no method entry" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ a := 41
        \\ b := 41
        \\ return if a.eq(b) { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The builtin recognizer must NOT add a phantom method (the method-count tests rely
    // on this): a program with no user impl has an empty method table.
    try testing.expectEqual(@as(usize, 0), c.result.methods.len);
    // The `a.eq(b)` call node types to bool.
    const nts = c.result.node_types[0];
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .call) {
            try testing.expectEqual(Kind.bool, nts[i].kind);
            found = true;
        }
    }
    try testing.expect(found);
}

test "`==` on a struct with `impl P has Eq` types to bool, zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn main() -> int {
        \\ p := P{ x: 1 }
        \\ q := P{ x: 1 }
        \\ return if p == q { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The `p == q` binary node types to bool. `main` is module 0's last fn; find the
    // eq_eq binary and assert bool.
    const nts = c.result.node_types[0];
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .binary and c.tokens[n.main_token].tag == .eq_eq) {
            try testing.expectEqual(Kind.bool, nts[i].kind);
            found = true;
        }
    }
    try testing.expect(found);
}

test "`==` on an all-Eq-fields struct with no impl DERIVES (was T0026)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ p := P{ x: 1 }
        \\ q := P{ x: 2 }
        \\ return if p == q { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The former error is now a source-less structural derive: no diagnostic, and
    // exactly one synthetic recipe (`Eq` for P, struct id 0 -> `Eq$eq$s0`).
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Eq$eq$s0", c.result.derives[0].name.?);
}

test "`==` on a PAYLOAD enum with no impl now DERIVES Eq (gap closed)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { A(int), B }
        \\fn main() -> int {
        \\ p := E.B
        \\ q := E.B
        \\ return if p == q { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // A payload enum's variant payloads recurse in `conforms`, so an all-`Eq`-payload
    // enum structurally conforms -> one source-less `Eq` recipe (`Eq$eq$e0`), zero diags.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Eq$eq$e0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.eq, c.result.derives[0].kind);
}

test "`<` on a PAYLOAD enum with no impl DERIVES Ord (discriminant-then-payload)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum N { Z, S(int) }
        \\fn main() -> int {
        \\ return if N.Z < N.S(1) { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // One source-less `Ord` recipe (`Ord$cmp$e0`), no separate Eq unit; zero diags.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Ord$cmp$e0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.ord, c.result.derives[0].kind);
    try testing.expectEqual(Kind.@"enum", c.result.derives[0].ret.kind); // ret is `Ordering`
}

test "str `==` and unit `==` type to bool (builtin-scalar Eq), zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn nothing() { return }
        \\fn main() -> int {
        \\ s := if "a" == "a" { 1 } else { 0 }
        \\ u := if nothing() == nothing() { 1 } else { 0 }
        \\ return s + u
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "a bounded generic body `fn eq2[T has Eq](a: T, b: T) -> bool { a == b }` checks once" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn eq2[T has Eq](a: T, b: T) -> bool { a == b }
        \\fn main() -> int {
        \\ return if eq2[int](7, 7) { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "`==` in a body bounded by a NON-Eq protocol is T0026 (bound is not the Eq axiom)" {
    const gpa = testing.allocator;
    // A bounded template body IS checked (bound-as-axiom); an UNBOUNDED one is skipped
    // (types.zig `unbounded template: skip as before`), so the type_var-conforms branch
    // is only exercised on a bounded body. `T has Doubler` gives `T` a bound, but it is
    // NOT `Eq`, so `a == b` cannot desugar and must be T0026.
    var c = try checkSource(
        \\protocol Doubler { fn dbl(self) -> int }
        \\fn bad[T has Doubler](a: T, b: T) -> bool { a == b }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    var n26: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0026) n26 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n26);
}

test "a cross-type `==` (both Eq) stays a homogeneity error, never T0026" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\struct Q { y: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\impl Q has Eq { fn eq(self, o: Q) -> bool { self.y == o.y } }
        \\fn main() -> int {
        \\ p := P{ x: 1 }
        \\ q := Q{ y: 1 }
        \\ return if p == q { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    // Homogeneity is enforced FIRST, so a cross-type compare is a "same type" error —
    // NOT T0026 — even though P and Q each conform to Eq.
    try testing.expect(c.result.diags[0].code != codes.Code.T0026);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "same type") != null);
}

test "a duplicate user `impl int has Eq` overlaps the builtin conformance (one T0020)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\impl int has Eq { fn eq(self, o: int) -> bool { true } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    var n20: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0020) n20 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n20);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "overlapping impl") != null);
}

test "the prelude Ordering enum has variants lt=0/eq=1/gt=2 (the discriminant lower reads)" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() -> int { return 0 }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var found = false;
    for (c.result.enum_layouts) |el| {
        if (!std.mem.eql(u8, el.name, "Ordering")) continue;
        found = true;
        try testing.expectEqual(@as(usize, 3), el.variants.len);
        try testing.expectEqualStrings("lt", el.variants[0].name);
        try testing.expectEqualStrings("eq", el.variants[1].name);
        try testing.expectEqualStrings("gt", el.variants[2].name);
        try testing.expectEqual(@as(u32, 8), el.size);
    }
    try testing.expect(found);
}

test "prelude Option/Result are registered generic enum templates with the right variants" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() -> int { return 0 }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var opt: ?EnumLayout = null;
    var res: ?EnumLayout = null;
    for (c.result.enum_layouts) |el| {
        if (std.mem.eql(u8, el.name, "Option")) opt = el;
        if (std.mem.eql(u8, el.name, "Result")) res = el;
    }
    // `Option[T] { some(T), none }`: a tuple `some` (one type_var payload) then a unit `none`.
    try testing.expect(opt != null);
    try testing.expectEqual(@as(usize, 2), opt.?.variants.len);
    try testing.expectEqualStrings("some", opt.?.variants[0].name);
    try testing.expectEqual(VariantForm.tuple, opt.?.variants[0].form);
    try testing.expectEqual(@as(usize, 1), opt.?.variants[0].field_types.len);
    try testing.expect(opt.?.variants[0].field_types[0].isTypeVar());
    try testing.expectEqualStrings("none", opt.?.variants[1].name);
    try testing.expectEqual(VariantForm.unit, opt.?.variants[1].form);
    // `Result[T,E] { ok(T), err(E) }`: two tuple variants whose payloads are distinct type_vars.
    try testing.expect(res != null);
    try testing.expectEqual(@as(usize, 2), res.?.variants.len);
    try testing.expectEqualStrings("ok", res.?.variants[0].name);
    try testing.expectEqualStrings("err", res.?.variants[1].name);
    try testing.expectEqual(@as(u32, 0), res.?.variants[0].field_types[0].typeVarOrd());
    try testing.expectEqual(@as(u32, 1), res.?.variants[1].field_types[0].typeVarOrd());
}

test "Option[int] and Result[int,str] construct + match with no import; reify to concrete enums" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := Option[int].some(7)
        \\ y := Result[int, str].ok(3)
        \\ a := match x { .some(v) -> v, .none -> 0 }
        \\ b := match y { .ok(v) -> v, .err(_) -> 0 }
        \\ return a + b
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var opt_int = false;
    var res_int_str = false;
    for (c.result.enum_layouts) |el| {
        if (std.mem.eql(u8, el.name, "Option$int")) opt_int = true;
        if (std.mem.eql(u8, el.name, "Result$int$str")) res_int_str = true;
    }
    try testing.expect(opt_int);
    try testing.expect(res_int_str);
}

test "a user `enum Option` shadows the prelude (user-first-wins)" {
    // The prelude injection into each module's enum map is if-absent, so a user decl keeps
    // the name: `Option.red` here resolves to the USER enum's variants (the prelude Option
    // has only `some`/`none` and needs a type arg — so a leak-through would NOT type-check).
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\enum Option { red, green }
        \\fn main() -> int { return match Option.red { .red -> 42, .green -> 0 } }
        \\
    ));
}

/// The `Kind` a method call `recv.member(..)` typed to (single-module test helper), or
/// null if no such call node is present. Scans for a `.call` over a `field_access` whose
/// member token matches — used by the native-method typing tests.
fn methodCallKind(c: Checked, member: []const u8) ?Kind {
    const nts = c.result.node_types[0];
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .call) continue;
        const callee = c.tree.nodes[(n.lhs).int()];
        if (callee.tag != .field_access) continue;
        if (std.mem.eql(u8, c.tokens[callee.main_token].text(c.source), member)) return nts[i].kind;
    }
    return null;
}

test "Option native methods type to bool (predicates) / T (unwrap) with zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := Option[int].some(7)
        \\ p := x.is_some()
        \\ q := x.is_none()
        \\ u := x.unwrap()
        \\ d := x.unwrap_or(0)
        \\ if p { if q { return u } }
        \\ return d
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_some").?);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_none").?);
    try testing.expectEqual(Kind.int, methodCallKind(c, "unwrap").?);
    try testing.expectEqual(Kind.int, methodCallKind(c, "unwrap_or").?);
}

test "Result native methods type to bool (predicates) / T (unwrap) with zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := Result[int, str].ok(7)
        \\ p := x.is_ok()
        \\ q := x.is_err()
        \\ u := x.unwrap()
        \\ d := x.unwrap_or(0)
        \\ if p { if q { return u } }
        \\ return d
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_ok").?);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_err").?);
    // `unwrap`/`unwrap_or` yield the FIRST type-arg (the `ok` payload), int here.
    try testing.expectEqual(Kind.int, methodCallKind(c, "unwrap").?);
    try testing.expectEqual(Kind.int, methodCallKind(c, "unwrap_or").?);
}

test "unwrap_or with a wrong-typed default is exactly one diagnostic" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn main() -> int {
        \\ x := Option[int].some(7)
        \\ return x.unwrap_or("s")
        \\}
        \\
    ));
}

test "a native predicate called with an argument is exactly one diagnostic (arity)" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn main() -> int {
        \\ x := Option[int].some(7)
        \\ b := x.is_some(1)
        \\ if b { return 1 }
        \\ return 0
        \\}
        \\
    ));
}

test "an unknown method on Option is still exactly one T0018" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := Option[int].some(7)
        \\ return x.frobnicate()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
}

test "a native method name on a user `enum Option` shadow is T0018, not the native path" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum Option { red, green }
        \\fn main() -> int {
        \\ a := Option.red
        \\ return a.is_some()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
}

test "unwrap on a str (non-scalar) payload is deferred — exactly one T0018, no crash" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := Option[str].some("hi")
        \\ s := x.unwrap()
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
}

test "unwrap_or on a str (non-scalar) payload is deferred — exactly one T0018" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn main() -> int {
        \\ x := Option[str].some("hi")
        \\ s := x.unwrap_or("d")
        \\ return 0
        \\}
        \\
    ));
}

test "unwrap on a struct payload is deferred — exactly one T0018, no crash" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ x := Option[P].some(P { x: 40 })
        \\ p := x.unwrap()
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
}

test "unwrap_or on a struct payload is deferred — exactly one T0018, no crash" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ d := P { x: 2 }
        \\ p := Option[P].none.unwrap_or(d)
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
}

test "predicates stay native for a non-scalar payload (tag-only, safe)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ x := Option[P].some(P { x: 40 })
        \\ b := x.is_some()
        \\ if b { return 1 }
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_some").?);
}

/// The `Kind` the (first) `try_expr` node typed to (single-module test helper), or null
/// if none is present. Used by the `?` typing tests.
fn tryExprKind(c: Checked) ?Kind {
    const nts = c.result.node_types[0];
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .try_expr) return nts[i].kind;
    }
    return null;
}

test "`?` on Option[int] in an Option fn types to the payload (int), zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn get(o: Option[int]) -> Option[int] {
        \\ v := o?
        \\ Option.some(v + 1)
        \\}
        \\fn main() -> int { return match get(Option.some(41)) { .some(v) -> v, .none -> 0 } }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Kind.int, tryExprKind(c).?);
}

test "`?` on Result[int,str] in a matching Result fn types to the ok payload (int), zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn use(o: Result[int, str]) -> Result[int, str] {
        \\ v := o?
        \\ Result.ok(v + 1)
        \\}
        \\fn main() -> int { return match use(Result[int, str].ok(41)) { .ok(v) -> v, .err(_) -> 0 } }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Kind.int, tryExprKind(c).?);
}

test "`?` in a fn returning a non-Option/Result is exactly one T0032" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn f(o: Option[int]) -> int {
        \\ v := o?
        \\ return v
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0032, c.result.diags[0].code);
}

test "`?` on a non-Option/Result operand is exactly one T0032" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn f(n: int) -> Option[int] {
        \\ v := n?
        \\ return Option.some(v)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0032, c.result.diags[0].code);
}

test "`?` on an Option inside a Result fn is exactly one T0033 (family mismatch)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn f(o: Option[int]) -> Result[int, str] {
        \\ v := o?
        \\ return Result.ok(v)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0033, c.result.diags[0].code);
}

test "`?` inside a generic Option fn typechecks clean and monomorphizes the instance" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn passthru[T](o: Option[T]) -> Option[T] {
        \\ v := o?
        \\ Option.some(v)
        \\}
        \\fn main() -> int { return match passthru[int](Option.some(41)) { .some(v) -> v, .none -> 0 } }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The `passthru$int` instance is monomorphized (its `?` operand grounds Option[int]).
    var found = false;
    for (c.result.instances) |inst| {
        if (std.mem.indexOf(u8, inst.name.?, "passthru$int") != null) found = true;
    }
    try testing.expect(found);
}

test "`?` on Result[_,E1] in a Result[_,E2] fn is exactly one T0033 (error-type mismatch)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn f(o: Result[int, bool]) -> Result[int, str] {
        \\ v := o?
        \\ return Result.ok(v)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0033, c.result.diags[0].code);
}

test "`?` widens Result error via `impl BigErr has From[SmallErr]` — zero diags, From witness stamped" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum SmallErr { bad }
        \\enum BigErr { small, other }
        \\impl BigErr has From[SmallErr] {
        \\ fn from(s: SmallErr) -> BigErr { BigErr.small }
        \\}
        \\fn inner() -> Result[int, SmallErr] { return Result[int, SmallErr].err(SmallErr.bad) }
        \\fn outer() -> Result[int, BigErr] {
        \\ v := inner()?
        \\ return Result.ok(v)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    // The self-less `from` (no P0006), the From coherence sig (no T0024), and the `?`
    // error-widen (no T0033) all pass — a clean compile.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // `checkCoherence` stamped the `from` witness with the From conformance's
    // `(protocol_id, protocol_args=[SmallErr])`, so the multi-conformance resolver can
    // select it by the operand (source) error type. recv is BigErr, the arg is SmallErr —
    // two DISTINCT concrete enums (the widen direction).
    var found_from = false;
    for (c.result.methods) |m| {
        if (!std.mem.eql(u8, m.name, "from")) continue;
        found_from = true;
        try testing.expect(m.protocol_id != null);
        try testing.expectEqual(Kind.@"enum", m.recv.kind);
        try testing.expectEqual(@as(usize, 1), m.protocol_args.len);
        try testing.expectEqual(Kind.@"enum", m.protocol_args[0].kind);
        try testing.expect(!Type.eql(m.recv, m.protocol_args[0]));
    }
    try testing.expect(found_from);
}

test "`?` on differing Result errors with NO From impl is exactly one T0033" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum SmallErr { bad }
        \\enum BigErr { small, other }
        \\fn f(o: Result[int, SmallErr]) -> Result[int, BigErr] {
        \\ v := o?
        \\ return Result.ok(v)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0033, c.result.diags[0].code);
}

test "multi-conformance From[Src] on one target error type is disambiguated by the operand error type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E1 { a }
        \\enum E2 { b }
        \\enum Big { x, y }
        \\impl Big has From[E1] { fn from(s: E1) -> Big { Big.x } }
        \\impl Big has From[E2] { fn from(s: E2) -> Big { Big.y } }
        \\fn from_e1() -> Result[int, E1] { return Result[int, E1].err(E1.a) }
        \\fn from_e2() -> Result[int, E2] { return Result[int, E2].err(E2.b) }
        \\fn g() -> Result[int, Big] {
        \\ v := from_e1()?
        \\ w := from_e2()?
        \\ return Result.ok(v + w)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    // Two `From[Src]` conformances on `Big` do not collide on coherence (keyed on
    // (protocol, recv, args)); both `?`s widen, each selecting its witness by the operand
    // error type. A clean compile proves the disambiguation.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "a From impl whose `from` returns the WRONG type still trips T0024 (coherence)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum SmallErr { bad }
        \\enum BigErr { small, other }
        \\impl BigErr has From[SmallErr] {
        \\ fn from(s: SmallErr) -> SmallErr { SmallErr.bad }
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    // `from` must return `Self` (BigErr); returning SmallErr is a signature incompatibility.
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0024, c.result.diags[0].code);
}

test "`<`/`>`/`<=`/`>=` on a struct with `impl P has Ord` type to bool, zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Ord {
        \\ fn cmp(self, o: P) -> Ordering {
        \\  if self.x < o.x { Ordering.lt } else if self.x == o.x { Ordering.eq } else { Ordering.gt }
        \\ }
        \\}
        \\fn main() -> int {
        \\ a := P{ x: 1 }
        \\ b := P{ x: 2 }
        \\ r := 0
        \\ if a < b { r = r + 1 }
        \\ if a > b { r = r + 1 }
        \\ if a <= b { r = r + 1 }
        \\ if a >= b { r = r + 1 }
        \\ return r
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Every comparison binary node types to bool (the 4 in main + the `<` inside cmp).
    const nts = c.result.node_types[0];
    var n_cmp: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .binary) continue;
        const tag = c.tokens[n.main_token].tag;
        if (tag == .lt or tag == .lt_eq or tag == .gt or tag == .gt_eq) {
            try testing.expectEqual(Kind.bool, nts[i].kind);
            n_cmp += 1;
        }
    }
    try testing.expect(n_cmp >= 4);
}

test "int/str/bool `<` type to bool (builtin Ord), zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ i := if 1 < 2 { 1 } else { 0 }
        \\ s := if "a" < "b" { 1 } else { 0 }
        \\ b := if false < true { 1 } else { 0 }
        \\ return i + s + b
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "`<` on a struct with no `Ord` impl now DERIVES Ord (was T0027)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ p := P{ x: 1 }
        \\ q := P{ x: 2 }
        \\ return if p < q { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // An all-`Ord`-fields struct with no explicit impl derives structurally — no
    // diagnostic, exactly one `Ord` recipe (`Ord$cmp$s0`), which also fills `(Eq, P)`.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Ord$cmp$s0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.ord, c.result.derives[0].kind);
}

test "struct used with BOTH `<` and `==` yields ONE Ord unit (no double-fire)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\ a := P{ x: 1, y: 0 }
        \\ b := P{ x: 1, y: 5 }
        \\ lt := a < b
        \\ eq := a == b
        \\ return if lt { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // A derived `Ord` fills the single `(Eq, P)` slot, so `==` routes through the derived
    // `cmp` and NO separate `Eq` unit is synthesized — exactly one recipe (the Ord cmp).
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Ord$cmp$s0", c.result.derives[0].name.?);
    for (c.result.derives) |d| try testing.expect(d.kind != .eq);
}

test "explicit `impl P has Ord` OVERRIDES the derive (zero synthetic units)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Ord {
        \\ fn cmp(self, o: P) -> Ordering {
        \\  if self.x < o.x { Ordering.lt } else if self.x == o.x { Ordering.eq } else { Ordering.gt }
        \\ }
        \\}
        \\fn main() -> int {
        \\ a := P{ x: 1 }
        \\ b := P{ x: 2 }
        \\ return if a < b { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "an UNUSED derivable struct emits zero derive recipes" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Q { v: int }
        \\fn main() -> int {
        \\ q := Q{ v: 42 }
        \\ return q.v
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "nested-struct + struct-with-payload-enum-field both derive Ord recursively" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Inner { a: int, b: int }
        \\enum N { Z, S(int) }
        \\struct Outer { p: Inner, n: N }
        \\fn main() -> int {
        \\ x := Outer{ p: Inner{ a: 1, b: 2 }, n: N.Z }
        \\ y := Outer{ p: Inner{ a: 1, b: 3 }, n: N.S(4) }
        \\ return if x < y { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // Outer's Ord fixpoint chases its field `p: Inner` (struct) and `n: N` (payload enum),
    // deriving `Ord` for all three — one recipe each, none is an Eq unit (Ord fills Eq).
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 3), c.result.derives.len);
    for (c.result.derives) |d| try testing.expectEqual(Derive.Kind.ord, d.kind);
}

test "`.hash()` on an all-Hash-fields struct with no impl DERIVES exactly one recipe" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\fn main() -> int {
        \\ a := P{ x: 1, y: 2 }
        \\ return a.hash()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // A direct `.hash()` call records ONE structural Hash derive (struct id 0 -> `Hash$hash$s0`),
    // no diagnostic. Hash is independent of Eq/Ord — the recipe's ret is `int`.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Hash$hash$s0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.hash, c.result.derives[0].kind);
    try testing.expectEqual(Kind.int, c.result.derives[0].ret.kind);
    // A Hash recipe's synthetic method takes ONLY `self` (1 param), unlike homogeneous Eq/Ord.
    try testing.expectEqual(@as(usize, 1), c.result.derives[0].params.len);
}

test "`.hash()` on a PAYLOAD enum derives one Hash recipe (Hash$hash$e0)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { A(int), B }
        \\fn main() -> int {
        \\ return E.A(3).hash()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Hash$hash$e0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.hash, c.result.derives[0].kind);
}

test "`.hash()` recurses through a NESTED aggregate + str field (two recipes)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Name { first: str, last: str }
        \\struct Person { name: Name, age: int }
        \\fn main() -> int {
        \\ a := Person{ name: Name{ first: "a", last: "b" }, age: 1 }
        \\ return a.hash()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // Person's Hash fixpoint chases its `name: Name` field, deriving Hash for both — two
    // recipes, both `.hash` kind. (Name = struct id 0, Person = struct id 1.)
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 2), c.result.derives.len);
    for (c.result.derives) |d| try testing.expectEqual(Derive.Kind.hash, d.kind);
}

test "an UNUSED derivable struct records zero Hash recipes (lazy)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Q { v: int }
        \\fn main() -> int {
        \\ q := Q{ v: 42 }
        \\ return q.v
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "explicit `impl P has Hash` OVERRIDES the derive (zero synthetic units)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Hash {
        \\ fn hash(self) -> int { self.x }
        \\}
        \\fn main() -> int {
        \\ a := P{ x: 1 }
        \\ return a.hash()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The explicit impl wins at `resolveConformanceMethod`, so no structural derive fires.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "Hash and Eq of the SAME struct are INDEPENDENT recipes (no refinement)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ a := P{ x: 1 }
        \\ b := P{ x: 2 }
        \\ return if a == b { 0 } else { a.hash() + b.hash() }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // `==` derives Eq; `.hash()` derives Hash. Hash does NOT fill/refine Eq (unlike Ord),
    // so BOTH recipes exist — one `Eq$eq$s0` and one `Hash$hash$s0`.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 2), c.result.derives.len);
    var saw_eq = false;
    var saw_hash = false;
    for (c.result.derives) |d| {
        if (d.kind == .eq) saw_eq = true;
        if (d.kind == .hash) saw_hash = true;
    }
    try testing.expect(saw_eq);
    try testing.expect(saw_hash);
}

test "firstNonConformingField names the field blocking a `Hash` derive (T0030 substrate)" {
    // T0030's message is built from `firstNonConformingField` (the SAME pid-parameterized
    // substrate T0029 uses). For a pure value-type program every scalar leaf conforms to
    // `Hash`, so this path is not reachable from source; a synthetic table WITHOUT a `str`
    // Hash conformance exercises the exact blocker+field-naming logic the message consumes.
    const gpa = testing.allocator;
    const hash_pid: u32 = 6; // Hash is prelude id 6; the substrate is pid-agnostic regardless.
    const s0_ft = [_]Type{Type.str};
    const s0_fn = [_][]const u8{"name"};
    const structs = [_]StructSym{
        .{ .decl_node = Ast.none, .name = "S0", .field_types = @constCast(&s0_ft), .field_names = @constCast(&s0_fn) },
    };
    const enums = [_]EnumSym{};
    // Register Hash for int/bool/unit but DELIBERATELY not str, so the `name: str` field blocks.
    const confs = [_]Conformance{
        .{ .protocol = hash_pid, .recv = Type.int },
        .{ .protocol = hash_pid, .recv = Type.@"bool" },
        .{ .protocol = hash_pid, .recv = Type.unit },
    };
    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    var co: Composite = .{};
    defer co.deinit(gpa);
    const off = try conform.firstNonConformingField(&structs, &enums, &confs, Type.structT(0), hash_pid, &memo, gpa, &co, &.{});
    try testing.expect(off != null);
    try testing.expectEqualStrings("name", off.?.name);
    try testing.expect(Type.eql(Type.str, off.?.ty));
}

test "`print(P{..})` on an all-Display-fields struct DERIVES exactly one recipe" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\fn main() {
        \\ print(P{ x: 4, y: 2 })
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The reworked `print` accepts a Display arg: it records ONE structural Display derive
    // (struct id 0 -> `Display$display$s0`), no diagnostic, ret unit, `self`-only (1 param).
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Display$display$s0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.display, c.result.derives[0].kind);
    try testing.expectEqual(Kind.unit, c.result.derives[0].ret.kind);
    try testing.expectEqual(@as(usize, 1), c.result.derives[0].params.len);
}

test "`print(enum value)` derives one Display recipe (Display$display$e0)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { A(int), B }
        \\fn main() {
        \\ print(E.A(3))
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Display$display$e0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.display, c.result.derives[0].kind);
}

test "`print` recurses through a NESTED aggregate (two Display recipes)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Point { x: int, y: int }
        \\struct Line { from: Point, to: Point }
        \\fn main() {
        \\ print(Line{ from: Point{ x: 0, y: 0 }, to: Point{ x: 4, y: 2 } })
        \\}
        \\
    );
    defer c.deinit(gpa);
    // Line's Display fixpoint chases its `Point` fields, deriving Display for both — two
    // recipes, both `.display` kind. (Point = struct id 0, Line = struct id 1.)
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 2), c.result.derives.len);
    for (c.result.derives) |d| try testing.expectEqual(Derive.Kind.display, d.kind);
}

test "an UNUSED Display-eligible struct records zero recipes (lazy)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Q { v: int }
        \\fn main() {
        \\ q := Q{ v: 42 }
        \\ print(q.v)
        \\}
        \\
    );
    defer c.deinit(gpa);
    // `print(q.v)` displays an int (a prelude conformance, no derive); `Q` itself is never
    // printed, so no `Display$display$s*` unit is synthesized.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "explicit `impl P has Display` OVERRIDES the derive (zero synthetic units)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Display {
        \\ fn display(self) { print("x") }
        \\}
        \\fn main() {
        \\ print(P{ x: 1 })
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The explicit impl wins at `resolveConformanceMethod`/`findConformance`, so no structural
    // derive fires.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "`print(\"..\")` still types clean and derives nothing (str path unchanged)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() {
        \\ print("hi")
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "`print('A')` derives ONE shared char Display witness (UTF-8 encoder, not inlined)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() {
        \\ print('A')
        \\}
        \\
    );
    defer c.deinit(gpa);
    // char no longer carries a bare Display row, so displaying it records a normal structural
    // Display derive: ONE `Display$display$s<charId>` recipe whose emitter is overridden to the
    // UTF-8 encoder. Every char-display site CALLs it (vs the former per-site inline encoder
    // that overflowed the frame past ~6 displays). Suppression ("A" not "char(65)") is upheld
    // by the overriding emitter — guarded end-to-end by examples/io/char_suppress.toy.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqual(Derive.Kind.display, c.result.derives[0].kind);
    try testing.expect(c.result.derives[0].conform_ty.kind == .@"struct");
    try testing.expectEqual(c.result.char_struct.?, c.result.derives[0].conform_ty.struct_id);
}

test "Ord refines Eq — `==` on an Ord-only struct types to bool, zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Ord {
        \\ fn cmp(self, o: P) -> Ordering {
        \\  if self.x < o.x { Ordering.lt } else if self.x == o.x { Ordering.eq } else { Ordering.gt }
        \\ }
        \\}
        \\fn main() -> int {
        \\ a := P{ x: 1 }
        \\ b := P{ x: 1 }
        \\ return if a == b { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const nts = c.result.node_types[0];
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .binary and c.tokens[n.main_token].tag == .eq_eq) {
            try testing.expectEqual(Kind.bool, nts[i].kind);
            found = true;
        }
    }
    try testing.expect(found);
}

test "a bounded generic body `fn lt2[T has Ord](a: T, b: T) -> bool { a < b }` checks once" {
    const gpa = testing.allocator;
    // A bounded template body IS checked (bound-as-axiom); `T has Ord` gives `T` the `Ord`
    // axiom, so `a < b` types clean. Exercises the `conformsToOrd` type_var branch.
    var c = try checkSource(
        \\fn lt2[T has Ord](a: T, b: T) -> bool { a < b }
        \\fn main() -> int {
        \\ return if lt2[int](1, 2) { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "`<` in a body bounded by a NON-Ord protocol is T0027 (bound is not the Ord axiom)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Doubler { fn dbl(self) -> int }
        \\fn bad[T has Doubler](a: T, b: T) -> bool { a < b }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    var n27: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0027) n27 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n27);
}

test "explicit `impl P has Eq` alongside `impl P has Ord` yields no T0020 (explicit authoritative)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\impl P has Ord {
        \\ fn cmp(self, o: P) -> Ordering {
        \\  if self.x < o.x { Ordering.lt } else if self.x == o.x { Ordering.eq } else { Ordering.gt }
        \\ }
        \\}
        \\fn main() -> int {
        \\ a := P{ x: 1 }
        \\ b := P{ x: 2 }
        \\ return if a < b { if a == a { 42 } else { 0 } } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "`+`/`-`/`*`/`/` on a struct with the matching impl type to the operand type, zero diags" {
    const gpa = testing.allocator;
    // 0 diags proves each `(Add/Sub/Mul/Div, V2)` conformance resolves (the operator would
    // otherwise be T0028); every top-level arithmetic binary types to the operand struct
    // type (Out=Self), so the value is usable as a V2.
    var c = try checkSource(
        \\struct V2 { x: int, y: int }
        \\impl V2 has Add { fn add(self, o: V2) -> V2 { V2{ x: self.x + o.x, y: self.y + o.y } } }
        \\impl V2 has Sub { fn sub(self, o: V2) -> V2 { V2{ x: self.x - o.x, y: self.y - o.y } } }
        \\impl V2 has Mul { fn mul(self, o: V2) -> V2 { V2{ x: self.x * o.x, y: self.y * o.y } } }
        \\impl V2 has Div { fn div(self, o: V2) -> V2 { V2{ x: self.x / o.x, y: self.y / o.y } } }
        \\fn use_all(p: V2, q: V2) -> V2 { p + q - p * q / p }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const nts = c.result.node_types[0];
    var n_arith: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .binary) continue;
        const tag = c.tokens[n.main_token].tag;
        if ((tag == .plus or tag == .minus or tag == .star or tag == .slash) and nts[i].kind == .@"struct") n_arith += 1;
    }
    try testing.expect(n_arith >= 4);
}

test "int `+`/`-`/`*`/`/` type to int (builtin, inline), zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int { return 1 + 2 - 3 * 4 / 5 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "`+` on a struct with no `Add` impl is exactly one T0028 at the operator" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct V2 { x: int, y: int }
        \\fn main() -> int {
        \\ a := V2{ x: 1, y: 2 }
        \\ b := V2{ x: 3, y: 4 }
        \\ s := a + b
        \\ return s.x
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0028, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "requires an 'Add' impl") != null);
}

test "`str + str` is T0028 (no builtin Add for str — never a silent allocation)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int { s := "a" + "b"
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    var n28: usize = 0;
    for (c.result.diags) |d| if (d.code == codes.Code.T0028) {
        n28 += 1;
        try testing.expect(std.mem.indexOf(u8, d.message, "requires an 'Add' impl for type 'str'") != null);
    };
    try testing.expectEqual(@as(usize, 1), n28);
}

test "`bool + bool` is T0028 (no builtin Add for bool)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn f(a: bool, b: bool) -> bool { a + b }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    var n28: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0028) n28 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n28);
}

test "a cross-type `+` (int + str) stays a homogeneity error, never T0028" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := 1 + "a"
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expect(c.result.diags[0].code != codes.Code.T0028);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "same type") != null);
}

test "a user `impl P has Add` whose method returns non-Self is T0024 (Out=Self enforced)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Add { fn add(self, o: P) -> int { self.x + o.x } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0024, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "add") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "Add") != null);
}

test "a user `impl int has (user protocol)` records a conformance and dispatches, no error" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Dbl { fn dbl(self) -> int }
        \\impl int has Dbl { fn dbl(self) -> int { self } }
        \\fn main() -> int {
        \\ a := 21
        \\ return a.dbl() + a.dbl()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // A user scalar impl registers a REAL method (recv = int) — the widened dispatch
    // guard routes `a.dbl()` through the normal fn_id Method path (no recognizer).
    try testing.expectEqual(@as(usize, 1), c.result.methods.len);
    try testing.expectEqualStrings("dbl", c.result.methods[0].name);
    try testing.expectEqual(Kind.int, c.result.methods[0].recv.kind);
}

test "the prelude `Eq` is nameable by BARE name with no import (user struct conformance)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct S { x: int }
        \\impl S has Eq { fn eq(self, o: S) -> bool { self.x == o.x } }
        \\fn main() -> int {
        \\ a := S{ x: 21 }
        \\ b := S{ x: 21 }
        \\ return if a.eq(b) { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    // If the prelude fallback failed, bare `Eq` would be undeclared -> T0021; zero diags
    // proves it resolves with no import.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.methods.len);
}

test "builtin `eq` arity / arg-type mismatch and unknown scalar method errors" {
    const gpa = testing.allocator;
    {
        // Arity: `a.eq()` wants exactly one argument.
        var c = try checkSource("fn main() -> int {\n a := 1\n return if a.eq() { 1 } else { 0 }\n}\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "argument") != null);
    }
    {
        // Arg-type: the homogeneous `eq` arg must be assignable to the receiver (int).
        var c = try checkSource("struct S { x: int }\nfn main() -> int {\n a := 1\n s := S{ x: 0 }\n return if a.eq(s) { 1 } else { 0 }\n}\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "expected int") != null);
    }
    {
        // Unknown method on a scalar is a clean T0018 (was a raw non-coded diagnostic).
        var c = try checkSource("fn main() -> int {\n a := 1\n return if a.foo() { 1 } else { 0 }\n}\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
    }
}

test "a bounded generic compiles ONCE against the bound and monomorphizes over a conforming type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Doubler { fn dbl(self) -> int }
        \\struct P { x: int }
        \\impl P has Doubler { fn dbl(self) -> int { self.x * 2 } }
        \\fn twice[T has Doubler](v: T) -> int { v.dbl() + v.dbl() }
        \\fn main() -> int { return twice(P{ x: 21 }) / 2 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Exactly one instance twice$P, grounded to a concrete return type (not a type_var).
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqual(Kind.int, c.result.instances[0].ret.kind);
    try testing.expect(!c.result.instances[0].params[0].isTypeVar());
    // The instance carries its resolved Doubler conformance (fp fold input).
    try testing.expectEqual(@as(usize, 1), c.result.instances[0].conformances.len);
    try testing.expectEqualStrings("Doubler", c.result.instances[0].conformances[0].protocol_name);
    try testing.expectEqual(@as(usize, 1), c.result.instances[0].conformances[0].witness_syms.len);
    try testing.expectEqualStrings("main.P.dbl$Doubler", c.result.instances[0].conformances[0].witness_syms[0]);
}

test "reify grounds a bounded instance's conform_ty when the type-arg is a generic-aggregate App" {
    const gpa = testing.allocator;
    // `Box[int]` structurally satisfies the `Eq` bound (its int field is Eq), so `eq2` is
    // instantiated with `conform_ty = App(Box,[int])`. After reify, the conform_ty copy
    // must be grounded to the reified struct in lockstep with `inst.args[0]`, never left an
    // un-reified App.
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\fn eq2[T has Eq](a: T, b: T) -> bool { a == b }
        \\fn main() -> int {
        \\ b := Box[int]{ v: 1 }
        \\ d := Box[int]{ v: 2 }
        \\ return if eq2(b, d) { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    const inst = c.result.instances[0];
    try testing.expectEqual(@as(usize, 1), inst.conformances.len);
    try testing.expect(inst.conformances[0].conform_ty.kind != .app);
    try testing.expect(Type.eql(inst.args[0], inst.conformances[0].conform_ty));
}

test "a non-conforming type at a bounded call is a use-site T0023 and skips the instance" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Doubler { fn dbl(self) -> int }
        \\struct P { x: int }
        \\impl P has Doubler { fn dbl(self) -> int { self.x * 2 } }
        \\struct Q { x: int }
        \\fn twice[T has Doubler](v: T) -> int { v.dbl() + v.dbl() }
        \\fn main() -> int { return twice(Q{ x: 1 }) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0023, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "Q") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "Doubler") != null);
    // The non-conforming instantiation is SKIPPED (no twice$Q instance minted).
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
}

test "a conforming impl whose method signature diverges from the protocol is T0024" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Doubler { fn dbl(self) -> int }
        \\struct P { x: int }
        \\impl P has Doubler { fn dbl(self) -> bool { self.x > 0 } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0024, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "dbl") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "Doubler") != null);
}

test "calling a method NOT in the bound protocol from a bounded body is T0018 (bound-as-axiom)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Doubler { fn dbl(self) -> int }
        \\struct P { x: int }
        \\impl P has Doubler { fn dbl(self) -> int { self.x * 2 } }
        \\fn bad[T has Doubler](v: T) -> int { v.nope() }
        \\fn main() -> int { return bad(P{ x: 1 }) }
        \\
    );
    defer c.deinit(gpa);
    // The bound-as-axiom body check reports `nope` once at the definition (T0018); the
    // template is POISONED so the `bad(P{..})` call does NOT re-cascade per instance.
    try testing.expectEqual(codes.Code.T0018, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "nope") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "Doubler") != null);
    var t0018_count: usize = 0;
    for (c.result.diags) |d| if (d.code == codes.Code.T0018) {
        t0018_count += 1;
    };
    try testing.expectEqual(@as(usize, 1), t0018_count); // reported ONCE, not per instance
    try testing.expectEqual(@as(usize, 0), c.result.instances.len); // poisoned: not instantiated
}

test "an undeclared bound protocol on a generic param is T0021 at the bound ref" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn twice[T has Ghost](v: T) -> int { 0 }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(codes.Code.T0021, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "Ghost") != null);
}

test "a doubly-conforming generic protocol registers TWO methods with NO T0020" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Into[U] { fn into(self) -> U }
        \\struct P { x: int }
        \\impl P has Into[int] { fn into(self) -> int { self.x } }
        \\impl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    // Two conformances keyed on (protocol, P, [int]) vs (protocol, P, [bool]) — distinct
    // keys, so NO coherence collision.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Both `into` methods register on P, stamped with distinct protocol_args.
    var into_on_p: usize = 0;
    var saw_int = false;
    var saw_bool = false;
    for (c.result.methods) |m| {
        if (std.mem.eql(u8, m.name, "into") and m.recv.kind == .@"struct") {
            into_on_p += 1;
            try testing.expect(m.protocol_id != null);
            try testing.expectEqual(@as(usize, 1), m.protocol_args.len);
            if (m.protocol_args[0].kind == .int) saw_int = true;
            if (m.protocol_args[0].kind == .bool) saw_bool = true;
        }
    }
    try testing.expectEqual(@as(usize, 2), into_on_p);
    try testing.expect(saw_int and saw_bool);
}

test "a concrete doubly-conforming use with no type-args emits exactly one T0025" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Into[U] { fn into(self) -> U }
        \\struct P { x: int }
        \\impl P has Into[int] { fn into(self) -> int { self.x } }
        \\impl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }
        \\fn main() -> int {
        \\ p := P{ x: 42 }
        \\ return p.into()
        \\}
        \\
    );
    defer c.deinit(gpa);
    var n25: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0025) n25 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n25);
    // Names the conflicting conformances deterministically, in source order.
    var msg: []const u8 = "";
    for (c.result.diags) |d| if (d.code == codes.Code.T0025) {
        msg = d.message;
    };
    try testing.expect(std.mem.indexOf(u8, msg, "ambiguous conformance") != null);
    const int_at = std.mem.indexOf(u8, msg, "Into[int]") orelse return error.TestUnexpectedResult;
    const bool_at = std.mem.indexOf(u8, msg, "Into[bool]") orelse return error.TestUnexpectedResult;
    try testing.expect(int_at < bool_at); // source order (Into[int] declared first)
}

test "an explicit-args concrete use v.into[int]() selects the Into[int] witness, no error" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Into[U] { fn into(self) -> U }
        \\struct P { x: int }
        \\impl P has Into[int] { fn into(self) -> int { self.x } }
        \\impl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }
        \\fn main() -> int {
        \\ p := P{ x: 42 }
        \\ return p.into[int]()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "the generic-protocol e2e monomorphizes one bounded instance carrying its protocol_args" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Into[U] { fn into(self) -> U }
        \\struct P { x: int }
        \\impl P has Into[int] { fn into(self) -> int { self.x } }
        \\impl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }
        \\fn use[T has Into[int]](v: T) -> int { v.into[int]() }
        \\fn main() -> int { return use(P{ x: 42 }) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    const inst = c.result.instances[0];
    try testing.expectEqual(@as(usize, 1), inst.conformances.len);
    try testing.expectEqualStrings("Into", inst.conformances[0].protocol_name);
    // The bound's protocol type-args ride the conformance (folded into the (e) fp).
    try testing.expectEqual(@as(usize, 1), inst.conformances[0].protocol_args.len);
    try testing.expectEqual(Kind.int, inst.conformances[0].protocol_args[0].kind);
    // The witness is the Into[int] method (mangled with the protocol identity + args),
    // NOT the Into[bool] sibling.
    try testing.expect(std.mem.indexOf(u8, inst.conformances[0].witness_syms[0], "into") != null);
    try testing.expect(std.mem.indexOf(u8, inst.conformances[0].witness_syms[0], "int") != null);
}

test "a single generic conformance resolves with no explicit args (byte-identical dispatch)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Into[U] { fn into(self) -> U }
        \\struct P { x: int }
        \\impl P has Into[int] { fn into(self) -> int { self.x } }
        \\fn main() -> int {
        \\ p := P{ x: 42 }
        \\ return p.into()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // With ONE conformance the resolver's `matches <= 1` fast path fires — `p.into()`
    // resolves with no args, no ambiguity.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "a [T has Convert[U]] bound resolves after substituting U at the mono worklist" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Convert[U] { fn conv(self) -> U }
        \\struct P { x: int }
        \\impl P has Convert[int] { fn conv(self) -> int { self.x } }
        \\fn use[T has Convert[U], U](v: T) -> int { 0 }
        \\fn main() -> int { return use[P, int](P{ x: 42 }) }
        \\
    );
    defer c.deinit(gpa);
    // The bound arg `U` is a `type_var`; at the `use[P, int]` call it substitutes to `int`,
    // so `findConformance(Convert, P, [int])` succeeds at the mono worklist — one instance,
    // no diagnostics. (The bound is resolved at monomorphization; the body need not call
    // `conv`, matching the contract's in-scope `[T has Convert[U]]` resolution.)
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances[0].conformances.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances[0].conformances[0].protocol_args.len);
    try testing.expectEqual(Kind.int, c.result.instances[0].conformances[0].protocol_args[0].kind);
}

test "Box[int] monomorphizes to a reified 1-int concrete struct (size 8)" {
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

test "Box[int] and Box[bool] reify to TWO distinct concrete layouts" {
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

test "unbounded f[T]->f[Box[T]] is rejected with T0017 and TERMINATES (no hang)" {
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

test "a non-generic struct with a concrete generic-struct field compiles + reifies" {
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

test "a non-generic enum with a concrete generic-struct payload reifies (no App survives the enum snapshot)" {
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

test "a legitimately deep-but-finite generic-struct nest still compiles" {
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

test "a generic struct with a generic-struct field reifies correctly with the wrapper declared FIRST" {
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

test "Box{v:1} infers Box[int] and dedups with explicit Box[int] to ONE reified struct" {
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
    // No `.app`/`.type_var` survives any node_types slot (the reification invariant).
    for (c.result.node_types) |mnt| for (mnt) |ty| try testing.expect(ty.kind != .app and ty.kind != .type_var);
}

test "an inferred Box{v:true} reifies a distinct Box$bool layout" {
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

test "conflicting inferred field types report exactly one T0015" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Pair[T] { a: T, b: T }\nfn main() -> int {\n p := Pair{ a: 1, b: true }\n return 0\n}\n");
    defer c.deinit(gpa);
    var n15: usize = 0;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0015) n15 += 1;
    }
    try testing.expectEqual(@as(usize, 1), n15);
}

test "a phantom (uninferable) struct type-param routes to T0016" {
    const gpa = testing.allocator;
    var c = try checkSource("struct P[T] { x: int }\nfn main() -> int {\n p := P{ x: 1 }\n return 0\n}\n");
    defer c.deinit(gpa);
    var saw16 = false;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0016) saw16 = true;
    }
    try testing.expect(saw16);
}

test "missing-field-with-inference still infers T then reports the missing field" {
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

test "a unit-typed inferred field value is gated with T0013 before internApp" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn nop() {}\nfn main() -> int {\n c := Box{ v: nop() }\n return 0\n}\n");
    defer c.deinit(gpa);
    var saw13 = false;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0013) saw13 = true;
    }
    try testing.expect(saw13);
}

test "the Either e2e typechecks clean and reifies exactly one concrete enum" {
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

test "a reified enum's payload offsets/size match a hand-written non-generic twin" {
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

test "Opt.none with no target reports exactly one T0016 (deterministic)" {
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

test "a construction payload-type mismatch reports one clean error (proves L->int subst)" {
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

test "an inferred construction Wrap.w(5) dedups with explicit Wrap[int].w(5)" {
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

test "a non-generic struct field of a generic-enum instance reifies + rewrites away" {
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

test "unbounded generic-enum type-growth is rejected with T0017 and TERMINATES" {
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

test "a bare (no-explicit-args) generic call infers its type-arg from the argument" {
    const gpa = testing.allocator;
    var c = try checkSource("fn id[T](x: T) -> T { x }\nfn main() -> int { return id(7) }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // `id(7)` infers T=int and monomorphizes to exactly one instance, same as id[int].
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.id$int", c.result.instances[0].name.?);
    try testing.expect(!c.result.instances[0].ret.isTypeVar());
}

test "nested bare inference (snd(true, id(42))) infers all type-args; args are concrete" {
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
        if (std.mem.startsWith(u8, inst.name.?, "main.snd$")) {
            saw_snd = true;
            try testing.expectEqualStrings("main.snd$bool$int", inst.name.?);
            try testing.expectEqual(@as(usize, 2), inst.params.len);
            try testing.expectEqual(Kind.bool, inst.params[0].kind);
            try testing.expectEqual(Kind.int, inst.params[1].kind);
            try testing.expectEqual(Kind.int, inst.ret.kind);
        }
    }
    try testing.expect(saw_snd);
}

test "an inferred call and its explicit form DEDUP to ONE instance (byte-for-byte parity)" {
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
        if (std.mem.startsWith(u8, inst.name.?, "main.snd$")) snd_count += 1;
    }
    try testing.expectEqual(@as(usize, 1), snd_count);
}

test "a diverging (never) argument still infers the var from another concrete arg" {
    const gpa = testing.allocator;
    // same[T](a:T,b:T): never at 0 binds nothing; int at 1 binds T=int. Check-only
    // (a break-less loop cannot run to an exit code).
    var c = try checkSource("fn same[T](a: T, b: T) -> T { b }\nfn main() -> int { return same(loop {}, 42) }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.same$int", c.result.instances[0].name.?);
}

test "a conflicting bare call reports T0015 naming BOTH argument spans; mints no instance" {
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

test "a return-only generic call reports T0016 (explicit args required)" {
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

test "explicit type args still override inference and satisfy an otherwise-uninferable call" {
    const gpa = testing.allocator;
    // The return-only `ro[T]` is uninferable bare, but explicit `ro[int]()` compiles.
    var c = try checkSource("fn ro[T]() -> T { loop {} }\nfn main() -> int { return ro[int]() }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.ro$int", c.result.instances[0].name.?);
}

test "an uncalled generic fn mints ZERO instances (free in the binary)" {
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

test "print of an int typechecks clean (Display, no longer an arg-type error)" {
    // `print` formerly only took `str`, so `print(42)` was `argument 1: expected str, got int`.
    // `print` is now a polymorphic Display-accepting builtin: `int` conforms to the
    // prelude `Display`, so `print(42)` is clean (no diagnostic) and derives nothing (int has
    // a prelude Display conformance, so `findConformance` wins over the structural path).
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n print(42)\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
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

test "builtinConvMethod verdict table (widen/narrow/reject)" {
    // into: same-sign widening/identity accepted, narrowing/sign-change rejected.
    try testing.expectEqual(ConvKind.widen, builtinConvMethod(Type.uint8, Type.uint, "into", null).?.kind);
    try testing.expectEqual(Type.uint, builtinConvMethod(Type.uint8, Type.uint, "into", null).?.target);
    try testing.expectEqual(ConvKind.widen, builtinConvMethod(Type.int, Type.int64, "into", null).?.kind); // plat==w64
    try testing.expect(builtinConvMethod(Type.uint, Type.uint8, "into", null) == null); // narrowing
    try testing.expect(builtinConvMethod(Type.uint8, Type.int, "into", null) == null); // sign-change
    // try_into: any int→int accepted (target = expected payload T).
    try testing.expectEqual(ConvKind.narrow, builtinConvMethod(Type.uint, Type.uint8, "try_into", null).?.kind);
    try testing.expectEqual(Type.uint8, builtinConvMethod(Type.uint, Type.uint8, "try_into", null).?.target);
    try testing.expectEqual(ConvKind.narrow, builtinConvMethod(Type.int, Type.uint, "try_into", null).?.kind);
    // Non-integer receiver / expected → null (leaves struct/enum dispatch untouched).
    try testing.expect(builtinConvMethod(Type.bool, Type.uint8, "try_into", null) == null);
    try testing.expect(builtinConvMethod(Type.uint8, Type.bool, "into", null) == null);
}

test "builtinConvMethod char verdicts (char↔int/byte)" {
    const ch: u32 = 7; // any struct id stands in for `char`
    const char_ty = Type.structT(ch);
    // char -> int (into): lossless into any int ≥ 32 bits, either sign.
    try testing.expectEqual(ConvKind.char_to_int, builtinConvMethod(char_ty, Type.int, "into", ch).?.kind);
    try testing.expectEqual(ConvKind.char_to_int, builtinConvMethod(char_ty, Type.uint32, "into", ch).?.kind);
    try testing.expect(builtinConvMethod(char_ty, Type.int16, "into", ch) == null); // too narrow
    // byte -> char (into): only a `uint8` source is always a valid scalar.
    try testing.expectEqual(ConvKind.byte_to_char, builtinConvMethod(Type.uint8, char_ty, "into", ch).?.kind);
    try testing.expectEqual(char_ty, builtinConvMethod(Type.uint8, char_ty, "into", ch).?.target);
    try testing.expect(builtinConvMethod(Type.uint16, char_ty, "into", ch) == null); // uint16 can be a surrogate
    // int -> char (try_into) + char -> byte (try_into): the fallible pair.
    try testing.expectEqual(ConvKind.int_to_char, builtinConvMethod(Type.int, char_ty, "try_into", ch).?.kind);
    try testing.expectEqual(ConvKind.char_to_byte, builtinConvMethod(char_ty, Type.uint8, "try_into", ch).?.kind);
    // char is inert when no `char_id` is known (a prelude-less caller).
    try testing.expect(builtinConvMethod(char_ty, Type.int, "into", null) == null);
}

test "witnessProtocolId resolves into/try_into to the bundle ids" {
    const ids: PreludeProtocolIds = .{ .into = 9, .try_into = 10 };
    try testing.expectEqual(@as(?u32, 9), witnessProtocolId(ids, "into"));
    try testing.expectEqual(@as(?u32, 10), witnessProtocolId(ids, "try_into"));
}

test "chained `wide.try_into().unwrap()` leaks the int target to try_into" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\  wide: uint = 200
        \\  back: uint8 = wide.try_into().unwrap()
        \\  return if back == 200 { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);

    var saw_try_into = false;
    var saw_unwrap = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .call) continue;
        const callee = c.tree.nodes[n.lhs.int()];
        if (callee.tag != .field_access) continue;
        const member = c.tokens[callee.main_token].text(c.source);
        const ty = c.result.node_types[0][i];
        if (std.mem.eql(u8, member, "try_into")) {
            saw_try_into = true;
            // Result[uint8, ConvErr] — an aggregate, never a bare scalar/invalid.
            try testing.expect(ty.kind == .@"enum" or ty.kind == .app);
        } else if (std.mem.eql(u8, member, "unwrap")) {
            saw_unwrap = true;
            try testing.expectEqual(Type.uint8, ty);
        }
    }
    try testing.expect(saw_try_into and saw_unwrap);
}

test "narrowing via `into` is rejected (no silent lossy conversion)" {
    const gpa = testing.allocator;
    // uint -> uint8 is narrowing, so `into` (lossless-only) must NOT resolve: the recognizer
    // returns null and the call falls through to T0018 rather than silently truncating.
    var c = try checkSource(
        \\fn main() -> int {
        \\  wide: uint = 300
        \\  x: uint8 = wide.into()
        \\  return if x == 44 { 0 } else { 1 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len > 0);
}
