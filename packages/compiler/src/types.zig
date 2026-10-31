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
const builtin = @import("builtin");
const Token = @import("ast/Token.zig").Token;
const Ast = @import("ast/Ast.zig");
const Resolve = @import("resolve.zig");
const Resolution = @import("symbols/Resolution.zig").Resolution;
const symbols_res = @import("symbols/Resolution.zig");
const Sig = @import("symbols/Sig.zig").Sig;
pub const Callee = @import("symbols/Callee.zig").Callee;
const Closure = @import("symbols/Closure.zig");
const Modes = @import("symbols/Modes.zig");
const Escape = @import("symbols/Escape.zig");
const Body = @import("symbols/Body.zig");
const symbols = @import("symbols/Sym.zig");
const Mono = @import("symbols/Mono.zig");
const Mangle = @import("symbols/Mangle.zig");
const Derive = @import("symbols/Derive.zig");
pub const DeriveRecipe = Derive.Derive;

/// One type demanded by a `descriptor_of[T]()` site, resolved after synthesis. `hashable`
/// records whether `T` conforms to `Hash`+`Eq` (so its descriptor carries erased
/// hash/eq references, else those slots are null/absent). PODs — the ty is a reified value.
pub const DescType = struct { ty: Type, hashable: bool };

/// An erased key witness to synthesize for a descriptor'd `Hashable` type: the key type,
/// which witness (`hash`/`eq`), the owning module sentinel, and the minted mangled name
/// (`descriptor$hash$…`/`descriptor$eq$…`, owned — freed at teardown / by `GraphResult`).
pub const ErasedUnit = struct { ty: Type, kind: Derive.ErasedKind, mod: u32 = 0, name: ?[]u8 = null };
const Composite = @import("symbols/Composite.zig");
const Infer = @import("symbols/Infer.zig");
const derive_synth = @import("types/derive_synth.zig");
pub const coherence = @import("types/coherence.zig");
const reify = @import("types/reify.zig");
const register = @import("symbols/register.zig");
const prelude_reg = @import("symbols/Prelude.zig");
const StdNames = @import("symbols/StdNames.zig");
const nearmiss = @import("diagnostics/nearmiss.zig");
const Engine = @import("query/Engine.zig");
const Walk = @import("ast/Walk.zig");
const ClosureTables = @import("types/closure_tables.zig");
const Span = @import("diagnostics/model.zig").Span;
const Io = std.Io;

const Typecheck = @This();

/// The type algebra + layout engine (the `Type`/`Kind`/`Layout` value types and
/// the struct/enum layout-cycle state machine) live in their own deep module. The
/// checker drives it via `LayoutEngine.layoutStruct`/`layoutEnum` (id-ordered, from
/// `checkGraph`/`runGraph`) and consumes the laid syms through the re-exported aliases.
const LayoutEngine = @import("layout/Engine.zig");
const Abi = @import("codegen/abi/Abi.zig");
const FrameLayout = @import("codegen/frame/FrameLayout.zig");

// Re-export the algebra/layout value types so every downstream importer keeps
// reading `Typecheck.Type`/`.Layout`/`.EnumLayout` etc. unchanged.
pub const Kind = @import("layout/Type.zig").Kind;
pub const Type = @import("layout/Type.zig").Type;
pub const IntDesc = @import("layout/Type.zig").IntDesc;
pub const Layout = LayoutEngine.Layout;
pub const isRefStruct = LayoutEngine.isRefStruct;
pub const somePayload = LayoutEngine.somePayload;
pub const errPayload = LayoutEngine.errPayload;
pub const payloadAddr = LayoutEngine.payloadAddr;
pub const VariantForm = LayoutEngine.VariantForm;
pub const VariantLayout = LayoutEngine.VariantLayout;
pub const EnumLayout = LayoutEngine.EnumLayout;

// Internal aliases for the checker's own scratch tables. These ARE the engine's
// table types (the `Model` aliases them; `BodyChecker` reads `.state`/`.poisoned`/
// `.field_*`/`.variants` through them at fingerprint + exhaustiveness time).
const StructSym = LayoutEngine.StructSym;
const VariantSym = LayoutEngine.VariantSym;
const EnumSym = LayoutEngine.EnumSym;

pub const type_names = @import("layout/Type.zig").builtin_names;

/// Shared type-reference resolution + token/diagnostic helpers, generic over the
/// checker (`*Typecheck` for Pass-A or `*BodyChecker` for Pass-C). Both expose the
/// same cursor (`tree`/`tokens`/`source`/`graph_mod`/`sink`) and table accessors
/// (`graphCtx`/`structSyms`/`enumSyms`/`activeStructMap`/`activeEnumMap`), so this
/// is the single home for the logic AND its diagnostic strings — the two passes
/// can't drift (diagnostic-string drift is otherwise gated only by the corpus harness).
pub const refs = struct {
    const err_unknown_type = "unknown type '{s}'";
    const err_unknown_module = "unknown module '{s}'";
    const err_module_no_type = "module has no type '{s}'";

    /// Near-miss candidate iterator over every type NAME visible at a use site: the
    /// builtin type keywords, then the active module's struct / enum / alias keys. Feeds
    /// `nearmiss.suggest` for the "did you mean" enrichment of the unknown-type diagnostic.
    /// `keyIterator` takes `*const Self`, so the same `KeyIterator` type serves both the
    /// mutable (Pass-A) and const (Pass-C) map accessors — one iterator struct for both.
    const TypeNameIter = struct {
        builtins: []const []const u8,
        bi: usize = 0,
        structs: std.StringHashMapUnmanaged(u32).KeyIterator,
        enums: std.StringHashMapUnmanaged(u32).KeyIterator,
        aliases: std.StringHashMapUnmanaged(Type).KeyIterator,
        pub fn next(self: *TypeNameIter) ?[]const u8 {
            if (self.bi < self.builtins.len) {
                defer self.bi += 1;
                return self.builtins[self.bi];
            }
            if (self.structs.next()) |k| return k.*;
            if (self.enums.next()) |k| return k.*;
            if (self.aliases.next()) |k| return k.*;
            return null;
        }
    };

    fn typeNameIter(self: anytype) TypeNameIter {
        return .{
            .builtins = type_names.keys(),
            .structs = self.activeStructMap().keyIterator(),
            .enums = self.activeEnumMap().keyIterator(),
            .aliases = self.activeAliasMap().keyIterator(),
        };
    }

    pub fn nameText(self: anytype, tok: u32) []const u8 {
        return self.tokens[tok].text(self.source);
    }

    pub fn byteOf(self: anytype, tok: u32) u32 {
        return self.tokens[tok].start;
    }

    /// The single token `tok`: what a diagnostic about a name spans.
    pub fn tokSpan(self: anytype, tok: u32) Span {
        return .{ .start = self.tokens[tok].start, .end = self.tokens[tok].end };
    }

    /// The source extent of the subtree at `node` in the active module.
    pub fn spanOf(self: anytype, node: Ast.Index) Span {
        return Walk.nodeSpan(.{ .tree = self.tree, .tokens = self.tokens, .source = self.source }, node);
    }

    /// Human-readable name of a type (a struct/enum's declared name, else its kind tag).
    pub fn typeName(self: anytype, ty: Type) []const u8 {
        if (ty.kind == .@"struct" and ty.struct_id < self.structSyms().len)
            return self.structSyms()[ty.struct_id].name;
        if (ty.kind == .@"enum" and ty.enum_id < self.enumSyms().len)
            return self.enumSyms()[ty.enum_id].name;
        // A composite `App` (`Box[int]` / `Either[int,bool]`) renders as its generic
        // ctor's name (`Box` / `Either`): this borrowing accessor cannot allocate, so a
        // message spells the args through `writeType`. The ctor id space is
        // struct-vs-enum disambiguated by `ctor_is_enum`, so render an enum-App off
        // `enumSyms`, a struct-App off `structSyms`.
        if (ty.kind == .app) {
            const e = self.composite.at(ty.appIdx());
            if (e.ctor_is_enum) {
                if (e.ctor < self.enumSyms().len) return self.enumSyms()[e.ctor].name;
            } else {
                if (e.ctor < self.structSyms().len) return self.structSyms()[e.ctor].name;
            }
        }
        if (ty.kind == .func) return self.composite.at(ty.funcIdx()).name;
        // An integer type renders its sign/width spelling (`int8`/`uint`/…), not the
        // bare `int` kind tag — the single source both passes share.
        if (ty.kind == .int) return ty.intName();
        return @tagName(ty.kind);
    }

    /// Map a type-reference node (an `identifier`, a `literal_unit` for `()`, or in
    /// graph mode a qualified `mod.Type` `field_access`) to a `Type`.
    pub fn typeFromNode(self: anytype, type_node: Ast.Index) Type {
        return refs.typeFromNodeAt(self, type_node, .value);
    }

    /// A declared return type (a fn, method, extern, protocol method, local fn, lambda or
    /// fn type); an omitted `-> R` is unit.
    pub fn retTypeFromNode(self: anytype, type_node: Ast.Index) Type {
        return refs.typeFromNodeAt(self, type_node, .ret);
    }

    /// `never` is spellable only in `.ret` position: a value of it cannot exist, so a
    /// param, field or binding of it would be dead weight that layout and codegen would
    /// have to size. It resolves after every user/imported name so a user type `never`
    /// still wins, as with every other builtin-ish name.
    fn typeFromNodeAt(self: anytype, type_node: Ast.Index, pos: enum { value, ret }) Type {
        if (type_node == Ast.none) return Type.unit;
        const tn = self.tree.nodes[type_node.int()];
        if (tn.tag == .literal_unit) return Type.unit; // explicit `-> ()` / `p: ()`
        // A type application `Box[int]` (in type position or a struct-construction lhs)
        // resolves to a composite `App`: the base is a generic struct ctor, each
        // arg is recursively resolved (subst-aware), interned to one composite index.
        if (tn.tag == .type_app) return refs.typeFromTypeApp(self, type_node, tn);
        if (tn.tag == .fn_type) return refs.typeFromFnType(self, tn);
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
                self.sink.report(.{ .span = refs.tokSpan(self, tok), .code = .T0001 }, "generic type '{s}' requires type arguments, e.g. {s}[int]", .{ name, name }) catch {};
                return .invalid;
            }
            return Type.structT(id);
        }
        if (self.activeEnumMap().get(name)) |id| {
            // A generic enum named WITHOUT type args (`x: Either`) is not a value type —
            // it needs its args. Diagnose rather than mis-resolve it to `enumT`
            // (mirror the generic-struct gate above).
            if (id < self.enumSyms().len and self.enumSyms()[id].is_generic) {
                self.sink.report(.{ .span = refs.tokSpan(self, tok), .code = .T0001 }, "generic type '{s}' requires type arguments, e.g. {s}[int]", .{ name, name }) catch {};
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
        if (std.mem.eql(u8, name, "never")) {
            if (pos == .ret) return Type.never;
            self.sink.report(.{ .span = refs.tokSpan(self, tok), .code = .T0001 }, "'never' can only be the return type of a function", .{}) catch {};
            return .invalid;
        }
        // An exact stdlib type/namespace named here means the import is missing —
        // name the module. The print family (spelling != null) is never a type, so
        // it falls through to the bare message unchanged.
        if (StdNames.importHintFor(name)) |hint| {
            if (hint.spelling == null) {
                self.sink.report(.{ .span = refs.tokSpan(self, tok), .code = .T0001 }, err_unknown_type ++ "; " ++ StdNames.add_import, .{ name, hint.module }) catch {};
                return .invalid;
            }
        }
        // The import hint returns only for an EXACT std name, so a typo of a visible type
        // name falls through here. Runs after the import-hint check so an exact std name
        // still names the module rather than suggesting a near neighbour.
        if (nearmiss.suggest(name, refs.typeNameIter(self))) |cand| {
            self.sink.report(.{ .span = refs.tokSpan(self, tok), .code = .T0001 }, err_unknown_type ++ "; did you mean '{s}'?", .{ name, cand }) catch {};
            return .invalid;
        }
        self.sink.report(.{ .span = refs.tokSpan(self, tok), .code = .T0001 }, err_unknown_type, .{name}) catch {};
        return .invalid;
    }

    /// Resolve a function type `fn(A, B) -> R` (a `fn_type` node) to its interned `func`.
    /// A missing `-> R` is unit, as in a declaration; any invalid part poisons the whole.
    pub fn typeFromFnType(self: anytype, tn: Ast.Node) Type {
        const param_nodes = Ast.rangeSlice(self.tree, tn.lhs.int());
        var buf: [8]Type = undefined;
        const params: []Type = if (param_nodes.len <= buf.len) buf[0..param_nodes.len] else (self.gpa.alloc(Type, param_nodes.len) catch return .invalid);
        defer if (param_nodes.len > buf.len) self.gpa.free(params);
        var bad = false;
        for (param_nodes, 0..) |pn, i| {
            params[i] = refs.typeFromNode(self, pn);
            if (params[i].kind == .invalid) bad = true;
        }
        const ret = refs.retTypeFromNode(self, tn.rhs);
        if (bad or ret.kind == .invalid) return .invalid;
        return refs.internFn(self, params, ret);
    }

    /// Intern the function type `fn(params) -> ret`. The label is rendered BEFORE the
    /// intern: rendering reads the composite table, whose spinlock is not reentrant. It
    /// names no type var, since one interned fn type serves every scope that spells it.
    /// Out of memory poisons to `.invalid`.
    pub fn internFn(self: anytype, params: []const Type, ret: Type) Type {
        var label: std.Io.Writer.Allocating = .init(self.gpa);
        defer label.deinit();
        refs.writeFnSig(self, &label.writer, params, ret, .{}) catch return .invalid;
        const name = label.toOwnedSlice() catch return .invalid;
        const idx = self.composite.internFn(self.gpa, params, ret, name) catch return .invalid;
        return Type.func(idx);
    }

    /// `ty` rebuilt with `mapper.map(t)` applied to each param and the return of a fn type,
    /// or to each arg of an `App`, then re-interned; any other type is returned as is. The
    /// one rebuild of a composite type: each caller maps the leaves (its type variables) and
    /// recurses through `mapper`. Out of memory poisons to `.invalid`.
    pub fn mapComposite(self: anytype, ty: Type, mapper: anytype) Type {
        if (ty.isFunc()) {
            const parts = Composite.fnParts(self.composite.at(ty.funcIdx()));
            var buf: [8]Type = undefined;
            const sub: []Type = if (parts.params.len <= buf.len) buf[0..parts.params.len] else (self.gpa.alloc(Type, parts.params.len) catch return Type.invalid);
            defer if (parts.params.len > buf.len) self.gpa.free(sub);
            for (parts.params, 0..) |p, i| sub[i] = mapper.map(p);
            return refs.internFn(self, sub, mapper.map(parts.ret));
        }
        if (ty.isApp()) {
            const e = self.composite.at(ty.appIdx());
            var buf: [8]Type = undefined;
            const sub: []Type = if (e.args.len <= buf.len) buf[0..e.args.len] else (self.gpa.alloc(Type, e.args.len) catch return Type.invalid);
            defer if (e.args.len > buf.len) self.gpa.free(sub);
            for (e.args, 0..) |a, i| sub[i] = mapper.map(a);
            const idx = self.composite.intern(self.gpa, e.ctor, sub, e.ctor_is_enum) catch return Type.invalid;
            return Type.app(idx);
        }
        return ty;
    }

    /// The names `writeType` gives type vars; a var with none renders as `typeName` does.
    pub const VarNames = struct {
        /// Names for the vars `base + i`; with `open`, only for those still open there.
        names: []const []const u8 = &.{},
        open: ?[]const bool = null,
        base: u32 = 0,
        /// Names for the vars `i` below `base`: a bounded template's own params in its
        /// abstract check, which a generic call inside it leaves unbound.
        outer: []const []const u8 = &.{},

        fn get(v: VarNames, ord: u32) ?[]const u8 {
            if (ord >= v.base) {
                const i = ord - v.base;
                const is_open = if (v.open) |o| i < o.len and o[i] else true;
                if (i < v.names.len and is_open) return v.names[i];
            }
            return if (ord < v.outer.len) v.outer[ord] else null;
        }
    };

    /// `ty` as a diagnostic spells it: an `App` with its args (`Map[str, Vec[int]]`), a fn
    /// type from its parts, a type var by `vars`, and any other type as `typeName`.
    pub fn writeType(self: anytype, w: *std.Io.Writer, ty: Type, vars: VarNames) std.Io.Writer.Error!void {
        // An instance re-check writes a report again as its template would, at any depth,
        // so the reports of instances that fail alike can merge into one.
        if (comptime @hasDecl(std.meta.Child(@TypeOf(self)), "templateSpelling"))
            if (self.templateSpelling(ty)) |name| return w.writeAll(name);
        if (ty.isTypeVar()) if (vars.get(ty.typeVarOrd())) |name| return w.writeAll(name);
        if (ty.isApp()) {
            try w.writeAll(refs.typeName(self, ty));
            try w.writeByte('[');
            for (self.composite.at(ty.appIdx()).args, 0..) |a, i| {
                if (i != 0) try w.writeAll(", ");
                try refs.writeTypePart(self, w, a, vars);
            }
            return w.writeByte(']');
        }
        if (ty.isFunc()) {
            const parts = Composite.fnParts(self.composite.at(ty.funcIdx()));
            return refs.writeFnSig(self, w, parts.params, parts.ret, vars);
        }
        try w.writeAll(refs.typeName(self, ty));
    }

    /// `writeType` as a `{f}` format argument, so a message spells its types in place.
    pub fn Fmt(comptime Self: type) type {
        return struct {
            self: Self,
            ty: Type,
            vars: VarNames = .{},
            part: bool = false,

            /// Spelled as a part of another type, where unit reads `()`.
            pub fn nested(f: @This()) @This() {
                var g = f;
                g.part = true;
                return g;
            }

            pub fn format(f: @This(), w: *std.Io.Writer) std.Io.Writer.Error!void {
                if (f.part) return refs.writeTypePart(f.self, w, f.ty, f.vars);
                try refs.writeType(f.self, w, f.ty, f.vars);
            }
        };
    }

    pub fn fmt(self: anytype, ty: Type, vars: VarNames) Fmt(@TypeOf(self)) {
        return .{ .self = self, .ty = ty, .vars = vars };
    }

    /// `fn(A, B) -> R`, with a unit `-> ()` left off as in the source spelling.
    fn writeFnSig(self: anytype, w: *std.Io.Writer, params: []const Type, ret: Type, vars: VarNames) std.Io.Writer.Error!void {
        try w.writeAll("fn(");
        for (params, 0..) |p, i| {
            if (i != 0) try w.writeAll(", ");
            try refs.writeTypePart(self, w, p, vars);
        }
        try w.writeByte(')');
        if (ret.kind == .unit) return;
        try w.writeAll(" -> ");
        try refs.writeTypePart(self, w, ret, vars);
    }

    /// A type nested in another's spelling, where unit reads as source writes it.
    pub fn writeTypePart(self: anytype, w: *std.Io.Writer, ty: Type, vars: VarNames) std.Io.Writer.Error!void {
        if (ty.kind == .unit) return w.writeAll("()");
        try refs.writeType(self, w, ty, vars);
    }

    /// Resolve a qualified `mod.Type` type-reference (a `field_access` in type
    /// position; the receiver binds to a `.module`) to the owning module's GLOBAL
    /// struct/enum id. Graph mode only. Visibility was already enforced by
    /// `resolve_graph`, so a private type here is a defensive `invalid`.
    pub fn typeFromQualified(self: anytype, node_idx: Ast.Index, n: Ast.Node) Type {
        _ = node_idx;
        const g = self.graphCtx();
        const recv = self.tree.nodes[n.lhs.int()];
        const target = refs.moduleOf(self, n.lhs) orelse {
            if (recv.tag != .identifier) return .invalid;
            const recv_name = refs.nameText(self, recv.main_token);
            self.sink.report(.{ .span = refs.tokSpan(self, recv.main_token), .code = .T0002 }, err_unknown_module, .{recv_name}) catch {};
            return .invalid;
        };
        const member = refs.nameText(self, n.main_token);
        const ty: Type, const kind: []const u8, const exported = if (g.mods[target].struct_ids.get(member)) |id|
            .{ Type.structT(id), "struct", refs.exportsType(g, target, member, self.structSyms()[id]) }
        else if (g.mods[target].enum_ids.get(member)) |id|
            .{ Type.enumT(id), "enum", refs.exportsType(g, target, member, self.enumSyms()[id]) }
        else {
            self.sink.report(.{ .span = refs.tokSpan(self, n.main_token), .code = .T0003 }, err_module_no_type, .{member}) catch {};
            return .invalid;
        };
        if (exported) return ty;
        self.sink.report(.{ .span = refs.tokSpan(self, n.main_token), .code = .R0005 }, "{s} '{s}' is not exported by module '{s}'", .{ kind, member, g.mods[target].path }) catch {};
        return .invalid;
    }

    /// Whether `target.member` names the type `sym` that `member` binds in `target`: a
    /// pub type `target` declares, or one a `pub import` of it re-exports. A type the
    /// module only imports stays its own, as the resolver rules where it walks type refs
    /// (not every type position is resolved).
    pub fn exportsType(g: *const GraphCtx, target: u32, member: []const u8, sym: anytype) bool {
        if (!sym.pub_export) return false;
        if (sym.mod == target) return true;
        for (g.mods[target].selections) |sel| {
            if (std.mem.eql(u8, sel.local, member)) return sel.exported;
        }
        return false;
    }

    /// The module a qualifier node names in graph mode: an import namespace, or `ns.sub`
    /// where `ns` re-exports the namespace `sub`. Read from the tree, not the resolver's
    /// bindings, since not every type position is resolved.
    pub fn moduleOf(self: anytype, idx: Ast.Index) ?u32 {
        const g = self.graphCtx();
        const n = self.tree.nodes[idx.int()];
        return switch (n.tag) {
            .identifier => g.namespaceOfIn(self.graph_mod, refs.nameText(self, n.main_token)),
            .field_access => g.mods[refs.moduleOf(self, n.lhs) orelse return null].pub_namespaces.get(refs.nameText(self, n.main_token)),
            else => null,
        };
    }

    /// Resolve a type application `Box[int]` / `Either[int,bool]` (a `type_app` node) to
    /// a composite `App`. The base must resolve to a GENERIC STRUCT or GENERIC
    /// ENUM ctor; each arg is recursively resolved via `typeFromNode` (so it grounds to a
    /// concrete type inside an instance re-check / a non-generic body, and to a
    /// `type_var` while decoding a template's field/payload patterns), then the
    /// `(ctor, is_enum, args)` tuple is interned to one composite index. A non-generic
    /// base or an arity mismatch is a clean type error.
    pub fn typeFromTypeApp(self: anytype, node_idx: Ast.Index, tn: Ast.Node) Type {
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
                if (StdNames.importHintFor(bname)) |hint| {
                    if (hint.spelling == null) {
                        self.sink.report(.{ .span = refs.tokSpan(self, base.main_token), .code = .T0001 }, err_unknown_type ++ "; " ++ StdNames.add_import, .{ bname, hint.module }) catch {};
                        return .invalid;
                    }
                }
                if (nearmiss.suggest(bname, refs.typeNameIter(self))) |cand| {
                    self.sink.report(.{ .span = refs.tokSpan(self, base.main_token), .code = .T0001 }, err_unknown_type ++ "; did you mean '{s}'?", .{ bname, cand }) catch {};
                    return .invalid;
                }
                self.sink.report(.{ .span = refs.tokSpan(self, base.main_token), .code = .T0001 }, err_unknown_type, .{bname}) catch {};
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
            self.sink.report(.{ .span = refs.spanOf(self, node_idx), .code = .T0001 }, "'{s}' is not generic; drop the type arguments", .{gen_name}) catch {};
            return .invalid;
        }
        const arg_nodes = Ast.rangeSlice(self.tree, tn.rhs.int());
        if (arg_nodes.len != gen_params.len) {
            const args_span: Span = .{ .start = refs.byteOf(self, tn.main_token), .end = refs.spanOf(self, node_idx).end };
            self.sink.report(.{ .span = args_span, .code = .T0001 }, "'{s}' expects {d} type argument(s), got {d}", .{ gen_name, gen_params.len, arg_nodes.len }) catch {};
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
    /// The fn's decl node (`Ast.none` for a bodyless seeded builtin; identify
    /// builtin-ness by `kind`, never by this sentinel).
    decl_node: Ast.Index,
    /// `.builtin` for a seeded builtin (`panic`/intrinsics), else `.user_fn` (from the resolver).
    kind: symbols.SymKind,
    /// Owning module id; meaningless for a homeless seeded builtin.
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
    /// One `[]Callee` per module, parallel to `node_types`: each method call's
    /// selection, on its member node. Generic instances carry their own
    /// (`Mono.Instance.callees`); a bounded template's span here holds its abstract picks.
    callees: [][]Callee,
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
    /// The protocol table, indexed by the protocol id a `.protocol` Callee names. Names
    /// are borrowed source / prelude strings; the arrays are owned (`freeProtocolEntries`).
    /// Read by the LSP only. An `App`-typed entry in `method_params`/`method_rets` is
    /// opaque here: the composite table it indexed died with the checker.
    protocols: []ProtocolSym = &.{},
    /// The authorized structural auto-derive recipes, in canonical order. Empty
    /// for a program with no derived `Eq` use. Each recipe's `name`, `params`, and
    /// `field_witnesses` OUTER slice are OWNED here; the `FieldWitness` name slices +
    /// `protocol_name` are BORROWED (sibling derive/instance/fn names + prelude/source
    /// protocol names — all outlive codegen).
    derives: []DeriveRecipe = &.{},
    /// The per-type descriptor plan demanded by `descriptor_of[T]()` sites, in canonical
    /// order. Empty for a program with no `descriptor_of` use. PODs (owned backing array).
    descriptor_types: []DescType = &.{},
    /// The erased hash/eq witnesses for the descriptor'd Hashable types, in canonical
    /// order. Each entry's `name` is OWNED (freed in `deinit`); the backing array is owned.
    erased_units: []ErasedUnit = &.{},
    /// The prelude protocol ids, snapshotted off the checker so codegen resolves
    /// each `==`/`<`/`+`/`.hash()`/`print`/`?`-widen witness by its SPECIFIC protocol (a
    /// sibling protocol reusing the name is excluded). Threaded into every job's `Frozen`
    /// / lower `Inputs`; lower and the fingerprint fold read the SAME bundle.
    prelude_ids: PreludeProtocolIds = .{},
    /// The compiler-provided `char` struct id (see `Prelude.char_struct`), threaded into
    /// lower `Inputs` so the conversion recognizer + char-literal lowering key off it.
    /// Null only for a prelude-less internal caller.
    char_struct: ?u32 = null,
    /// The fn-value thunks, sorted by `(kind, key)`: one per fn some body names as a
    /// value. Each `name` is owned; `target`/`sig` borrow `sigs` and the resolver names.
    adapters: []Closure.Adapter = &.{},
    /// One unit per lambda per lowerable owner (a non-generic fn, or an instance), sorted
    /// by `(owner, node)`. Owned.
    lambda_units: []Closure.LambdaUnit = &.{},
    /// The boxed locals of each lowerable owner that has one, sorted by owner. Owned.
    owner_closures: []Closure.OwnerClosure = &.{},
    /// Every lowered body written in source, in canonical order (see `Body`): a body's
    /// index here is its closure owner, its escape unit and its codegen unit. The list is
    /// owned; each body borrows its tables from this result.
    bodies: Body.List = .{},
    /// Each fn's boxed locals by global id, which its body borrows. Owned.
    fn_boxed: []const []const u32 = &.{},
    /// The nodes the monomorphization tail's call scans visited: a work count that must
    /// grow linearly with the program, which a test can pin where a timing would flake.
    mono_scanned_nodes: usize = 0,

    pub fn deinit(self: *GraphResult, gpa: std.mem.Allocator) void {
        for (self.node_types) |nt| gpa.free(nt);
        gpa.free(self.node_types);
        for (self.callees) |cs| gpa.free(cs);
        gpa.free(self.callees);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        for (self.sigs) |s| gpa.free(@constCast(s.params));
        gpa.free(self.sigs);
        LayoutEngine.freeLayouts(gpa, self.layouts);
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
            gpa.free(inst.callees);
            gpa.free(@constCast(inst.params));
            gpa.free(@constCast(inst.name.?));
            gpa.free(inst.boxed);
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
        freeProtocolEntries(gpa, self.protocols);
        gpa.free(self.protocols);
        // Derive recipes: free each unit's minted `name`, its `params`, and its
        // `field_witnesses` OUTER slice (the FieldWitness name slices are borrowed), then the
        // backing array.
        freeDeriveEntries(gpa, self.derives);
        gpa.free(self.derives);
        gpa.free(self.descriptor_types);
        for (self.erased_units) |u| if (u.name) |nm| gpa.free(nm);
        gpa.free(self.erased_units);
        ClosureTables.freeAdapters(gpa, self.adapters);
        Closure.freeLambdaUnits(gpa, self.lambda_units);
        Closure.freeOwnerClosures(gpa, self.owner_closures);
        gpa.free(self.bodies.items);
        freeFnBoxed(gpa, self.fn_boxed);
        self.* = undefined;
    }
};

/// The body list: every non-generic user fn with a body in global id order (a generic one
/// is lowered only through its instances), then every instance in instance order. Each
/// body borrows its tables from the result.
fn buildBodies(
    gpa: std.mem.Allocator,
    fns: []const GraphFnInput,
    fsyms: []const FnSym,
    sigs: []const Sig,
    node_types: []const []const Type,
    callees: []const []const Callee,
    fn_boxed: []const []const u32,
    instances: []const Mono.Instance,
) !Body.List {
    var list: std.ArrayList(Body) = .empty;
    errdefer list.deinit(gpa);
    for (fns, fsyms, 0..) |f, fs, gid| {
        if (f.decl_node == Ast.none or f.kind != .user_fn or fs.isGeneric()) continue;
        try list.append(gpa, .{
            .of = .{ .base = @intCast(gid) },
            .mod = f.module,
            .decl = f.decl_node,
            .sig = sigs[gid],
            .nodes = .{ .types = node_types[f.module], .callees = callees[f.module] },
            .boxed = if (gid < fn_boxed.len) fn_boxed[gid] else &.{},
        });
    }
    const n_base: u32 = @intCast(list.items.len);
    for (instances, 0..) |inst, k| try list.append(gpa, .{
        .of = .{ .instance = @intCast(k) },
        .mod = inst.mod,
        .decl = inst.decl_node,
        .sig = .{ .kind = .user_fn, .name = inst.name.?, .params = inst.params, .ret = inst.ret, .mut_mask = sigs[inst.template_gid].mut_mask },
        .nodes = inst.nodes(),
        .boxed = inst.boxed,
    });
    return .{ .items = try list.toOwnedSlice(gpa), .n_base = n_base };
}

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
    /// An ASSOCIATED (self-less) function `Vec[T].new()`: dispatched via a `Type[args]`
    /// receiver, with NO `self` arg prepended at the call site. `false` for an ordinary
    /// instance method (the common case), so every existing method entry is byte-identical.
    is_static: bool = false,
    /// A REIFIED-DISPATCH entry: appended in the mono tail for each reachable
    /// `(instance, method)` of a generic-type `impl`. `recv` is the reified concrete
    /// `structT`/`enumT` and `instance` indexes `GraphResult.instances`: `sealCallees` picks
    /// the entry on the reified receiver, so a call's Callee leads `lower` and `CallVisitor`
    /// to the instance's mangled symbol. `null` for an inherent / conformance /
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
    /// builtin entry. Appended in the synthesis barrier for each authorized derive so the
    /// seal and the operator witness sites pick it as `.one`, and `witnessCallee`/`foldWitness`
    /// route to the synthetic unit's mangled name (checked BEFORE `instance`/`fn_id`). Grows
    /// `Method` (never `Type`), which is never content-cache-memcpy'd, so `@sizeOf(Type)`
    /// is unaffected.
    derive: ?u32 = null,
    /// Declared in a bundled std/core module. A call in the program takes a method the
    /// program declares on a type over a bundled one of the same name, so the pick never
    /// depends on import order and a program keeps its own method when std gains one of
    /// that name.
    bundled: bool = false,
    /// A program's method with the name of a bundled method or a compiler builtin on the
    /// same receiver (W0011). A call written in a bundled module never selects it: std's
    /// bounds come from `len`/`byte_at` and its own methods, so a program's `len` reached
    /// from std would steer std's raw reads.
    shadows: bool = false,
};

/// The symbol method row `m` calls: a source-less derive's unit (checked first: a derive
/// row has `fn_id == 0`), a generic-type instance, else the fn itself. `names` is indexed
/// by fn id and gives `.kind` and `.name`. Null when `m` indexes past a table (a partial
/// view).
pub fn witnessSym(m: Method, derives: []const DeriveRecipe, instances: []const Mono.Instance, names: anytype) ?symbols.SymName {
    if (m.derive) |di| return if (di < derives.len) .{ .kind = .user_fn, .name = derives[di].name.? } else null;
    if (m.instance) |ii| return if (ii < instances.len) .{ .kind = .user_fn, .name = instances[ii].name.? } else null;
    if (m.fn_id >= names.len) return null;
    return .{ .kind = names[m.fn_id].kind, .name = names[m.fn_id].name };
}

/// The signature of the symbol `witnessSym` picks, from the same three sources; `sigs` is
/// indexed by fn id. An instance takes its template's modes. Null as for `witnessSym`.
pub fn witnessSig(m: Method, derives: []const DeriveRecipe, instances: []const Mono.Instance, sigs: anytype) ?struct { params: []const Type, ret: Type, mut_mask: u64 = 0 } {
    if (m.derive) |di| return if (di < derives.len) .{ .params = derives[di].params, .ret = derives[di].ret } else null;
    if (m.instance) |ii| {
        if (ii >= instances.len) return null;
        const tg = instances[ii].template_gid;
        return .{ .params = instances[ii].params, .ret = instances[ii].ret, .mut_mask = if (tg < sigs.len) sigs[tg].mut_mask else 0 };
    }
    if (m.fn_id >= sigs.len) return null;
    return .{ .params = sigs[m.fn_id].params, .ret = sigs[m.fn_id].ret, .mut_mask = sigs[m.fn_id].mut_mask };
}

/// A method TEMPLATE registered for a generic-type `impl Box[T] { .. }`, or for a method
/// with type params of its own on a non-generic struct or enum, keyed by that type's own
/// id (see `templateRecv`). Kept in
/// its OWN table (never `Method`) precisely because it carries NO `recv: Type`: a template's
/// receiver would be a check-time `App` whose composite index is freed post-typecheck, so
/// making it un-representable here turns the old prose-only "never deref a template's App
/// recv" rule into a type-level guarantee. Dispatch keys off `(recv_ctor, recv_is_enum)`
/// (plain ints, never a composite deref) via `findGenericMethod`, so it is deterministic at
/// any `-jN` and safe while the composite is alive (Pass A / Pass C / the mono tail). Read
/// only during checking; the mono tail turns each reachable `(instance, template)` into a
/// concrete reified `Method`. `name` and `protocol_args` are BORROWED (source, and the
/// conformance's row in `template_conformances`); the table owns no heap data.
pub const TemplateMethod = struct {
    recv_ctor: u32,
    recv_is_enum: bool,
    name: []const u8,
    fn_id: u32,
    mut_self: bool = false,
    /// Whether the method takes a leading `self` receiver. `false` for an ASSOCIATED
    /// (self-less) function `fn new() -> Vec[T]` in an inherent `impl`; the mono tail
    /// reifies its dispatch entry off the impl receiver rather than `inst.params[0]`,
    /// and the call site prepends no `self` arg. `true` for every ordinary method.
    has_self: bool = true,
    /// Declared in a bundled std/core module (see `Method.bundled`).
    bundled: bool = false,
    /// See `Method.shadows`.
    shadows: bool = false,
    /// The generic conformance `impl Box[T] has P[..]` this method witnesses (stamped by
    /// coherence), or null for an inherent method. `protocol_args` is that conformance's
    /// arg pattern over the impl's params; explicit args on a call pick among a receiver's
    /// conformances by it (`pickExplicitTemplate`).
    protocol_id: ?u32 = null,
    protocol_args: []const Type = &.{},
};

/// A generic impl's receiver pattern (`Box[T]`) and its params' bounds, decoded once from
/// the impl's header so that an impl with no methods is checked and conforms like any
/// other. Only impls whose pattern T0013 accepts have one.
pub const ImplHead = struct {
    mod: u32,
    recv_node: Ast.Index,
    recv: Type,
    /// Owned, ordinal-indexed like `FnSym.generic_bounds`.
    bounds: []const ?u32,
    bound_args: []const []const Type,
    /// The impl's param names in order (borrowed source slices in an owned array).
    gnames: []const []const u8,

    fn deinit(h: ImplHead, gpa: std.mem.Allocator) void {
        (Bounds{ .ids = @constCast(h.bounds), .args = @constCast(h.bound_args) }).deinit(gpa);
        gpa.free(h.gnames);
    }
};

/// An impl T0013 rejected: for its receiver pattern (`impl Box[int]`), or for a protocol
/// arg it cannot take (`impl P has Into[Option[int]]`). That is its only error, so a
/// conformance or method it would have provided is not reported missing.
pub const RejectedImpl = struct {
    mod: u32,
    /// The impl's receiver type-ref node.
    recv_node: Ast.Index,
    /// The receiver: a concrete type, or for a generic one its ctor as a nominal type
    /// (`structT(ctor)`), which no value has.
    recv: Type,
    /// The protocol of a conformance, null for an inherent impl.
    protocol: ?u32,

    fn receives(r: RejectedImpl, composite: *Composite, ty: Type) bool {
        if (!ty.isApp()) return Type.eql(r.recv, ty);
        const e = composite.at(ty.appIdx());
        const kind: Kind = if (e.ctor_is_enum) .@"enum" else .@"struct";
        return r.recv.kind == kind and r.recv.nominalId() == e.ctor;
    }
};

/// Whether a rejected impl would have made `ty` conform to `pid`.
pub fn rejectedConformance(rejected: []const RejectedImpl, composite: *Composite, ty: Type, pid: u32) bool {
    for (rejected) |r| if (r.protocol == pid and r.receives(composite, ty)) return true;
    return false;
}

/// Whether a rejected conformance would have given `ty` method `member`.
pub fn rejectedProtocolMethod(rejected: []const RejectedImpl, protocols: []const ProtocolSym, composite: *Composite, ty: Type, member: []const u8) bool {
    for (rejected) |r| {
        const pid = r.protocol orelse continue;
        if (!r.receives(composite, ty)) continue;
        for (protocols[pid].methods) |m| if (std.mem.eql(u8, m, member)) return true;
    }
    return false;
}

/// Where a method call is written, which decides between same-named methods. In the
/// program, its own method wins over a bundled one. In a bundled module, a program's
/// method that shadows a bundled method or a builtin (`Method.shadows`) is not a
/// candidate, so std's own method or the builtin is taken; a program's other methods
/// stay reachable, since they are how its types implement the protocols std calls.
pub const Site = enum {
    program,
    bundled,

    pub fn of(bundled_module: bool) Site {
        return if (bundled_module) .bundled else .program;
    }
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
    hashable: ?u32 = null,
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

    /// The compiler-provided managed-box generic struct templates: `ref[T]` (an 8-byte
    /// cell pointer) and the reserved `gc_array[T]`. Appended after `char` so their ids
    /// stay a pure function of source; `refFamily` keys a reified instance's family
    /// marker off them.
    ref_struct: ?u32 = null,
    gc_array_struct: ?u32 = null,

    /// The native-enum family of an `App` ctor: `.option`/`.result` when `ctor` is the
    /// prelude Option/Result TEMPLATE id, else `.none`. A user `enum Option` shadow has its
    /// own distinct id (never the prelude template id), so the native path stays silent on it.
    pub fn optResultFamily(p: Prelude, ctor: u32) LayoutEngine.NativeEnumFamily {
        if (p.option_enum) |oid| if (oid == ctor) return .option;
        if (p.result_enum) |rid| if (rid == ctor) return .result;
        return .none;
    }

    /// The managed-box family of a struct-`App` ctor: `.ref`/`.gc_array` when `ctor` is
    /// the prelude Ref/gc_array TEMPLATE id, else `.none`. A user `struct Ref` shadow has
    /// its own distinct id, so the native path stays silent on it. Mirrors `optResultFamily`.
    pub fn refFamily(p: Prelude, ctor: u32) LayoutEngine.NativeStructFamily {
        if (p.ref_struct) |id| if (id == ctor) return .ref;
        if (p.gc_array_struct) |id| if (id == ctor) return .gc_array;
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

/// The witness pick behind the checker's method dispatch (recorded as a Callee) and the
/// operator/derive/`?`-widen witness sites. A linear scan over `Type.eql` + name (table =
/// source fn-id order → deterministic):
///   * with `explicit_args`: pick the conformance method whose `protocol_args` match
///     (a generic protocol's conformances are coherence-deduped, so at most one matches),
///     additionally filtered to `witness_pid` when given;
///   * without explicit args, `witness_pid == null` (general `v.m(..)` dispatch) and
///     `<= 1` matching `(recv, name)`: BYTE-IDENTICAL to `findMethod` (the legacy path —
///     inherent, single conformance, prelude, generic instance) so no existing program's
///     dispatch/fingerprint changes;
///   * without explicit args and `>= 2` CONFORMANCE methods match: `.ambiguous`;
///   * otherwise the first match, a program's own before a bundled one (`Method.bundled`).
/// At a `.bundled` site a shadowing row (`Method.shadows`) is never a candidate (see `Site`).
/// `witness_pid` (operator/derive desugaring: `==`/`<`/`+`/`.hash()`/`print`/`?`-widen) keeps
/// ONLY methods witnessing that SPECIFIC protocol — so a same-named inherent method (no
/// `protocol_id`) AND a sibling protocol reusing the name are both excluded, and the operator
/// binds its own protocol's witness or nothing. Those methods stay callable through the
/// general `v.m(..)` path (`witness_pid == null`).
pub fn resolveConformanceMethod(methods: []const Method, recv: Type, name: []const u8, witness_pid: ?u32, explicit_args: ?[]const Type) MethodPick {
    // A witness pick serves every site alike: a program's conformance is how its type
    // implements the protocol, even where its method name shadows another.
    return switch (pickMethodIndex(methods, recv, name, witness_pid, explicit_args, .program)) {
        .none => .none,
        .one => |i| .{ .one = methods[i] },
        .ambiguous => .ambiguous,
    };
}

/// `resolveConformanceMethod`'s pick as an index into `methods`, for a caller that
/// records the selection (an index stays valid as the table grows; a `Method` copy
/// cannot be told apart from a same-content sibling).
pub const MethodIndexPick = union(enum) { none, one: u32, ambiguous };

pub fn pickMethodIndex(methods: []const Method, recv: Type, name: []const u8, witness_pid: ?u32, explicit_args: ?[]const Type, site: Site) MethodIndexPick {
    if (explicit_args) |ea| {
        for (methods, 0..) |m, i| {
            if (site == .bundled and m.shadows) continue;
            if (Type.eql(m.recv, recv) and std.mem.eql(u8, m.name, name) and
                m.protocol_id != null and eqlTypeVec(m.protocol_args, ea))
            {
                if (witness_pid) |want| if (m.protocol_id.? != want) continue;
                return .{ .one = @intCast(i) };
            }
        }
        return .none;
    }
    var first: ?u32 = null;
    var first_user: ?u32 = null;
    var total: usize = 0;
    var conform_count: usize = 0;
    for (methods, 0..) |m, i| {
        if (site == .bundled and m.shadows) continue;
        if (Type.eql(m.recv, recv) and std.mem.eql(u8, m.name, name)) {
            if (witness_pid) |want| {
                const got = m.protocol_id orelse continue;
                if (got != want) continue;
            }
            if (first == null) first = @intCast(i);
            if (first_user == null and !m.bundled) first_user = @intCast(i);
            total += 1;
            // A reified instance row names its conformance only for a witness pick; a
            // plain call on the instance was already resolved by its template.
            if (m.protocol_id != null and m.instance == null) conform_count += 1;
        }
    }
    if (total == 0) return .none;
    if (total == 1) return .{ .one = first.? };
    if (conform_count >= 2) return .ambiguous;
    return .{ .one = first_user orelse first.? };
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

/// Free the per-protocol owned arrays: `methods`, `generic_params` (outer arrays; the name
/// slices are borrowed), the decoded `method_params` (inner + outer) and `method_rets`, and
/// `mut_self`.
pub fn freeProtocolEntries(gpa: std.mem.Allocator, protocols: []const ProtocolSym) void {
    for (protocols) |p| {
        gpa.free(@constCast(p.methods));
        if (p.generic_params.len > 0) gpa.free(@constCast(p.generic_params));
        for (p.method_params) |mp| gpa.free(@constCast(mp));
        if (p.method_params.len > 0) gpa.free(@constCast(p.method_params));
        if (p.method_rets.len > 0) gpa.free(@constCast(p.method_rets));
        if (p.mut_self.len > 0) gpa.free(@constCast(p.mut_self));
    }
}

/// Look up an inherent method by receiver type + source name. A linear scan over
/// `Type.eql` (pure content comparison of kind+id) + name equality — no hashmap /
/// thread order, so every consumer selects the SAME method at any `-jN`. Every entry's
/// `recv` is a concrete type (generic `impl Box[T]` templates live in `TemplateMethod`,
/// keyed by ctor int), so this never dereferences a stale check-time `App` — use
/// `findGenericMethod` for the check-time generic-receiver dispatch instead. It picks at
/// the program's site: only lower's desugars (`[..]`, `for`-in) call it, and no bundled
/// module uses one (pinned by a test beside the bundled sources).
pub fn findMethod(methods: []const Method, recv: Type, name: []const u8) ?Method {
    var first: ?Method = null;
    for (methods) |mth| {
        if (!Type.eql(mth.recv, recv) or !std.mem.eql(u8, mth.name, name)) continue;
        if (!mth.bundled) return mth;
        if (first == null) first = mth;
    }
    return first;
}

/// Look up a method TEMPLATE entry by its receiver type-constructor, or by a non-generic
/// struct or enum's own id for its own-generic methods.
/// Keys off `recv_ctor`/`recv_is_enum` (plain ints, never a composite deref)
/// so it is safe while the composite is alive (Pass A / Pass C / the mono tail) and
/// deterministic at any `-jN`. Returns the first matching template entry, a program's own
/// before a bundled one (see `Method.bundled`); at a `.bundled` site a shadowing entry is
/// never a candidate (see `Site`).
pub fn findGenericMethod(templates: []const TemplateMethod, ctor: u32, is_enum: bool, name: []const u8, site: Site) ?TemplateMethod {
    var first: ?TemplateMethod = null;
    for (templates) |tmpl| {
        if (tmpl.recv_ctor != ctor or tmpl.recv_is_enum != is_enum or !std.mem.eql(u8, tmpl.name, name)) continue;
        if (site == .bundled and tmpl.shadows) continue;
        if (!tmpl.bundled) return tmpl;
        if (first == null) first = tmpl;
    }
    return first;
}

/// The key a method template on `recv` is filed under (see `templateRecv`).
pub const TemplateRecv = struct { ctor: u32, is_enum: bool, args: []const Type };

/// Where to look up `recv`'s method templates: a generic type's instance (an `App`) by its
/// ctor, its args binding the impl's params; a non-generic struct or enum by itself, with
/// no args, since its only templates are methods with type params of their own. Null for
/// any other type. A reified instance's fresh id keys no template.
pub fn templateRecv(composite: *Composite, recv: Type) ?TemplateRecv {
    if (recv.isApp()) {
        const e = composite.at(recv.appIdx());
        return .{ .ctor = e.ctor, .is_enum = e.ctor_is_enum, .args = e.args };
    }
    return switch (recv.kind) {
        .@"struct" => .{ .ctor = recv.struct_id, .is_enum = false, .args = &.{} },
        .@"enum" => .{ .ctor = recv.enum_id, .is_enum = true, .args = &.{} },
        else => null,
    };
}

/// What a call of method `name` on a generic-type receiver selects with no explicit
/// type args (see `pickGenericMethod`).
pub const TemplatePick = union(enum) { none, one: TemplateMethod, ambiguous };

/// `pickMethodIndex` over the method templates of the receiver's ctor, so a generic
/// receiver is ambiguous (T0025) exactly where a concrete one is: two or more templates
/// of that name witness conformances and no type args choose one. Otherwise the only
/// candidate, or the first, a program's own before a bundled one (see `Site`).
pub fn pickGenericMethod(templates: []const TemplateMethod, ctor: u32, is_enum: bool, name: []const u8, site: Site) TemplatePick {
    var first: ?TemplateMethod = null;
    var first_user: ?TemplateMethod = null;
    var total: usize = 0;
    var conform_count: usize = 0;
    for (templates) |tm| {
        if (tm.recv_ctor != ctor or tm.recv_is_enum != is_enum or !std.mem.eql(u8, tm.name, name)) continue;
        if (site == .bundled and tm.shadows) continue;
        if (first == null) first = tm;
        if (first_user == null and !tm.bundled) first_user = tm;
        total += 1;
        if (tm.protocol_id != null) conform_count += 1;
    }
    if (total == 0) return .none;
    if (total == 1) return .{ .one = first.? };
    if (conform_count >= 2) return .ambiguous;
    return .{ .one = first_user orelse first.? };
}

/// What explicit type args select on a call of method `name` on a generic-type receiver
/// (`b.m[A..](..)`). The checker and discovery both ask, so they pick one instance.
pub const ExplicitPick = union(enum) {
    /// `targs` (OWNED by the caller) are every generic param of `m`, the impl's first.
    /// `own`: the explicit args are the method's own type params, not protocol args.
    ok: struct { m: TemplateMethod, targs: []Type, own: bool },
    /// No method of that name.
    none,
    /// The method has `want` type params of its own.
    arity: struct { m: TemplateMethod, want: usize },
    /// No conformance method of that name has these protocol args.
    no_conformance,
};

/// A method with type params of its own (`fn none[U](self) -> Option[U]`) takes the
/// explicit args as those; the receiver binds the impl's. Any other method is called with
/// protocol args, as on a concrete receiver: they pick the conformance whose protocol
/// args, grounded through the receiver's args, equal them. `recv_args` are the receiver
/// App's args.
pub fn pickExplicitTemplate(gpa: std.mem.Allocator, composite: *Composite, templates: []const TemplateMethod, fns: []const FnSym, ctor: u32, is_enum: bool, recv_args: []const Type, name: []const u8, site: Site, explicit: []const Type) error{OutOfMemory}!ExplicitPick {
    const first = findGenericMethod(templates, ctor, is_enum, name, site) orelse return .none;
    const ff = fns[first.fn_id];
    const n_impl = implParamCount(composite, ff);
    if (n_impl < ff.generic_params.len) {
        const want = ff.generic_params.len - n_impl;
        if (explicit.len != want) return .{ .arity = .{ .m = first, .want = want } };
        const targs = (try bindImplParamsPrefix(gpa, composite, ff, recv_args, n_impl)) orelse return .none;
        @memcpy(targs[n_impl..], explicit);
        return .{ .ok = .{ .m = first, .targs = targs, .own = true } };
    }
    var grounded: std.ArrayList(Type) = .empty;
    defer grounded.deinit(gpa);
    for ([_]u1{ 0, 1 }) |pass| {
        for (templates) |tm| {
            if (tm.recv_ctor != ctor or tm.recv_is_enum != is_enum or !std.mem.eql(u8, tm.name, name)) continue;
            if (siteRank(tm.bundled, tm.shadows, site) != pass or tm.protocol_id == null) continue;
            grounded.clearRetainingCapacity();
            for (tm.protocol_args) |pa| try grounded.append(gpa, try conform.substPattern(composite, gpa, pa, recv_args));
            if (!eqlTypeVec(grounded.items, explicit)) continue;
            const targs = (try bindImplParams(gpa, composite, fns[tm.fn_id], recv_args)) orelse continue;
            return .{ .ok = .{ .m = tm, .targs = targs, .own = false } };
        }
    }
    return .no_conformance;
}

/// A builtin scalar protocol-method recognizer: the pure, table-free source
/// of truth for the compiler-registered `Eq` conformance on builtin scalars. Returns the
/// method's return `Type` for a recognized `(recv, name)`, else null. BodyChecker types
/// the call and records `builtin` as its Callee, which lower (an inline machine op) and
/// the fingerprint fold (a fixed sentinel sig) read — so a builtin
/// scalar method is NEVER a phantom `t.fns`/`t.methods` entry (that would desync the
/// `names`/`sigs` parallel arrays and churn the method-count unit tests). All four scalars
/// ship `eq` -> bool: `int`/`bool` lower to an inline `icmp`; `str` to a heap-free
/// byte-compare (`load_byte` loop); `unit` to a trivially-true `bconst` (the two
/// `()` values are always equal). None gets a `t.fns` entry — each lowers to a machine op.
pub fn builtinScalarMethod(recv: Type, name: []const u8) ?struct { ret: Type, arity: usize, builtin: Callee.Builtin } {
    if (!recv.isScalar()) return null;
    // `eq(self, other) -> bool` is 2-ary (arity counts the non-self args = 1); `hash
    // (self) -> int` is 1-ary (0 non-self args). The `arity` lets the checker gate arg
    // count without hardcoding a per-method constant.
    if (std.mem.eql(u8, name, "eq")) return .{ .ret = Type.bool, .arity = 1, .builtin = .scalar_eq };
    if (std.mem.eql(u8, name, "hash")) return .{ .ret = Type.int, .arity = 0, .builtin = .scalar_hash };
    return null;
}

/// The string builtin methods, kept OUT of `builtinScalarMethod`, whose params are all
/// `Self`: `byte_at` takes an int. A pure, table-free recognizer (no `t.methods`/`t.fns`
/// row); the checker records its `builtin` as the call's Callee, which lower reads.
/// `to_string` on int/bool/str/unit produces a str
/// (it is `self.display()` for every Display-conforming scalar); struct/enum/float
/// `to_string` is gated separately (it may record a structural Display derive, and float's
/// Display is an impl in std/float), so it is NOT recognized here.
pub fn builtinStrMethod(recv: Type, name: []const u8) ?struct { ret: Type, params: []const Type, builtin: Callee.Builtin } {
    if (recv.kind == .str) {
        if (std.mem.eql(u8, name, "concat")) return .{ .ret = Type.str, .params = &.{Type.str}, .builtin = .str_concat };
        if (std.mem.eql(u8, name, "len")) return .{ .ret = Type.int, .params = &.{}, .builtin = .str_len };
        if (std.mem.eql(u8, name, "byte_at")) return .{ .ret = Type.int, .params = &.{Type.int}, .builtin = .str_byte_at };
        if (std.mem.eql(u8, name, "to_string")) return .{ .ret = Type.str, .params = &.{}, .builtin = .scalar_to_string };
    }
    if ((recv.isInteger() or recv.kind == .bool or recv.kind == .unit) and std.mem.eql(u8, name, "to_string"))
        return .{ .ret = Type.str, .params = &.{}, .builtin = .scalar_to_string };
    return null;
}

/// Whether `recv` is a builtin scalar with the compiler-registered `Ord` conformance
/// (`int` of any width, `bool`, `str`), whose `cmp` lowers inline. Kept apart from
/// `builtinScalarMethod` because its `Ordering` result is a prelude enum, not a fixed type.
pub fn isBuiltinOrdScalar(recv: Type) bool {
    return recv.isInteger() or recv.kind == .bool or recv.kind == .str;
}

/// Whether the compiler provides method `name` on the builtin type `recv`, with no method
/// row: a scalar's `eq`/`hash`/`cmp`, `str`'s `len`/`byte_at`/`concat`, a scalar's `to_string`,
/// and `into`/`try_into` on an integer, a float or `char`. The derivable `hash`/`to_string`
/// of a struct or enum is not counted: a type's own method of that name is how the type
/// implements it.
pub fn isBuiltinTypeMethod(recv: Type, name: []const u8, char_id: ?u32) bool {
    if (builtinScalarMethod(recv, name) != null or builtinStrMethod(recv, name) != null) return true;
    if (isBuiltinOrdScalar(recv) and std.mem.eql(u8, name, "cmp")) return true;
    return (recv.isInteger() or recv.kind == .float or isCharTy(recv, char_id)) and
        (std.mem.eql(u8, name, "into") or std.mem.eql(u8, name, "try_into"));
}

/// The kind of a conversion the `Into`/`TryInto` recognizer accepts. `widen`/`narrow`
/// are the int↔int cases; the four `char_*`/`*_char` cases are the char surface.
pub const ConvKind = enum { widen, narrow, char_to_int, byte_to_char, int_to_char, char_to_byte, int_to_float, float_to_int };

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
/// chained `expect`). Returns null unless BOTH are integers, so a struct/enum receiver
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
        // int -> float: lossless in range; scvtf rounds |x|>2^53 to nearest, never fails.
        // Signed / sub-64-bit only: a 64-bit unsigned >= 2^63 would scvtf negative (T0018).
        if (recv.isInteger() and expected.kind == .float and (recv.isSigned() or recv.intBits() < 64))
            return .{ .kind = .int_to_float, .target = expected };
        return null;
    }
    if (std.mem.eql(u8, member, "try_into")) {
        // float -> int: fallible (NaN / +-inf / |f|>=2^63 -> Err). Target restricted to plain
        // `int`: the shared witness is program-global with ONE captured Result[int,ConvErr] and a
        // fixed 8-byte int store; a narrower/unsigned target would need a per-target witness (T0018).
        if (recv.kind == .float and Type.eql(expected, Type.int))
            return .{ .kind = .float_to_int, .target = expected };
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
/// `is_tag0`/`is_tag1` are the two predicates (typed `bool`), `expect`/`or` yield
/// the payload type-arg (variant 0's payload). Like `builtinScalarMethod`, this is a pure
/// recognizer with NO `t.methods`/`t.fns` entry — the checker types the call and `lower`
/// inlines the tag test / payload load, so a native method never becomes a phantom table
/// row (which would desync `names`/`sigs` and churn the method-count tests). The family's
/// predicate NAMES differ (`is_some`/`is_none` vs `is_ok`/`is_error`) but both map to the
/// same tag-0/tag-1 test; `expect`/`or` are shared. Variant order is FIXED by
/// `registerPrelude` (payload = index 0, absence/error = index 1).
pub const OptResultMethod = enum { is_tag0, is_tag1, expect, @"or" };

pub fn optResultBuiltin(op: OptResultMethod) Callee.Builtin {
    return switch (op) {
        .is_tag0 => .opt_is_tag0,
        .is_tag1 => .opt_is_tag1,
        .expect => .opt_expect,
        .@"or" => .opt_or,
    };
}

/// The inverse of `optResultBuiltin`: null for a builtin that is not an Option/Result op.
pub fn builtinOptResult(k: Callee.Builtin) ?OptResultMethod {
    return switch (k) {
        .opt_is_tag0 => .is_tag0,
        .opt_is_tag1 => .is_tag1,
        .opt_expect => .expect,
        .opt_or => .@"or",
        .scalar_eq, .scalar_hash, .scalar_cmp, .str_concat, .str_len, .str_byte_at, .scalar_to_string, .display_to_string, .conv_into, .conv_try_into => null,
    };
}

pub fn optionResultMethod(family: LayoutEngine.NativeEnumFamily, name: []const u8) ?OptResultMethod {
    switch (family) {
        .none => return null,
        .option => {
            if (std.mem.eql(u8, name, "is_some")) return .is_tag0;
            if (std.mem.eql(u8, name, "is_none")) return .is_tag1;
        },
        .result => {
            if (std.mem.eql(u8, name, "is_ok")) return .is_tag0;
            if (std.mem.eql(u8, name, "is_error")) return .is_tag1;
        },
    }
    if (std.mem.eql(u8, name, "expect")) return .expect;
    if (std.mem.eql(u8, name, "or")) return .@"or";
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

/// Whether `t` has fields a structural derive could walk: a scalar has none, so "make every
/// field conform" is no fix for it.
fn isAggregate(t: Type) bool {
    return t.kind == .@"struct" or t.kind == .@"enum" or t.isApp();
}

/// Whether `ty` is or holds the poison type `.invalid`.
fn holdsPoison(composite: *Composite, ty: Type) bool {
    if (ty.kind == .invalid) return true;
    if (!ty.isApp() and !ty.isFunc()) return false;
    for (composite.at(ty.struct_id).args) |a| if (holdsPoison(composite, a)) return true;
    return false;
}

/// The std module to import for the blocking `box`'s Display impl: std/ref for a `Ref`,
/// a prelude type whose impl a program may not have imported. A collection's impl comes
/// with its own module, so its box is the private `gc_array`, which no import fixes.
fn refImportModule(model: *const Model, composite: *Composite, box: Type) ?[]const u8 {
    const rs = (model.prelude orelse return null).ref_struct orelse return null;
    if (!box.isApp()) return null;
    const e = composite.at(box.appIdx());
    if (e.ctor_is_enum or e.ctor != rs) return null;
    return StdNames.familyModule(.ref);
}

/// The std module whose `impl` makes the scalar named `tname` conform to protocol `pid`, or
/// null. Only a prelude protocol qualifies: a program's own protocol may reuse a prelude
/// name, and no std import satisfies it.
pub fn scalarImplModule(model: *const Model, tname: []const u8, pid: u32) ?[]const u8 {
    const ids = model.preludeProtocols();
    inline for (std.meta.fields(PreludeProtocolIds)) |f| {
        if (@field(ids, f.name) == pid) return StdNames.scalarImplModule(tname, model.protocols[pid].name);
    }
    return null;
}

/// `scalarImplModule` for a method call `x.member()` on a scalar named `tname`: the
/// prelude protocol declaring `member` (or Display, for `to_string`) has its impl there.
pub fn scalarMethodModule(model: *const Model, tname: []const u8, member: []const u8) ?[]const u8 {
    const ids = model.preludeProtocols();
    inline for (std.meta.fields(PreludeProtocolIds)) |f| if (@field(ids, f.name)) |pid| {
        const declares = for (model.protocols[pid].methods) |m| {
            if (std.mem.eql(u8, m, member)) break true;
        } else false;
        const via_display = ids.display == pid and std.mem.eql(u8, member, "to_string");
        if (declares or via_display) if (StdNames.scalarImplModule(tname, model.protocols[pid].name)) |m| return m;
    };
    return null;
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
    /// The expression that demanded the derive (`s.hash()`, `a == b`): where a derive the
    /// synthesis later finds impossible is reported, since synthesis itself has no source.
    site: ?DeriveSite = null,
};

/// An expression node in a module. Kept as the node, not a measured span: every `==` and
/// `.hash()` records one, and only a derive that turns out impossible ever measures it.
pub const DeriveSite = struct { scope: u32, node: Ast.Index };

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
    /// Whether method i's requirement takes `mut self`, parallel to `methods`; empty for a
    /// prelude protocol, whose requirements all take `self`. A conformance method must take
    /// its receiver the same way (T0076), so a desugar that calls a witness (`==`, `for`)
    /// knows the mode from the protocol. Owned by `t.protocols`.
    mut_self: []const bool = &.{},

    pub fn requiresMutSelf(p: ProtocolSym, i: usize) bool {
        return i < p.mut_self.len and p.mut_self[i];
    }
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

/// One recorded GENERIC conformance `impl Ctor[T] has P[..]` — kept in its OWN table
/// (never `conformances`) because its receiver is a template, not a concrete type: an
/// `App` row in `conformances` would violate the concrete-only invariant `existence`
/// relies on. Pre-reify queries reach it through `conform.generic`, which grounds the
/// row through an App receiver and checks the impl's own `[T has P]` bounds; post-reify
/// each reached instance gets a concrete `conformances` row (see `monomorphize`).
/// `protocol_args` is the arg PATTERN (`Iterator[T]` -> `[type_var(0)]`) over the impl's
/// params. OWNED by `t.template_conformances` (`protocol_args` only).
pub const TemplateConformance = struct {
    protocol_id: u32,
    recv_ctor: u32,
    recv_is_enum: bool,
    protocol_args: []const Type = &.{},
    /// The impl's receiver pattern (`Box[T]`), which binds the impl's params from a
    /// receiver App's args.
    recv: Type = .invalid,
    /// How many leading generic params are the impl's own (`implParamCount`).
    n_params: u32 = 0,
    /// The impl's per-param bounds and bound args, BORROWED from its `ImplHead`.
    bounds: []const ?u32 = &.{},
    bound_args: []const []const Type = &.{},
    /// Set on the `Eq` row an `Ord` impl implies (Ord refines Eq): the protocol whose
    /// methods witness this row.
    refines: ?u32 = null,
    /// The `impl` decl that declares the row, and its module, so an overlap reports like
    /// a concrete one (`coherence.reportOverlap`).
    impl_mod: u32 = 0,
    impl_decl: Ast.Index = Ast.none,
};

/// A top-level function's signature, decoded once up front so calls can be
/// checked against it (and forward references work).
pub const FnSym = struct {
    /// The fn's decl node (`Ast.none` for a bodyless seeded builtin; identify
    /// builtin-ness by `kind`, never by this sentinel).
    decl_node: Ast.Index,
    /// `.builtin` for a seeded builtin (`panic`/intrinsics), else `.user_fn`.
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
    /// Bit i: params[i] is `mut` (passed by the caller's place address); a method's
    /// `self` is params[0]. The one source of the modes: `Sig.mut_mask` and
    /// `Method.mut_self` are copies of it.
    mut_mask: u64 = 0,

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
/// accumulates the merge of all value-break sites (starting at `never`). `expected` is
/// the type expected of the construct's value, which each value break is checked against.
const CtxKind = enum { loop, while_for, labeled_block };
pub const LoopCtx = struct {
    kind: CtxKind,
    label: ?[]const u8,
    construct_node: Ast.Index,
    is_value: bool,
    join: Type,
    saw_value_break: bool,
    saw_bare_break: bool,
    expected: ?@import("BodyChecker.zig").Expected = null,
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
/// Per-module Callee slices, parallel to `gph_node_types`.
gph_callees: ?[][]Callee = null,
/// `t.methods.items.len` when `sealCallees` ran; a Debug guard that no method row is
/// appended after the seal (a pending slot could have needed it).
sealed_methods_len: usize = 0,

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

/// The methods of every impl that T0013 rejects for its receiver pattern (see
/// `implPatternFault`), shaped like `templates` but never dispatched. They exist so that
/// the rejection is the impl's only error: a call of one is not reported again.
rejected_methods: std.ArrayList(TemplateMethod) = .empty,

/// Every impl T0013 rejects, once each (see `RejectedImpl`).
rejected_impls: std.ArrayList(RejectedImpl) = .empty,

/// Every generic impl T0013 accepts, built serially before Phase A (see `ImplHead`).
impl_heads: std.ArrayList(ImplHead) = .empty,

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

/// Recorded GENERIC conformances `impl Ctor[T] has P[..]`, filled by the serial
/// `checkCoherence` phase. Kept SEPARATE from `conformances` (see `TemplateConformance`).
/// Frozen onto the `Model`; freed at teardown (each entry owns `protocol_args`).
template_conformances: std.ArrayList(TemplateConformance) = .empty,

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
conv_float_int_result: ?Type = null,

/// The authorized structural auto-derive recipes, built by the synthesis barrier
/// at the end of `monomorphize` in canonical order. Transferred whole into
/// `GraphResult.derives` by `checkGraph`; leftover (error path) freed by the teardown.
derives: std.ArrayList(DeriveRecipe) = .empty,

/// The types demanded by `descriptor_of[T]()` sites, merged in fn-id order by
/// `checkBodies` (mirroring `derive_reqs`), then reified and consumed by the synthesis
/// barrier (deduped + canonically sorted into `descriptor_types`/`erased_units`). Empty
/// for a program with no `descriptor_of` use → no synthesis → corpus byte-identical.
descriptor_reqs: std.ArrayList(Type) = .empty,

/// The resolved per-type descriptor plan, produced by the synthesis barrier in canonical
/// order. Transferred whole into `GraphResult.descriptor_types` by `checkGraph`. PODs.
descriptor_types: std.ArrayList(DescType) = .empty,

/// The erased hash/eq witnesses to codegen, produced by the synthesis barrier in canonical
/// order (names minted AFTER the sort). Transferred into `GraphResult.erased_units`; each
/// `name` is owned (freed there / at teardown).
erased_units: std.ArrayList(ErasedUnit) = .empty,

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

/// The pair struct minted for each grounded function signature (`reify.fnStructFor`):
/// grounded `params ++ [ret]` key bytes (owned) -> struct id. Freed at teardown.
fn_struct_dedup: std.StringHashMapUnmanaged(u32) = .empty,

/// The instance minted for each generic ctor over grounded args (`reify.claimGrounded`):
/// owned key bytes -> its reified `structT`/`enumT`. Freed at teardown.
app_reify_dedup: std.StringHashMapUnmanaged(Type) = .empty,

/// OWNS the mangled names minted for reified generic-struct instances. A reified
/// `StructSym.name` is a view into one of these; `snapshotLayouts` dupes it, so these
/// are freed wholesale at teardown (the struct teardown frees `field_*`/`offsets` but
/// NOT `name`, which for a source struct is a borrowed source slice).
reified_names: std.ArrayList([]u8) = .empty,

/// One-shot latch for the T0017 instantiation-depth diagnostic: the guard fires
/// at the FIRST too-deep instantiation in the (serial) worklist and then goes quiet,
/// so an unbounded `f[T] -> f[Box[T]]` yields exactly one diagnostic at `-j1`/`-jN`.
mono_depth_capped: bool = false,

/// The nodes `scanCalls` has visited (see `GraphResult.mono_scanned_nodes`).
mono_scanned_nodes: usize = 0,

/// The instance re-checks' hints as their templates spell them (see
/// `BodyChecker.template_spellings`), until `unifyInstanceHints` consumes them. Owned.
instance_hints: std.ArrayList(BodyChecker.TemplateSpelling) = .empty,

/// Per-global-fn-id poison flags: true for a BOUNDED generic template whose
/// bound-as-axiom body check (`bodyUnit`) emitted at least one diagnostic. The mono
/// tail (`enqueueInstance`) skips instantiating a poisoned template so a definition
/// error is reported ONCE at the template, never re-cascaded per instance. Allocated
/// serially in `checkBodies` (post-join) in fn-id order; freed at teardown. Empty
/// (`&.{}`) until then, so the guard is inert on any pre-checkBodies path.
bound_poisoned: []bool = &.{},

/// Per-global-fn-id boxed locals (`Closure.OwnerClosure`), moved out of each body's
/// check in fn-id order by `checkBodies`; each list owned. Empty until then.
fn_boxed: [][]u32 = &.{},

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

/// How the per-fn body region (Pass C) dispatches. See `Jobs`.
jobs: Jobs = .{},

/// How the per-fn body region (Pass C, `checkBodies`) dispatches its units. Every
/// setting feeds the SAME slots + merge + sort, so the result is byte-identical
/// regardless; these are perf levers only.
pub const Jobs = struct {
    /// The runtime the body region dispatches onto. Set in graph mode (carries the
    /// `-j N` worker cap from the driver's own pool). The body region fans out per-fn
    /// body checks via `Engine.chunkedFanOut(io, ...)` whenever a pool is present
    /// (`io != null`). `.limited(0)` (`-j1`) drives every unit onto the inline serial
    /// path — the byte-identity baseline. `io == null` (single-file internal callers +
    /// inline tests) is the only serial trigger: the single-source path passes null so
    /// the per-file pipeline's own per-file fan-out is the only parallelism (no
    /// nested-pool oversubscription). `-o`/`--emit ir` pass the real `io`.
    io: ?Io = null,
    /// The `-j` jobs knob: the chunk-count basis for the fan-out, 0 => host cpu count.
    /// Body checks dispatch ~`ncpu` contiguous chunks (each looping its fns serially)
    /// instead of one task per fn, so `-jN` scales instead of drowning in per-task
    /// overhead. Only meaningful when `io != null`.
    ncpu: usize = 0,
    /// Fan out only above this many fns. The determinism tests lower it so a program of
    /// a few fns still splits across workers: one large enough to cross the production
    /// threshold took minutes per test under the stack-tracing test allocator (#34).
    threshold: usize = Engine.Chunk.body.threshold,
};

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
        /// The resolver's lambda boundaries of this module's fns, sorted by fn node.
        closures: []const Closure.FnClosures = &.{},
        /// Bare struct name → GLOBAL struct id (this module's own decls only).
        struct_ids: std.StringHashMapUnmanaged(u32) = .empty,
        /// Bare enum name → GLOBAL enum id.
        enum_ids: std.StringHashMapUnmanaged(u32) = .empty,
        /// Bare protocol name → GLOBAL protocol id (this module's own protocol decls).
        /// Populated in Phase 0c (`registerProtocols`).
        protocol_ids: std.StringHashMapUnmanaged(u32) = .empty,
        /// Import namespace name → imported module id (graph module id).
        namespaces: std.StringHashMapUnmanaged(u32) = .empty,
        /// The namespaces a `pub import` re-exports, a subset of `namespaces`.
        pub_namespaces: std.StringHashMapUnmanaged(u32) = .empty,
        /// Bare alias name -> RESOLVED target `Type` (this module's `type X = Y` decls
        /// + the if-absent-seeded prelude `byte = uint8`). A pure lookup in
        /// `typeFromNode`; all cycle risk is confined to the one-shot resolve pass that
        /// fills this. Aliases mint no id, so this map is never read by lower/codegen/fp.
        alias_ids: std.StringHashMapUnmanaged(Type) = .empty,
        /// Served from the embedded std/core, not the program's own files.
        bundled: bool = false,
        /// The import path (`std/vec`), for diagnostics that name a module.
        path: []const u8 = "",
        /// The name a diagnostic location shows (`Graph.Module.displayName`).
        file: []const u8 = "",
        /// The modules this one imports directly (graph module ids).
        imports: []const u32 = &.{},
        /// The types and protocols this module's selective imports bind, each followed
        /// through re-exports to the module that declares it (the resolver reports every
        /// selection that is not exported).
        selections: []const Selection = &.{},
    };

    /// A type or protocol `import a/b.{name as local}` binds: `target` declares `name`.
    /// `exported` when a `pub import` re-exports it.
    pub const Selection = struct {
        local: []const u8,
        name: []const u8,
        target: u32,
        kind: enum { type, protocol },
        exported: bool = false,
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
    /// The methods of the impls T0013 rejected, frozen from Pass A (see
    /// `Typecheck.rejected_methods`). Read only to keep a call of one from a second error.
    rejected_methods: []const TemplateMethod,
    /// The impls T0013 rejected, frozen after coherence (see `RejectedImpl`).
    rejected_impls: []const RejectedImpl,
    /// The program-wide protocol table + recorded conformances, frozen before
    /// the parallel body pass. Read-only; no Pass-C consumer (sets up the conformance query).
    protocols: []const ProtocolSym,
    conformances: []const Conformance,
    /// Recorded GENERIC conformances (`impl Ctor[T] has P[..]`), frozen from
    /// `checkCoherence`. Read through `conform.generic` and `conform.iteratorItem`.
    template_conformances: []const TemplateConformance,
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
        .rejected_methods = t.rejected_methods.items,
        .rejected_impls = t.rejected_impls.items,
        .protocols = t.protocols.items,
        .conformances = t.conformances.items,
        .template_conformances = t.template_conformances.items,
        .prelude = t.prelude,
    };
}

/// The node-parallel tables one BodyChecker writes. Passed whole so a caller must
/// choose both slices: an instance re-check handing only its own `node_types` would
/// otherwise write its callees into the module slice.
pub const NodeSlots = struct { node_types: []Type, callees: []Callee };

/// Construct a `BodyChecker` for fn `f`, wiring its cursor to the fn's owning
/// module (`graph.mods[f.mod]`) and the node tables in `slots`. The per-fn
/// scratch starts empty.
fn bodyCheckerFor(t: *const Typecheck, model: *const Model, f: FnSym, slots: NodeSlots) BodyChecker {
    var bc: BodyChecker = .{
        .model = model,
        .gpa = t.gpa,
        .tree = t.tree,
        .tokens = t.tokens,
        .source = t.source,
        .resolutions = t.resolutions,
        .node_types = slots.node_types,
        .callees = slots.callees,
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
    bc.closures = Closure.findFn(mc.closures, f.decl_node);
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
/// forward to `sink.report` with the SAME literal format strings the layout code
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
        fn castTokSpan(ctx: *anyopaque, tok: u32) Span {
            return tokSpan(@ptrCast(@alignCast(ctx)), tok);
        }
        fn emitRecursive(ctx: *anyopaque, at: Span, requester: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.report(.{ .span = at, .code = .T0004 }, "recursive type '{s}' has infinite size", .{requester});
        }
        fn emitEmptyStruct(ctx: *anyopaque, at: Span, name: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.report(.{ .span = at, .code = .T0005 }, "empty struct '{s}' is not allowed", .{name});
        }
        fn emitEmptyEnum(ctx: *anyopaque, at: Span, name: []const u8) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            try tc.sink.report(.{ .span = at, .code = .T0006 }, "empty enum '{s}' is not allowed", .{name});
        }
        fn emitTooLarge(ctx: *anyopaque, at: Span, w: LayoutEngine.TooLarge) error{OutOfMemory}!void {
            const tc: *Typecheck = @ptrCast(@alignCast(ctx));
            if (w.holding) |h|
                try tc.sink.report(.{ .span = at, .code = .T0077 }, "an instance of {s} '{s}' holding '{s}' is {d} bytes, over the {d}-byte limit for a value; hold large data in a Vec", .{ @tagName(w.kind), w.name, h, w.size, w.limit })
            else
                try tc.sink.report(.{ .span = at, .code = .T0077 }, "{s}{s} '{s}' is {d} bytes, over the {d}-byte limit for a value; hold large data in a Vec", .{ if (w.instance) "an instance of " else "", @tagName(w.kind), w.name, w.size, w.limit });
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
        .tokSpan = T.castTokSpan,
        .emitRecursive = T.emitRecursive,
        .emitEmptyStruct = T.emitEmptyStruct,
        .emitEmptyEnum = T.emitEmptyEnum,
        // A value lives in a frame, so the largest frame codegen addresses bounds it.
        .max_size = FrameLayout.MAX_FRAME,
        .emitTooLarge = T.emitTooLarge,
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
/// `fns` is parallel to the resolver's global fn table (seeded builtins last, bodyless).
/// Caller owns the returned `GraphResult`.
pub fn checkGraph(
    gpa: std.mem.Allocator,
    ctx: *GraphCtx,
    mods: []const GraphModuleInput,
    fns: []const GraphFnInput,
    entry_mod: u32,
    jobs: Jobs,
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
    const callees = try gpa.alloc([]Callee, mods.len);
    var cs_built: usize = 0;
    errdefer {
        for (callees[0..cs_built]) |cs| gpa.free(cs);
        gpa.free(callees);
    }
    for (mods, 0..) |m, i| {
        const cs = try gpa.alloc(Callee, m.tree.nodes.len);
        @memset(cs, .{});
        callees[i] = cs;
        cs_built += 1;
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
        .jobs = jobs,
        .composite = composite,
    };
    defer {
        composite.deinit(gpa);
        gpa.destroy(composite);
        t.reify_map.deinit(gpa);
        var fdk = t.fn_struct_dedup.keyIterator();
        while (fdk.next()) |k| gpa.free(k.*);
        t.fn_struct_dedup.deinit(gpa);
        var adk = t.app_reify_dedup.keyIterator();
        while (adk.next()) |k| gpa.free(k.*);
        t.app_reify_dedup.deinit(gpa);
        for (t.reified_names.items) |nm| gpa.free(nm);
        t.reified_names.deinit(gpa);
        for (t.instance_hints.items) |h| gpa.free(h.template);
        t.instance_hints.deinit(gpa);
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
            gpa.free(inst.callees);
            gpa.free(@constCast(inst.params));
            gpa.free(inst.boxed);
            freeInstanceConformances(gpa, inst.conformances);
        }
        t.mono.deinit(gpa);
        // Derive requests (PODs, no owned data) + any un-transferred derive recipes
        // (success path empties `t.derives` via `toOwnedSlice`; an error path frees them).
        t.derive_reqs.deinit(gpa);
        freeDeriveEntries(gpa, t.derives.items);
        t.derives.deinit(gpa);
        // Descriptor requests + resolved plan (PODs). Any un-transferred erased units
        // (error path) own their minted `name`; the success path empties the list first.
        t.descriptor_reqs.deinit(gpa);
        t.descriptor_types.deinit(gpa);
        for (t.erased_units.items) |u| if (u.name) |nm| gpa.free(nm);
        t.erased_units.deinit(gpa);
        // The per-fn bound-poison flags (owned; empty until checkBodies ran).
        if (t.bound_poisoned.len > 0) gpa.free(t.bound_poisoned);
        for (t.fn_boxed) |bx| gpa.free(bx);
        if (t.fn_boxed.len > 0) gpa.free(t.fn_boxed);
        // The method table's backing array (entries' names are borrowed source
        // slices). Each entry may own a `protocol_args` dupe (freed here). On success
        // `toOwnedSlice` empties it, so both are a no-op there (the result frees them).
        freeMethodEntries(gpa, t.methods.items);
        t.methods.deinit(gpa);
        // The template table owns no per-entry heap data (names are borrowed source);
        // on success `toOwnedSlice` empties it, so this frees only the backing array.
        t.templates.deinit(gpa);
        t.rejected_methods.deinit(gpa);
        t.rejected_impls.deinit(gpa);
        for (t.impl_heads.items) |h| h.deinit(gpa);
        t.impl_heads.deinit(gpa);
        // On success `toOwnedSlice` empties the protocol table (the result frees it).
        freeProtocolEntries(gpa, t.protocols.items);
        t.protocols.deinit(gpa);
        // Each conformance may own a `protocol_args` dupe.
        for (t.conformances.items) |c| if (c.protocol_args.len > 0) gpa.free(@constCast(c.protocol_args));
        t.conformances.deinit(gpa);
        for (t.template_conformances.items) |c| if (c.protocol_args.len > 0) gpa.free(@constCast(c.protocol_args));
        t.template_conformances.deinit(gpa);
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
            if (s.fn_sig) |fs| gpa.free(fs.params);
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
    t.gph_callees = callees;

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
    // Phase A appends exactly one sig per global fn (a bodyless builtin gets a placeholder
    // sig, not a monomorph instance — those live in `t.mono`), so `t.fns` stays parallel to
    // `fns`; the sig name is the resolver's qualified/bare name for that id.
    std.debug.assert(t.fns.items.len == fns.len);
    for (t.fns.items, 0..) |f, i| {
        sigs[i] = .{ .kind = f.kind, .name = fns[i].name, .params = try gpa.dupe(Type, f.params), .ret = f.ret, .mut_mask = f.mut_mask };
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
            gpa.free(inst.callees);
            gpa.free(@constCast(inst.params));
            gpa.free(@constCast(inst.name.?));
            gpa.free(inst.boxed);
            freeInstanceConformances(gpa, inst.conformances);
        }
        gpa.free(instances);
    }

    // Transfer the method table out of the live list before the teardown defer
    // sees it. `toOwnedSlice` empties `t.methods`; the entries' names stay borrowed.
    // Each entry may own a `protocol_args` dupe, freed on the error path here (the
    // window after transfer, before the result owns it) — else it would leak.
    if (builtin.mode == .Debug) std.debug.assert(t.methods.items.len == t.sealed_methods_len);
    const methods = try t.methods.toOwnedSlice(gpa);
    errdefer {
        freeMethodEntries(gpa, methods);
        gpa.free(methods);
    }

    // Transfer the template table out of the live list (mirrors methods); it owns no
    // per-entry heap data, so the error path frees only the backing array.
    const templates_out = try t.templates.toOwnedSlice(gpa);
    errdefer gpa.free(templates_out);

    const protocols = try t.protocols.toOwnedSlice(gpa);
    errdefer {
        freeProtocolEntries(gpa, protocols);
        gpa.free(protocols);
    }

    // Transfer the derive recipes out of the live list before the teardown defer
    // sees them (mirrors instances/methods). `toOwnedSlice` empties `t.derives`; on the
    // error path here the recipes' owned data is freed (else it would leak).
    const derives = try t.derives.toOwnedSlice(gpa);
    errdefer {
        freeDeriveEntries(gpa, derives);
        gpa.free(derives);
    }

    // Transfer the descriptor plan + erased units (mirrors derives). `descriptor_types`
    // holds PODs; each erased unit owns its minted `name`, freed on the error path here.
    const descriptor_types = try t.descriptor_types.toOwnedSlice(gpa);
    errdefer gpa.free(descriptor_types);
    const erased_units = try t.erased_units.toOwnedSlice(gpa);
    errdefer {
        for (erased_units) |u| if (u.name) |nm| gpa.free(nm);
        gpa.free(erased_units);
    }

    try sealCalls(gpa, t.graph, node_types, callees, instances, sigs);

    const fn_boxed = t.fn_boxed;
    t.fn_boxed = &.{};
    errdefer freeFnBoxed(gpa, fn_boxed);
    const bodies = try buildBodies(gpa, fns, t.fns.items, sigs, node_types, callees, fn_boxed, instances);
    errdefer gpa.free(bodies.items);

    const adapters = try ClosureTables.buildAdapters(gpa, t.graph, bodies, callees, instances, fns, sigs, methods, derives);
    errdefer ClosureTables.freeAdapters(gpa, adapters);

    var layouts = try LayoutEngine.snapshotLayouts(gpa, t.structs.items);
    errdefer LayoutEngine.freeLayouts(gpa, layouts);
    const enum_layouts = try LayoutEngine.snapshotEnumLayouts(gpa, t.enums.items);
    errdefer LayoutEngine.freeEnumLayouts(gpa, enum_layouts);

    const lambda_units = try ClosureTables.buildLambdaUnits(gpa, t.graph, bodies, layouts, enum_layouts);
    errdefer Closure.freeLambdaUnits(gpa, lambda_units);
    var esc = try ClosureTables.analyzeEscapes(&t, fns, sigs, node_types, callees, bodies, instances, methods, lambda_units, layouts, enum_layouts);
    defer esc.deinit(gpa);
    const owner_closures = try ClosureTables.buildOwnerClosures(gpa, bodies, esc.unboxable);
    errdefer Closure.freeOwnerClosures(gpa, owner_closures);
    layouts = try ClosureTables.appendEnvLayouts(gpa, layouts, lambda_units, esc.frame, bodies, owner_closures);
    if (builtin.mode == .Debug) ClosureTables.checkOwners(bodies, lambda_units, owner_closures);

    // Diagnostics already carry their owning module in `scope`, sorted by
    // runGraph. Hand the owned slices to the result.
    const owned = try t.sink.toOwned();

    return GraphResult{
        .node_types = node_types,
        .callees = callees,
        .diags = owned.diags,
        .owned_msgs = owned.owned,
        .sigs = sigs,
        .layouts = layouts,
        .enum_layouts = enum_layouts,
        .instances = instances,
        .methods = methods,
        .templates = templates_out,
        .protocols = protocols,
        .derives = derives,
        .descriptor_types = descriptor_types,
        .erased_units = erased_units,
        .prelude_ids = gatherPreludeIds(t),
        .char_struct = if (t.prelude) |p| p.char_struct else null,
        .adapters = adapters,
        .lambda_units = lambda_units,
        .owner_closures = owner_closures,
        .mono_scanned_nodes = t.mono_scanned_nodes,
        .bodies = bodies,
        .fn_boxed = fn_boxed,
    };
}

/// Record each free-fn call's callee on its slot: `.direct` for a non-generic fn,
/// `.instance` for the instance a generic call runs (`Mono.instanceOf`), `.generic` for a
/// generic call with none here (one in a template's own body, or one that did not check).
/// A turbofish of a non-template, one of the core intrinsics like `size_of[T]()`, names no
/// code and stays `.none`. Runs once the instances, their names and the sigs are final,
/// over every module slice and every instance's own body, so lower, the fingerprint, the
/// escape pass, the spec table and the LSP all read one selection.
fn sealCalls(
    gpa: std.mem.Allocator,
    graph: *const GraphCtx,
    node_types: []const []const Type,
    callees: []const []Callee,
    instances: []const Mono.Instance,
    sigs: []const Sig,
) !void {
    const V = struct {
        gpa: std.mem.Allocator,
        view: Mono.CallView,
        /// The view's callees, writable.
        callees: []Callee,
        instances: *const Mono.Lookup,
        sigs: []const Sig,

        fn seal(self: *const @This(), call: Ast.Index) !void {
            const tree = self.view.tree;
            const n = tree.nodes[call.int()];
            if (n.tag != .call or n.lhs == Ast.none) return;
            const slot = Callee.callSlot(tree, n).?;
            if (!self.view.nodes.has(slot)) return;
            const out = &self.callees[slot.int() - self.view.nodes.base];
            if (out.tag != .none) return;
            const cn = tree.nodes[n.lhs.int()];
            switch (cn.tag) {
                .type_app, .identifier, .field_access => {},
                else => return,
            }
            const head = if (cn.tag == .type_app) cn.lhs else n.lhs;
            if (head == Ast.none or head.int() >= self.view.resolutions.len) return;
            const gid = switch (self.view.resolutions[head.int()]) {
                .func => |g| g,
                else => return,
            };
            const generic = gid < self.sigs.len and self.sigs[gid].isGenericTemplate();
            const c: Callee = if (cn.tag == .type_app) blk: {
                const targ_nodes = Ast.rangeSlice(tree, cn.rhs.int());
                const targs = try self.gpa.alloc(Type, targ_nodes.len);
                defer self.gpa.free(targs);
                for (targ_nodes, targs) |tn, *ta| ta.* = self.view.nodes.typeOf(tn);
                if (self.instances.find(gid, targs)) |ii| break :blk Callee.instanceCall(@intCast(ii));
                break :blk if (generic) Callee.genericCall(gid) else .{};
            } else if (!generic)
                Callee.direct(gid)
            else if (try Mono.instanceOf(self.gpa, self.view, self.instances, call)) |ii|
                Callee.instanceCall(@intCast(ii))
            else
                Callee.genericCall(gid);
            out.* = c;
        }

        pub fn on(self: *@This(), ev: Walk.Event) !void {
            switch (ev) {
                .enter => |e| if (e.tag == .call) try self.seal(e.idx),
                else => {},
            }
        }
    };
    var lookup = try Mono.Lookup.init(gpa, instances);
    defer lookup.deinit(gpa);
    for (graph.mods, node_types, callees) |*mc, nts, cs| {
        const v: V = .{ .gpa = gpa, .view = .{ .tree = mc.tree, .resolutions = mc.resolutions, .nodes = .{ .types = nts, .callees = cs }, .templates = .{ .sigs = sigs } }, .callees = cs, .instances = &lookup, .sigs = sigs };
        for (0..mc.tree.nodes.len) |i| try v.seal(Ast.Index.from(@intCast(i)));
    }
    for (instances) |inst| {
        const mc = &graph.mods[inst.mod];
        var v: V = .{ .gpa = gpa, .view = .{ .tree = mc.tree, .resolutions = mc.resolutions, .nodes = inst.nodes(), .templates = .{ .sigs = sigs } }, .callees = inst.callees, .instances = &lookup, .sigs = sigs };
        try Walk.walk(.{ .tree = mc.tree, .tokens = mc.tokens, .source = mc.source }, inst.decl_node, &v);
    }
}

fn freeFnBoxed(gpa: std.mem.Allocator, fn_boxed: []const []const u32) void {
    for (fn_boxed) |bx| gpa.free(bx);
    gpa.free(fn_boxed);
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
    // A selected alias resolves in its own module first, so the alias pass walks the
    // import DAG dependencies-first. A selected struct or enum is bound before any alias
    // can name it.
    for (t.graph.mods) |*m| for (m.selections) |sel| {
        if (sel.kind != .type) continue;
        const owner = &t.graph.mods[sel.target];
        if (owner.struct_ids.get(sel.name)) |id| {
            if (t.structs.items[id].mod == sel.target) try m.struct_ids.put(t.gpa, sel.local, id);
        } else if (owner.enum_ids.get(sel.name)) |id| {
            if (t.enums.items[id].mod == sel.target) try m.enum_ids.put(t.gpa, sel.local, id);
        }
    };
    const alias_order = try importPostOrder(t.gpa, t.graph.mods);
    defer t.gpa.free(alias_order);
    for (alias_order) |mod| {
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;
        const m = &t.graph.mods[mod];
        for (m.selections) |sel| {
            if (sel.kind != .type) continue;
            if (t.graph.mods[sel.target].alias_ids.get(sel.name)) |ty| try m.alias_ids.put(t.gpa, sel.local, ty);
        }
        try register.registerAliases(t, Ast.rangeSlice(t.tree, prog.lhs.int()));
    }

    // Prelude: native-register `Eq` (+ its builtin scalar conformances) BEFORE the
    // per-module protocol loop, so `Eq` is global id 0 and is bare-nameable everywhere
    // with no import (the `print` precedent — no module-graph/fingerprint surface).
    t.prelude = try prelude_reg.register(t.gpa, &t.protocols, &t.conformances, &t.enums, &t.structs, t.graph.mods);

    // Bare imported-type visibility: a `pub` struct/enum in an imported module is
    // nameable UNQUALIFIED in the importer (the `std/vec` `Vec` surface), mirroring the
    // prelude enum/struct injection. Injected AFTER every module's own decls + the
    // prelude, if-absent + pub-only, so a local or prelude name always shadows an import;
    // structs/enums are scanned in global-id order so a two-import name clash resolves to
    // the lowest id deterministically (a clashing bare name is only reachable when a
    // program actually writes it — the corpus references imports qualified, so this is
    // additive). Only the importer's OWN direct imports contribute (no transitive
    // re-export): the owning module must be one this importer imports.
    for (mods, 0..) |_, mi| {
        const importer = &t.graph.mods[mi];
        for (t.structs.items, 0..) |s, sid| {
            if (!s.pub_export or !importerImports(importer, s.mod)) continue;
            if (importer.struct_ids.get(s.name) == null)
                try importer.struct_ids.put(t.gpa, s.name, @intCast(sid));
        }
        for (t.enums.items, 0..) |e, eid| {
            if (!e.pub_export or !importerImports(importer, e.mod)) continue;
            if (importer.enum_ids.get(e.name) == null)
                try importer.enum_ids.put(t.gpa, e.name, @intCast(eid));
        }
        // A type a direct import re-exports is bare too, as if that module declared it.
        for (t.graph.mods, 0..) |*om, oi| {
            if (!importerImports(importer, @intCast(oi))) continue;
            for (om.selections) |sel| {
                if (!sel.exported or sel.kind != .type) continue;
                if (om.struct_ids.get(sel.local)) |id| {
                    if (importer.struct_ids.get(sel.local) == null) try importer.struct_ids.put(t.gpa, sel.local, id);
                } else if (om.enum_ids.get(sel.local)) |id| {
                    if (importer.enum_ids.get(sel.local) == null) try importer.enum_ids.put(t.gpa, sel.local, id);
                }
            }
        }
    }

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
    for (t.graph.mods) |*m| for (m.selections) |sel| {
        if (sel.kind != .protocol) continue;
        const id = t.graph.mods[sel.target].protocol_ids.get(sel.name) orelse continue;
        if (t.protocols.items[id].mod == sel.target) try m.protocol_ids.put(t.gpa, sel.local, id);
    };

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

    try t.decodeImplHeads(mods);

    // Phase A: decode every fn signature into the GLOBAL fn table, in the exact
    // order of `fns` (parallel to the resolver's global fn ids), so `.func` ids
    // index this table directly. The synthetic bodyless `panic` is one of them.
    for (fns) |gf| {
        if (gf.kind == .builtin) {
            try t.appendBuiltinSig(if (std.mem.eql(u8, gf.name, "panic")) Type.never else Type.unit);
        } else if (gf.kind == .import) {
            _ = t.gphSelect(gf.module);
            try t.decodeExternSig(gf.decl_node, gf.module);
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

    // After coherence, which drops an overlapping impl's method rows, and before the
    // derive barrier and the mono tail append method rows no source declares.
    try t.checkStdShadows();
    try t.checkFnFieldMethods();
    try t.checkVariantNamedFns();

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
/// body-checked; an instance re-check grounds its `type_var`s via `bc.subst`). Admits
/// `unit`: `()` is a concrete zero-sized value type-arg (as in `Map[T, ()]`), reified
/// to a 0-byte field/payload. Admits a function type (reified to its pair struct). Still
/// excludes `type_var` and poison.
fn isConcreteValue(ty: Type) bool {
    return switch (ty.kind) {
        .int, .bool, .float, .str, .@"struct", .@"enum", .app, .unit, .func => true,
        else => false,
    };
}

/// Substitute a template type through a concrete arg tuple: a `type_var(ord)`
/// becomes `args[ord]`; an `App(ctor, [pat..])` or a fn type recursively substitutes
/// each arg and re-interns (so a template param spelled `Box[T]` grounds to `Box[int]`);
/// anything else passes through unchanged.
fn substType(t: *Typecheck, ty: Type, args: []const Type) Type {
    if (ty.isTypeVar()) {
        const ord = ty.typeVarOrd();
        return if (ord < args.len) args[ord] else Type.invalid;
    }
    return refs.mapComposite(t, ty, struct {
        t: *Typecheck,
        args: []const Type,
        pub fn map(m: @This(), x: Type) Type {
            return m.t.substType(x, m.args);
        }
    }{ .t = t, .args = args });
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

    // Seed: scan each module that holds a non-generic user fn for call-position generic
    // calls, reading the Pass-C-populated per-module node_types, in the order of each
    // module's first such fn. A scan per fn would see its whole module each time (only
    // template bodies stay untyped in it), and `seen` would drop every repeat: O(fns x
    // calls) for the same worklist.
    const nts = t.gph_node_types orelse return; // graph mode always sets it
    const cs = t.gph_callees.?;
    // Each template's facts as the checker holds them: `generic_params.len` is the count.
    const template_refs = try t.gpa.alloc(?Mono.TemplateRef, model.fns.len);
    defer t.gpa.free(template_refs);
    for (model.fns, template_refs) |f, *r| r.* = if (f.isGeneric()) .{ .params = f.params, .count = @intCast(f.generic_params.len) } else null;
    const templates: Mono.Templates = .{ .refs = .{ .of = template_refs, .decomposer = t.composite.decomposer() } };
    const seeded = try t.gpa.alloc(bool, nts.len);
    defer t.gpa.free(seeded);
    @memset(seeded, false);
    for (model.fns) |f| {
        if (f.kind != .user_fn or f.isGeneric()) continue; // bodyless builtins/externs have no calls
        if (seeded[f.mod]) continue;
        seeded[f.mod] = true;
        try t.scanCalls(model, templates, f.mod, nts[f.mod], cs[f.mod], .{ .start = 0, .end = @intCast(nts[f.mod].len) }, &worklist, &seen);
    }
    // Each template's decl subtree range, computed once and shared by its instances.
    const spans = try t.gpa.alloc(?NodeRange, model.fns.len);
    defer t.gpa.free(spans);
    @memset(spans, null);
    // One module-sized pair of tables per module, which every instance re-check in it
    // writes and resets: the instance keeps only its template's decl range (`Body.Nodes`).
    const scratch = try t.gpa.alloc(?NodeSlots, nts.len);
    defer {
        for (scratch) |sc| if (sc) |x| {
            t.gpa.free(x.node_types);
            t.gpa.free(x.callees);
        };
        t.gpa.free(scratch);
    }
    @memset(scratch, null);

    var visited: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer visited.deinit(t.gpa);
    var demanded: usize = 0;
    try t.drainDemands(model, &demanded, &worklist, &seen, &visited);
    var capped = false;
    var idx: usize = 0;
    while (idx < worklist.items.len) : (idx += 1) {
        if (t.mono.items.len >= mono_instance_cap) {
            if (!capped) {
                const f = model.fns[worklist.items[idx].gid];
                _ = t.gphSelect(f.mod);
                const decl = t.tree.nodes[f.decl_node.int()];
                try t.sink.report(.{ .span = t.tokSpan(decl.main_token), .code = .T0014 }, "monomorphization instance limit exceeded", .{});
                capped = true;
            }
            break;
        }
        const p = worklist.items[idx];
        const f = model.fns[p.gid];
        const span = spans[p.gid] orelse blk: {
            const r = try t.declRange(f.mod, f.decl_node);
            spans[p.gid] = r;
            break :blk r;
        };
        const sc = scratch[f.mod] orelse blk: {
            const n = t.graph.mods[f.mod].tree.nodes.len;
            const sc_nt = try t.gpa.alloc(Type, n);
            errdefer t.gpa.free(sc_nt);
            const x: NodeSlots = .{ .node_types = sc_nt, .callees = try t.gpa.alloc(Callee, n) };
            @memset(x.node_types, .invalid);
            @memset(x.callees, .{});
            scratch[f.mod] = x;
            break :blk x;
        };
        const inst = try t.recheck(model, p.gid, p.args, span, sc);
        try t.mono.append(t.gpa, inst);
        // Discover nested generic calls in the instance body, on the tables it was checked into.
        try t.scanCalls(model, templates, inst.mod, sc.node_types, sc.callees, span, &worklist, &seen);
        try t.drainDemands(model, &demanded, &worklist, &seen, &visited);
        @memset(sc.node_types[span.start..span.end], .invalid);
        @memset(sc.callees[span.start..span.end], .{});
    }

    // Reification: mint a fresh concrete `struct_id` for every reachable ground
    // `App` (in all node_types + instance sigs + non-generic fn sigs) and REWRITE
    // every `.app` to that `structT`. Serial, BEFORE the sort/mangle (so instance
    // args are plain `structT` for `Mangle.instance`) and BEFORE the layout snapshot (so
    // codegen/fingerprint/cache see only concrete structs). Ids are assigned in an
    // index-INDEPENDENT structural-key order, so they are a pure function of source.
    try reify.reifyApps(t, nts);

    // Canonical order (template gid, then arg bytes) drives BOTH the mangled-name
    // assignment and the downstream codegen enumeration — never discovery order.
    std.mem.sort(Mono.Instance, t.mono.items, {}, Mono.lessThan);
    for (t.mono.items) |*inst| {
        const tname = if (t.gph_fn_names) |fns| fns[inst.template_gid] else t.nameText(t.tree.nodes[model.fns[inst.template_gid].decl_node.int()].main_token);
        inst.name = try Mangle.instance(t.gpa, tname, inst.args);
    }
    // A mangled name is a pure function of (template name, arg tuple); two distinct
    // instances therefore never collide (Debug/ReleaseSafe guard).
    if (std.debug.runtime_safety) {
        var names: std.StringHashMapUnmanaged(void) = .empty;
        defer names.deinit(t.gpa);
        for (t.mono.items) |inst| std.debug.assert(!(try names.getOrPut(t.gpa, inst.name.?)).found_existing);
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
    // which is what `sealCallees` keys off to resolve a pending method call, and what
    // lower's desugar lookups (`[]`, `for`-in) key off.
    if (t.templates.items.len > 0) {
        for (t.mono.items, 0..) |inst, i| {
            for (t.templates.items) |tmpl| {
                if (tmpl.fn_id != inst.template_gid) continue;
                // An instance method's receiver is its (reified) `self` param; an
                // ASSOCIATED function has no `self` param, so its dispatch receiver is
                // the impl's own reified type (`Vec[int]` for `Vec[T].new()`), computed
                // from the template's `self_type` substituted through the instance args.
                const recv: Type = if (tmpl.has_self)
                    inst.params[0]
                else
                    (try t.reifiedStaticRecv(model.fns[tmpl.fn_id].self_type, inst.args) orelse continue);
                const pargs = if (tmpl.protocol_id != null) try t.groundProtocolArgs(tmpl.protocol_args, inst.args) else null;
                errdefer if (pargs) |a| t.gpa.free(a);
                try t.methods.append(t.gpa, .{
                    .recv = recv,
                    .name = tmpl.name,
                    .fn_id = tmpl.fn_id,
                    .mut_self = tmpl.mut_self,
                    .is_static = !tmpl.has_self,
                    .instance = @intCast(i),
                    .bundled = tmpl.bundled,
                    .shadows = tmpl.shadows,
                    .protocol_id = if (pargs != null) tmpl.protocol_id else null,
                    .protocol_args = pargs orelse &.{},
                });
                // The instance witnesses a generic conformance of the reified type: give
                // it the concrete row every post-reify query reads (the derive synthesis
                // then reuses this witness instead of deriving one).
                if (pargs) |a| if (tmpl.has_self) {
                    const pid = tmpl.protocol_id.?;
                    if (!conform.existsIn(t.conformances.items, pid, recv, a))
                        try t.conformances.append(t.gpa, .{ .protocol = pid, .recv = recv, .protocol_args = try t.gpa.dupe(Type, a) });
                    // Ord refines Eq, as `deriveEqFromOrd` records before reify.
                    if (t.prelude) |pre| if (pre.protocols.ord == pid) if (pre.protocols.eq) |eq_pid| if (!conform.existsIn(t.conformances.items, eq_pid, recv, &.{}))
                        try t.conformances.append(t.gpa, .{ .protocol = eq_pid, .recv = recv });
                };
                break;
            }
        }
    }

    // Auto-derive SYNTHESIS BARRIER. Same realloc-safe point as the append
    // above (nothing reads `model.methods` again): turn Pass C's derive requests into
    // canonical source-less recipes + their synthetic method-table entries.
    try derive_synth.synthesizeDerives(t);

    // Descriptor synthesis runs AFTER derive synthesis so the natural Hash/Eq recipes +
    // methods the erased walk delegates to already exist.
    try derive_synth.synthesizeDescriptors(t);

    t.sealCallees();
    try t.unifyInstanceHints();

    // The mono tail may have emitted instance-body diagnostics (and T0014); re-sort
    // the shared stream so the final (scope, byte_offset) order is deterministic.
    t.sink.sort();
}

/// An unbounded template's body is checked once per instance, so a report in it names
/// each instance's own types: one report per instance, and following a hint in one
/// breaks the others. Where the reports at one site differ but all read alike as the
/// template spells them, each takes that text, and the sink's dedup keeps one. Where
/// they agree already, a concrete type that one instance's binding merely equals stays
/// as written. Where they differ and the template spellings do too (a concrete type
/// equal to one instance's arg spells as the param there), the site still reports once,
/// as its shortest instance text: polymorphic recursion makes one instance per depth.
fn unifyInstanceHints(t: *Typecheck) !void {
    defer {
        for (t.instance_hints.items) |h| t.gpa.free(h.template);
        t.instance_hints.clearRetainingCapacity();
    }
    const diags = t.sink.diags.items;
    for (t.instance_hints.items) |*h| {
        if (h.template.len == 0) continue;
        const d = for (diags) |d| {
            if (d.message.ptr == h.message.ptr) break d;
        } else continue;
        var differ = false;
        var alike = true;
        for (diags) |e| if (e.scope == d.scope and e.byte_offset == d.byte_offset and e.code == d.code) {
            if (!std.mem.eql(u8, e.message, d.message)) differ = true;
            if (!std.mem.eql(u8, t.templateText(e.message), h.template)) alike = false;
        };
        if (!differ) continue;
        const text = if (alike) adopted: {
            try t.sink.adopt(h.template);
            const tmpl = h.template;
            h.template = &.{};
            break :adopted tmpl;
        } else shortest: {
            var best = d.message;
            for (diags) |e| if (e.scope == d.scope and e.byte_offset == d.byte_offset and e.code == d.code) {
                if (!t.hasInstanceHint(e.message)) break :shortest null;
                if (e.message.len < best.len or (e.message.len == best.len and std.mem.lessThan(u8, e.message, best))) best = e.message;
            };
            break :shortest best;
        } orelse continue;
        for (diags) |*e| if (e.scope == d.scope and e.byte_offset == d.byte_offset and e.code == d.code) {
            e.message = text;
        };
    }
}

/// Whether `message` is an instance re-check's report that names the instance's types.
fn hasInstanceHint(t: *const Typecheck, message: []const u8) bool {
    for (t.instance_hints.items) |h| if (h.message.ptr == message.ptr) return true;
    return false;
}

/// `message` as its instance's template spells it: its recorded spelling, else itself.
fn templateText(t: *const Typecheck, message: []const u8) []const u8 {
    for (t.instance_hints.items) |h| if (h.message.ptr == message.ptr) return if (h.template.len == 0) message else h.template;
    return message;
}

/// A generic conformance's arg pattern grounded through a reified instance's args, OWNED
/// by the caller (empty for a zero-arg protocol). Null when an arg nests a composite: its
/// ground App was never reified, so no concrete row can name it.
fn groundProtocolArgs(t: *Typecheck, pattern: []const Type, args: []const Type) !?[]const Type {
    const out = try t.gpa.alloc(Type, pattern.len);
    for (pattern, out) |p, *o| {
        o.* = t.substType(p, args);
        if (o.isApp() or o.isFunc()) {
            t.gpa.free(out);
            return null;
        }
    }
    return out;
}

/// Resolve every pending slot: only now are receiver types reified and every method row appended.
fn sealCallees(t: *Typecheck) void {
    for (t.gph_node_types.?, t.gph_callees.?, 0..) |nts, cs, m| t.sealSlice(@intCast(m), .{ .types = nts, .callees = cs }, cs);
    for (t.mono.items) |inst| t.sealSlice(inst.mod, inst.nodes(), inst.callees);
    if (builtin.mode == .Debug) t.sealed_methods_len = t.methods.items.len;
}

/// Each slot is a pure lookup over the final method table by reified receiver and
/// member (and, for a call of a method with its own type params, by the call's reified
/// arg types), so the order slots are sealed in cannot matter. An unresolvable slot
/// becomes `.none` rather than a panic: `toy check` accepts such a program and only
/// codegen rejects it.
fn sealSlice(t: *const Typecheck, mod: u32, nodes: Body.Nodes, cs: []Callee) void {
    const mc = &t.graph.mods[mod];
    const site = Site.of(mc.bundled);
    for (cs, nodes.base..) |*c, i| {
        if (!c.isPending() or c.tag == .pending_template) continue;
        const fa = mc.tree.nodes[i];
        const member = mc.tokens[fa.main_token].text(mc.source);
        if (c.tag == .pending_call) {
            const crumb = nodes.callee(Ast.Index.from(c.id));
            c.* = t.pickInstanceRow(mc, nodes, fa, member, c.id, if (crumb.tag == .pending_template) crumb else null);
            continue;
        }
        c.* = switch (pickMethodIndex(t.methods.items, nodes.typeOf(fa.lhs), member, null, null, site)) {
            .one => |mi| if (c.tag == .pending_bound) Callee.boundMethod(mi) else Callee.method(mi),
            .none, .ambiguous => .{},
        };
    }
    for (cs) |*c| if (c.tag == .pending_template) {
        c.* = .{};
    };
}

/// The instance row an own-generic call selects: the first row, in table order, with an
/// instance, the call's reified receiver and member, whose instance params accept the
/// call's reified args (`Type.assignable`, as `Mono.findByReifiedParams`). Rows are tried
/// in the order `findGenericMethod` gives their templates at the call's site. A static
/// row's params are all value params; an instance method's `params[0]` is `self`. Every
/// row is matched, never just the only candidate, so a check/discovery disagreement
/// surfaces as a lowering error rather than a wrong instance. A call whose args did not
/// check picks no row (another call's instance would show the wrong types) but keeps its
/// template. A call with explicit type args (`crumb`) takes only its template's rows, and
/// when the args are the method's own, the row whose instance has them: a type param
/// that no value param names (`fn none[U](self) -> Option[U]`) leaves the arg types alike.
fn pickInstanceRow(t: *const Typecheck, mc: *const GraphCtx.ModuleCtx, nodes: Body.Nodes, fa: Ast.Node, member: []const u8, call: u32, crumb: ?Callee) Callee {
    const recv = nodes.typeOf(fa.lhs);
    const site = Site.of(mc.bundled);
    const call_node = mc.tree.nodes[call];
    const args = Ast.rangeSlice(mc.tree, call_node.rhs.int());
    for (args) |a| if (nodes.typeOf(a).kind == .invalid) return t.templateCallee(recv, member, site);
    const explicit: []const Ast.Index = if (crumb != null and crumb.?.aux == 1) Ast.rangeSlice(mc.tree, mc.tree.nodes[call_node.lhs.int()].rhs.int()) else &.{};
    for ([_]u1{ 0, 1 }) |pass| {
        rows: for (t.methods.items, 0..) |m, mi| {
            const ii = m.instance orelse continue;
            if (siteRank(m.bundled, m.shadows, site) != pass) continue;
            if (!Type.eql(m.recv, recv) or !std.mem.eql(u8, m.name, member)) continue;
            if (crumb) |cr| {
                const inst = t.mono.items[ii];
                if (inst.template_gid != cr.id or inst.args.len < explicit.len) continue;
                for (inst.args[inst.args.len - explicit.len ..], explicit) |ia, e| if (!Type.eql(ia, nodes.typeOf(e))) continue :rows;
            }
            const ps = t.mono.items[ii].params;
            const off: usize = if (m.is_static) 0 else 1;
            if (ps.len != args.len + off) continue;
            for (ps[off..], args) |p, a| if (!Type.assignable(p, nodes.typeOf(a))) continue :rows;
            return Callee.method(@intCast(mi));
        }
    }
    return t.templateCallee(recv, member, site);
}

/// A method row's place in a lookup at `site`: 0 for the first candidates, 1 for the
/// fallbacks, null when it is no candidate there (see `Site`).
fn siteRank(bundled: bool, shadows: bool, site: Site) ?u1 {
    return switch (site) {
        .program => if (bundled) 1 else 0,
        .bundled => if (shadows) null else 0,
    };
}

/// `Callee.template` for the generic method `member` of `recv` itself (a non-generic
/// struct or enum) or of the type `recv` was reified from, else none. Only a call that already failed to check gets here, so the scan of the
/// reify table is off the hot path.
fn templateCallee(t: *const Typecheck, recv: Type, member: []const u8, site: Site) Callee {
    if (recv.kind != .@"struct" and recv.kind != .@"enum") return .{};
    if (findGenericMethod(t.templates.items, recv.nominalId(), recv.kind == .@"enum", member, site)) |m| return Callee.template(m.fn_id);
    var it = t.reify_map.iterator();
    while (it.next()) |kv| {
        if (!Type.eql(kv.value_ptr.*, recv)) continue;
        const e = t.composite.at(kv.key_ptr.*);
        if (e.is_fn) continue;
        const m = findGenericMethod(t.templates.items, e.ctor, e.ctor_is_enum, member, site) orelse return .{};
        return Callee.template(m.fn_id);
    }
    return .{};
}

/// Every module, each after all the modules it imports. The import graph is a DAG.
fn importPostOrder(gpa: std.mem.Allocator, mods: []const GraphCtx.ModuleCtx) ![]u32 {
    const order = try gpa.alloc(u32, mods.len);
    errdefer gpa.free(order);
    const seen = try gpa.alloc(bool, mods.len);
    defer gpa.free(seen);
    @memset(seen, false);
    // Explicit stack of (module, next import index), so a deep chain cannot overflow.
    var stack: std.ArrayList([2]u32) = .empty;
    defer stack.deinit(gpa);
    var n: usize = 0;
    for (0..mods.len) |root| {
        if (seen[root]) continue;
        seen[root] = true;
        try stack.append(gpa, .{ @intCast(root), 0 });
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const imps = mods[top[0]].imports;
            if (top[1] < imps.len) {
                const next = imps[top[1]];
                top[1] += 1;
                if (next < mods.len and !seen[next]) {
                    seen[next] = true;
                    try stack.append(gpa, .{ next, 0 });
                }
                continue;
            }
            order[n] = top[0];
            n += 1;
            _ = stack.pop();
        }
    }
    return order;
}

/// Whether module `importer` directly imports the module with id `target` (any namespace
/// binding resolves to it). Used to gate bare imported-type visibility to a module's own
/// direct imports.
fn importerImports(importer: *const GraphCtx.ModuleCtx, target: u32) bool {
    var it = importer.namespaces.valueIterator();
    while (it.next()) |v| if (v.* == target) return true;
    return false;
}

/// The reified concrete receiver an ASSOCIATED function dispatches off: the template's
/// `self_type` App (`Vec[T]`) over the instance's already-reified type-args, looked up by
/// its grounded key. Returns null when `self_type` is not an App or that instance was
/// never reified (defensive: an instance's own signature keeps its receiver reachable).
fn reifiedStaticRecv(t: *Typecheck, self_type: Type, args: []const Type) error{OutOfMemory}!?Type {
    if (self_type.kind == .@"struct" or self_type.kind == .@"enum") return self_type;
    if (!self_type.isApp()) return null;
    const e = t.composite.at(self_type.appIdx());
    var abuf: [8]Type = undefined;
    if (e.args.len > abuf.len) return null;
    const sub = abuf[0..e.args.len];
    for (e.args, 0..) |a, i| {
        if (a.isTypeVar()) {
            const ord = a.typeVarOrd();
            if (ord >= args.len) return null;
            sub[i] = args[ord];
        } else if (a.isApp() or a.isFunc()) {
            sub[i] = t.reify_map.get(a.struct_id) orelse return null;
        } else sub[i] = a;
    }
    const key = try reify.groundedAppKey(t, e.ctor, e.ctor_is_enum, sub);
    defer t.gpa.free(key);
    return t.app_reify_dedup.get(key);
}

/// Bind an impl's type-params by matching the template method `mf`'s `Self` pattern
/// (`Vec[T]`) against a receiver App's concrete `recv_args`, via the same `Infer.match`
/// Pass C runs. Returns a freshly-allocated slice of the bound params (length
/// `mf.generic_params.len`, the CALLER owns and frees it) on a successful match, or null
/// when the pattern arity disagrees or the match conflicts / leaves a param unbound.
/// Concreteness of the bound args is the caller's to check.
pub fn bindImplParams(gpa: std.mem.Allocator, composite: *Composite, mf: FnSym, recv_args: []const Type) error{OutOfMemory}!?[]Type {
    const pat: []const Type = if (mf.self_type.isApp()) composite.at(mf.self_type.appIdx()).args else &.{};
    if (pat.len != recv_args.len) return null;
    const n_gp: u32 = @intCast(mf.generic_params.len);
    const out = try gpa.alloc(Type, n_gp);
    const bnd = try gpa.alloc(bool, n_gp);
    defer gpa.free(bnd);
    const fp = try gpa.alloc(usize, n_gp);
    defer gpa.free(fp);
    switch (Infer.match(n_gp, pat, recv_args, out, bnd, fp, composite.decomposer())) {
        .ok => return out,
        else => {
            gpa.free(out);
            return null;
        },
    }
}

/// `bindImplParams` for a method with type params of its own: the receiver binds the
/// first `n_impl` (the impl's); the rest of the returned slice (OWNED by the caller) is
/// left for the caller to fill.
fn bindImplParamsPrefix(gpa: std.mem.Allocator, composite: *Composite, mf: FnSym, recv_args: []const Type, n_impl: u32) error{OutOfMemory}!?[]Type {
    const pat: []const Type = if (mf.self_type.isApp()) composite.at(mf.self_type.appIdx()).args else &.{};
    if (pat.len != recv_args.len) return null;
    const out = try gpa.alloc(Type, mf.generic_params.len);
    errdefer gpa.free(out);
    const bnd = try gpa.alloc(bool, n_impl);
    defer gpa.free(bnd);
    const fp = try gpa.alloc(usize, n_impl);
    defer gpa.free(fp);
    switch (Infer.match(n_impl, pat, recv_args, out[0..n_impl], bnd, fp, composite.decomposer())) {
        .ok => return out,
        else => {
            gpa.free(out);
            return null;
        },
    }
}

/// The impl's own type params at the front of `mf.generic_params` (the parser puts them
/// first): one past the largest `type_var` ordinal among the `Self` pattern's args, 0
/// when the `Self` type is not an `App`. Any params past it are the method's own.
pub fn implParamCount(composite: *Composite, mf: FnSym) u32 {
    if (!mf.self_type.isApp()) return 0;
    var n: u32 = 0;
    for (composite.at(mf.self_type.appIdx()).args) |a| {
        if (a.isTypeVar()) n = @max(n, a.typeVarOrd() + 1);
    }
    return n;
}

/// Whether a method of a generic type declares type params of its own (`fn map[U]`),
/// which its call's args bind rather than its receiver.
pub fn hasOwnTypeParams(composite: *Composite, mf: FnSym) bool {
    return implParamCount(composite, mf) < mf.generic_params.len;
}

/// The type args of a call of a method with its own type params (`b.map(f)` for
/// `fn map[U](self, f: fn(T) -> U)`, or `Box.with(3, true)`): one left-to-right match of
/// `[Self] ++ params` against `[recv] ++` the value args' `node_types`, the final match
/// Pass C's `inferArgs` runs, so both pick one instance. Null for any other method or
/// call; the caller owns the slice.
fn ownGenericArgs(gpa: std.mem.Allocator, composite: *Composite, mf: FnSym, has_self: bool, tree: Ast.Tree, node_types: []const Type, call: Ast.Node, recv: Type) error{OutOfMemory}!?[]Type {
    if (!hasOwnTypeParams(composite, mf)) return null;
    const args = Ast.rangeSlice(tree, call.rhs.int());
    const n_params = mf.params.len + @intFromBool(!has_self);
    if (args.len + 1 != n_params) return null;
    const params = try gpa.alloc(Type, n_params);
    defer gpa.free(params);
    if (has_self) @memcpy(params, mf.params) else {
        params[0] = mf.self_type;
        @memcpy(params[1..], mf.params);
    }
    const arg_types = try gpa.alloc(Type, n_params);
    defer gpa.free(arg_types);
    arg_types[0] = recv;
    for (args, arg_types[1..]) |a, *at| at.* = node_types[a.int()];
    return Infer.infer(gpa, @intCast(mf.generic_params.len), params, arg_types, composite.decomposer());
}

/// A half-open run `[start, end)` of one module's node indices.
const NodeRange = struct { start: u32, end: u32 };

/// The node range of the decl subtree at `decl` in module `mod`: every node an instance
/// re-check of that fn can type. The parser builds children before their parent, so the
/// decl is the subtree's last node; the walk finds its first (an impl method's receiver
/// type is the impl's, built before the method).
fn declRange(t: *const Typecheck, mod: u32, decl: Ast.Index) !NodeRange {
    const mc = &t.graph.mods[mod];
    var v: struct {
        lo: u32 = std.math.maxInt(u32),
        hi: u32 = 0,
        pub fn on(self: *@This(), ev: Walk.Event) void {
            switch (ev) {
                .enter => |e| {
                    self.lo = @min(self.lo, e.idx.int());
                    self.hi = @max(self.hi, e.idx.int());
                },
                else => {},
            }
        }
    } = .{};
    try Walk.walk(.{ .tree = mc.tree, .tokens = mc.tokens, .source = mc.source }, decl, &v);
    return .{ .start = v.lo, .end = v.hi + 1 };
}

/// Scan module `mod`'s nodes in `range` for call-position generic calls (and generic-type
/// methods bound as values, from `callees`), reading `node_types` for the concrete
/// type-args, and enqueue each new `(gid, args)`. Uniform for the base seed (a whole
/// module's node_types) and an instance re-check (the instance's own node_types over its
/// template's decl range — nodes outside the body stay `.invalid` there, so only THIS
/// body's calls are seen).
fn scanCalls(t: *Typecheck, model: *const Model, templates: Mono.Templates, mod: u32, node_types: []const Type, callees: []const Callee, range: NodeRange, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void)) !void {
    const mc = &t.graph.mods[mod];
    const tree = mc.tree;
    const resolutions = mc.resolutions;
    const view: Mono.CallView = .{ .tree = tree, .resolutions = resolutions, .nodes = .{ .types = node_types, .callees = callees }, .templates = templates };
    const nodes = tree.nodes[range.start..range.end];
    t.mono_scanned_nodes += nodes.len;
    for (nodes, range.start..) |n, ci| {
        if (n.tag != .call or n.lhs == Ast.none) continue;
        const call_at: Anchor = .{ .node = Ast.Index.from(@intCast(ci)) };
        // The explicit-turbofish `id[int](..)` and bare-inferred `id(7)` shapes route
        // through the SAME arg-extraction the fingerprint fold uses (`callInstanceRef`),
        // so discovery selects the exact instance the caller's fingerprint targets.
        if (try Mono.callInstanceRef(t.gpa, view, n)) |ref| {
            defer t.gpa.free(ref.args);
            // Turbofish arity (a no-op for the inferred shape + any valid turbofish):
            // guards a wrong-arity `id[int,bool]()` on an already-erroring program.
            if (ref.args.len != model.fns[ref.gid].generic_params.len) continue;
            var conc = true;
            for (ref.args) |ta| if (!isConcreteValue(ta)) {
                conc = false;
                break;
            };
            if (!conc) continue;
            try t.enqueueInstance(model, ref.gid, ref.args, worklist, seen, call_at, mod);
            continue;
        }
        const callee = tree.nodes[n.lhs.int()];
        // A method call with explicit type args on a generic-type instance (`b.none[int]()`,
        // `b.into[int]()`): the args pick the template and fill its params as the checker's
        // `pickExplicitTemplate` did.
        if (callee.tag == .type_app and tree.nodes[callee.lhs.int()].tag == .field_access) {
            const fa = tree.nodes[callee.lhs.int()];
            const re = templateRecv(t.composite, node_types[fa.lhs.int()]) orelse continue;
            const targ_nodes = Ast.rangeSlice(tree, callee.rhs.int());
            const explicit = try t.gpa.alloc(Type, targ_nodes.len);
            defer t.gpa.free(explicit);
            for (targ_nodes, explicit) |tn, *e| e.* = node_types[tn.int()];
            const member = mc.tokens[fa.main_token].text(mc.source);
            switch (try pickExplicitTemplate(t.gpa, t.composite, model.templates, model.fns, re.ctor, re.is_enum, re.args, member, Site.of(mc.bundled), explicit)) {
                .ok => |p| {
                    defer t.gpa.free(p.targs);
                    const conc = for (p.targs) |ta| {
                        if (!isConcreteValue(ta)) break false;
                    } else true;
                    if (conc) try t.enqueueInstance(model, p.m.fn_id, p.targs, worklist, seen, call_at, mod);
                },
                else => {},
            }
            continue;
        }
        if (callee.tag == .field_access) {
            // A method call on a generic-type instance `b.get()`: the receiver
            // types to an `App` (reify runs later in this tail), whose ctor selects the
            // `impl Box[T]` method TEMPLATE. Bind the impl's type-params by matching the
            // template's `Self` pattern-args against the receiver App's concrete args —
            // the SAME `Infer.match` Pass C runs — then enqueue `(method_gid, targs)`.
            // A method with its own `[U]` binds them from the call's value args too; on a
            // non-generic struct or enum those are all it has.
            const recv = node_types[callee.lhs.int()];
            const re = templateRecv(t.composite, recv) orelse continue;
            const member = mc.tokens[callee.main_token].text(mc.source);
            const m = switch (pickGenericMethod(model.templates, re.ctor, re.is_enum, member, Site.of(mc.bundled))) {
                .one => |m| m,
                .none => continue,
                // The checker picked one through a bound (`boundTemplate`) and left it here.
                .ambiguous => blk: {
                    const crumb = callees[ci];
                    if (crumb.tag != .pending_template) continue;
                    for (model.templates) |tm| if (tm.fn_id == crumb.id) break :blk tm;
                    continue;
                },
            };
            const mf = model.fns[m.fn_id];
            const out = (if (hasOwnTypeParams(t.composite, mf))
                try ownGenericArgs(t.gpa, t.composite, mf, m.has_self, tree, node_types, n, recv)
            else
                try bindImplParams(t.gpa, t.composite, mf, re.args)) orelse continue;
            defer t.gpa.free(out);
            var conc = true;
            for (out) |ta| {
                if (!isConcreteValue(ta)) {
                    conc = false;
                    break;
                }
            }
            if (!conc) continue;
            try t.enqueueInstance(model, m.fn_id, out, worklist, seen, call_at, mod);
        }
    }
    // An annotated empty-list literal `xs: Vec[int] = []` desugars to the type's
    // associated `new()`: discover that instance from the literal's typed `Vec[int]`
    // receiver, mirroring the `Type[int].new()` field_access branch above.
    for (nodes, range.start..) |n, i| {
        if (n.tag != .empty_list) continue;
        for (StdNames.desugarMethods(.empty_list)) |dm| {
            const recv = desugarRecvType(node_types, n, i, dm.recv);
            try t.discoverRecvMethod(model, recv, dm.name, .program, worklist, seen, .{ .token_of = Ast.Index.from(@intCast(i)) }, mod);
        }
    }
    var prim_gids: [std.enums.values(StdNames.ElemPrim).len]?u32 = undefined;
    for (std.enums.values(StdNames.ElemPrim), &prim_gids) |prim, *g| g.* = elemPrimGid(model, prim);
    // A non-empty list literal `[e0, ..]` desugars to `new()` + one `push` per element
    // over its typed `Vec[V]` receiver, and a `for x in xs` desugars to `it := xs.iter()`
    // then a loop over `it.next()`: discover the template instances lower will call (the
    // transitive `Vec$T.get`/`mem.ga_*`/`Option$T` follow from re-checking each body).
    // The method set + receiver source come from the shared `desugarMethods` table so
    // discovery enqueues exactly what lower emits. Lower resolves a desugar at the
    // program's site wherever it is written, so discovery does too.
    for (nodes, range.start..) |n, i| {
        switch (n.tag) {
            .list_literal, .for_in_stmt, .for_in2_stmt => {
                for (StdNames.desugarMethods(n.tag)) |dm| {
                    const recv = desugarRecvType(node_types, n, i, dm.recv);
                    // A `for` node is the whole loop, body included; it keeps its token.
                    const at: Anchor = if (n.tag == .list_literal) .{ .node = Ast.Index.from(@intCast(i)) } else .{ .token_of = Ast.Index.from(@intCast(i)) };
                    try t.discoverRecvMethod(model, recv, dm.name, .program, worklist, seen, at, mod);
                }
            },
            // `xs[i]` desugars to `mem.ga_at[V]`, the store `xs[i] = v` to `mem.ga_set_at[V]`,
            // and an element read and written back to both, as the checker marked it;
            // discover those instances keyed on the index node's element type V so lower
            // can resolve their mangled symbols. A value-headed
            // `type_app` is a value index the parser could not tell from a turbofish
            // (`b.items[i].f`) and lowers the same way. A real type-app's leftmost
            // identifier is not `.local`, so this never fires for one.
            .index, .type_app => if (n.tag == .index or symbols_res.leftmostHeadIsValue(tree, resolutions, n.lhs)) {
                const elem = node_types[i];
                const at = Ast.Index.from(@intCast(i));
                const mark: Callee = if (i < callees.len) callees[i] else .{};
                if (isConcreteValue(elem)) for (StdNames.ElemPrim.of(mark)) |prim| if (prim_gids[@intFromEnum(prim)]) |gid| {
                    const args = [_]Type{elem};
                    try t.enqueueInstance(model, gid, &args, worklist, seen, .{ .node = at }, mod);
                };
            },
            // A bound method on a generic-type receiver (`get := b.get`) calls the method's
            // instance from its adapter, though no call names it.
            .field_access => if (i < callees.len and callees[i].tag == .pending_bound) {
                const member = mc.tokens[n.main_token].text(mc.source);
                try t.discoverRecvMethod(model, node_types[n.lhs.int()], member, Site.of(mc.bundled), worklist, seen, .{ .node = Ast.Index.from(@intCast(i)) }, mod);
            },
            else => {},
        }
    }
}

/// The global fn id of `core/mem`'s element primitive `prim`. Found by its qualified name
/// (bundled `core/mem` is always in the graph when a program indexes a `Vec`); `null` when
/// the name table is absent or the module was not pulled in.
fn elemPrimGid(model: *const Model, prim: StdNames.ElemPrim) ?u32 {
    const names = model.gph_fn_names orelse return null;
    for (names, 0..) |nm, i| {
        if (std.mem.eql(u8, nm, prim.name())) return @intCast(i);
    }
    return null;
}

/// Resolve the receiver type a `desugarMethods` entry selects its method on, from the
/// sugar node `n` (at index `i`) and the pass's `node_types`.
fn desugarRecvType(node_types: []const Type, n: Ast.Node, i: usize, src: StdNames.RecvSource) Type {
    return switch (src) {
        .node => node_types[i],
        .rhs => node_types[n.rhs.int()],
    };
}

/// Discover-and-enqueue the concrete instance of the template method `name` on the
/// `App` receiver `recv` (binding the impl's params by matching the template's `Self`
/// pattern against `recv`'s args — the same `Infer.match` Pass C runs). A miss on any
/// step (not an App, no such template, non-concrete binding) enqueues nothing.
fn discoverRecvMethod(t: *Typecheck, model: *const Model, recv: Type, name: []const u8, site: Site, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void), at: Anchor, mod: u32) !void {
    if (!recv.isApp()) return;
    const re = t.composite.at(recv.appIdx());
    const m = findGenericMethod(model.templates, re.ctor, re.ctor_is_enum, name, site) orelse return;
    const out = (try bindImplParams(t.gpa, t.composite, model.fns[m.fn_id], re.args)) orelse return;
    defer t.gpa.free(out);
    for (out) |ta| if (!isConcreteValue(ta)) return;
    try t.enqueueInstance(model, m.fn_id, out, worklist, seen, at, mod);
}

/// Where an instantiation diagnostic lands in the scanned module: a whole node, or just a
/// node's main token. Measured only when a diagnostic fires, since discovery visits
/// every call of every instance.
const Anchor = union(enum) { node: Ast.Index, token_of: Ast.Index };

fn anchorSpan(t: *const Typecheck, mod: u32, a: Anchor) Span {
    const mc = &t.graph.mods[mod];
    return switch (a) {
        .node => |n| Walk.nodeSpan(.{ .tree = mc.tree, .tokens = mc.tokens, .source = mc.source }, n),
        .token_of => |n| refs.tokSpan(mc, mc.tree.nodes[n.int()].main_token),
    };
}

/// `demandWitnesses` for each derive request from `next.*` on.
fn drainDemands(t: *Typecheck, model: *const Model, next: *usize, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void), visited: *std.AutoHashMapUnmanaged(u64, void)) !void {
    while (next.* < t.derive_reqs.items.len) : (next.* += 1) {
        const r = t.derive_reqs.items[next.*];
        const site = r.site orelse continue;
        try t.demandWitnesses(model, r.protocol_id, r.conform_ty, worklist, seen, visited, .{ .node = site.node }, site.scope);
    }
}

/// Instantiate the witnesses a derive request leans on through explicit generic
/// conformances: the request's type itself when an `impl Ctor[T] has P` covers it, or
/// any component of the type the structural derive will reach. A builtin witness site
/// (`to_string`, `==`) is no call, so `scanCalls` never discovers these instances.
/// `visited` dedups `(pid, type)` across requests.
fn demandWitnesses(t: *Typecheck, model: *const Model, pid: u32, ty: Type, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void), visited: *std.AutoHashMapUnmanaged(u64, void), at: Anchor, mod: u32) !void {
    switch (ty.kind) {
        .@"struct", .@"enum", .app => {},
        else => return,
    }
    const id = if (ty.isApp()) ty.appIdx() else ty.nominalId();
    const key = (@as(u64, pid) << 40) | (@as(u64, @intFromEnum(ty.kind)) << 32) | id;
    if ((try visited.getOrPut(t.gpa, key)).found_existing) return;
    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(t.gpa);
    if (ty.isApp()) {
        const e = t.composite.at(ty.appIdx());
        switch (try conform.generic(t.conformTables(model, &.{}), &memo, ty, pid, &.{})) {
            .none => {},
            .unmet => return,
            .met => |m| {
                defer m.deinit(t.gpa);
                const wpid = m.row.refines orelse pid;
                for (model.templates) |tm| {
                    if (tm.protocol_id != wpid or tm.recv_ctor != e.ctor or tm.recv_is_enum != e.ctor_is_enum) continue;
                    if (!eqlTypeVec(tm.protocol_args, m.row.protocol_args)) continue;
                    const mf = model.fns[tm.fn_id];
                    if (hasOwnTypeParams(t.composite, mf)) continue;
                    const targs = (try bindImplParams(t.gpa, t.composite, mf, e.args)) orelse continue;
                    defer t.gpa.free(targs);
                    try t.enqueueInstance(model, tm.fn_id, targs, worklist, seen, at, mod);
                }
                return;
            },
        }
        if (e.is_fn) return;
        if (e.ctor_is_enum) {
            if (e.ctor < model.enums.len) for (model.enums[e.ctor].variants) |v| for (v.field_types) |ft|
                try t.demandWitnesses(model, pid, try conform.substPattern(t.composite, t.gpa, ft, e.args), worklist, seen, visited, at, mod);
        } else if (e.ctor < model.structs.len) for (model.structs[e.ctor].field_types) |ft|
            try t.demandWitnesses(model, pid, try conform.substPattern(t.composite, t.gpa, ft, e.args), worklist, seen, visited, at, mod);
        return;
    }
    if (ty.kind == .@"enum") {
        if (ty.enum_id < model.enums.len) for (model.enums[ty.enum_id].variants) |v| for (v.field_types) |ft|
            try t.demandWitnesses(model, pid, ft, worklist, seen, visited, at, mod);
    } else if (ty.struct_id < model.structs.len) for (model.structs[ty.struct_id].field_types) |ft|
        try t.demandWitnesses(model, pid, ft, worklist, seen, visited, at, mod);
}

/// `P[A, B]`, each arg as a diagnostic spells it.
fn writeProtocolRef(t: *Typecheck, out: *std.ArrayList(u8), pname: []const u8, args: []const Type) !void {
    try out.appendSlice(t.gpa, pname);
    try out.append(t.gpa, '[');
    for (args, 0..) |a, i| {
        if (i != 0) try out.appendSlice(t.gpa, ", ");
        try out.print(t.gpa, "{f}", .{t.fmtType(a).nested()});
    }
    try out.append(t.gpa, ']');
}

/// Instantiate the explicit generic impls a Hashable key's erased hash/eq dispatch calls
/// through `pids`. The dispatch is no call, so neither `scanCalls` nor a derive request
/// discovers them, and without their instances the derive would stand in for them.
fn demandKeyWitnesses(t: *Typecheck, model: *const Model, pids: []const u32, key: Type, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void), at: Anchor, mod: u32) error{OutOfMemory}!void {
    var visited: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer visited.deinit(t.gpa);
    for (pids) |pid| try t.demandWitnesses(model, pid, key, worklist, seen, &visited, at, mod);
}

/// The conformance tables over `model`, as a mono-tail query (no bounded template in
/// scope unless `bound_protocols` says so) sees them.
fn conformTables(t: *Typecheck, model: *const Model, bound_protocols: []const ?u32) conform.Tables {
    return .{ .structs = model.structs, .enums = model.enums, .conformances = model.conformances, .gen = .of(model), .composite = t.composite, .gpa = t.gpa, .bound_protocols = bound_protocols };
}

/// Enqueue `(gid, args)` for instantiation, deduped on the canonical key. `args` is
/// BORROWED (the caller keeps its scratch); a fresh owned copy is stored on the
/// worklist. Shared by the explicit and inferred `scanCalls` branches so their dedup
/// path is byte-identical. `at`/`mod` anchor the T0017 termination diagnostic.
fn enqueueInstance(t: *Typecheck, model: *const Model, gid: u32, args: []const Type, worklist: *std.ArrayList(Pending), seen: *std.StringHashMapUnmanaged(void), at: Anchor, mod: u32) !void {
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
            try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0017 }, "instantiation too deep: generic type nesting exceeds the depth limit", .{});
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
        const is_hashable = if (model.prelude) |pre| pre.protocols.hashable == pid else false;
        const unmet: ?conform.UnmetBound = switch (try conform.generic(t.conformTables(model, &.{}), &cmemo, args[ord], pid, subst_buf)) {
            .none => null,
            .met => |m| {
                m.deinit(t.gpa);
                if (is_hashable) try t.demandKeyWitnesses(model, &.{pid}, args[ord], worklist, seen, at, mod);
                continue;
            },
            .unmet => |u| u,
        };
        // A `[K has Hashable]` bound is satisfied structurally by an aggregate whose fields are
        // all Hashable AND which holds no managed box (a `Ref` has no stable hashable value) —
        // the SAME `conform.hashable` authority the descriptor's erased key dispatch gates on,
        // so a key that passes here always has resolvable hash/eq witnesses. A Ref-tainted /
        // non-hashable key falls through to the clean T0023 below (never the mislocated
        // internal T0030 the descriptor synth would otherwise emit).
        if (unmet == null and subst_buf.len == 0 and is_hashable) if (model.prelude) |pre| if (pre.protocols.hash) |hp| if (pre.protocols.eq) |ep| {
            if (try conform.hashable(model.structs, model.enums, model.conformances, .of(model), t.composite, args[ord], pid, hp, pre.ref_struct, pre.gc_array_struct, &cmemo, t.gpa)) {
                try t.demandKeyWitnesses(model, &.{ ep, hp }, args[ord], worklist, seen, at, mod);
                continue;
            }
        };
        // The frozen table holds only EXPLICIT/prelude conformances. A struct/enum that
        // merely DERIVES Eq/Ord/Hash structurally (no `impl`) satisfies a `[T has Ord]`
        // bound too — the same relation the `<`/`==`/`.hash()` operator predicates
        // use. Fall back to the recursive `conforms` query, gated to a zero-arg DERIVABLE
        // protocol (structural conformance is undefined for a parameterized/custom one, so
        // a custom-protocol bound with no impl still correctly emits T0023). The per-
        // instance re-check records the ground derive request, so the witness is synthesized.
        if (unmet == null and subst_buf.len == 0 and isDerivableProtocol(model, pid) and
            try conform.structural(model.structs, model.enums, model.conformances, .of(model), args[ord], pid, &cmemo, t.gpa, t.composite, &.{}))
            continue;
        if (rejectedConformance(model.rejected_impls, t.composite, args[ord], pid)) return;
        // A poisoned arg was reported where it failed; its conformance is no new error.
        if (holdsPoison(t.composite, args[ord])) return;
        _ = t.gphSelect(mod);
        const tname = t.typeName(args[ord]);
        const shown = t.fmtType(args[ord]);
        const pname = model.protocols[pid].name;
        // A generic impl exists but an impl-param bound fails: suggesting that impl again
        // would be wrong, so name the type arg that fails it.
        if (unmet) |u| {
            try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; its generic impl needs type argument '{f}' to conform to '{s}'", .{ shown, pname, t.fmtType(u.arg), model.protocols[u.pid].name });
        } else if (if (subst_buf.len > 0) try conform.givenArgs(t.conformTables(model, &.{}), args[ord], pid) else null) |given| {
            // Its generic impl exists at other protocol args: name both rather than
            // suggest writing the impl again.
            defer t.gpa.free(given);
            var want: std.ArrayList(u8) = .empty;
            defer want.deinit(t.gpa);
            var got: std.ArrayList(u8) = .empty;
            defer got.deinit(t.gpa);
            try t.writeProtocolRef(&want, pname, subst_buf);
            try t.writeProtocolRef(&got, pname, given);
            try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; its generic impl gives '{s}'", .{ shown, want.items, got.items });
        } else if (args[ord].isFunc()) {
            // No impl can be written for a function type, so the usual fixes do not apply.
            try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "function type '{f}' does not conform to protocol '{s}'; a function value only compares by identity ('==') and renders as '<fn>'", .{ shown, pname });
        } else if (scalarImplModule(model, tname, pid)) |m| {
            try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; its impl is in std/{s}: " ++ StdNames.add_import, .{ shown, pname, m, m });
        } else if (try conform.displayBlockingBox(model.structs, model.enums, model.conformances, .of(model), t.composite, args[ord], pid, &cmemo, t.gpa)) |box| {
            if (refImportModule(model, t.composite, box)) |m| {
                if (Type.eql(box, args[ord])) {
                    try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; its impl is in std/{s}: " ++ StdNames.add_import, .{ shown, pname, m, m });
                } else {
                    try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; its field type '{f}' has its impl in std/{s}: " ++ StdNames.add_import, .{ shown, pname, t.fmtType(box), m, m });
                }
            } else try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; it is or holds a managed box (a 'Ref' or a collection), which never derives '{s}': add an `impl {s} has {s} {{ ... }}`", .{ shown, pname, pname, tname, pname });
        } else if (isDerivableProtocol(model, pid) and isAggregate(args[ord])) {
            try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; make every field of '{s}' conform to '{s}' (it then derives automatically), or add an `impl {s} has {s} {{ ... }}`", .{ shown, pname, tname, pname, tname, pname });
        } else {
            try t.sink.report(.{ .span = t.anchorSpan(mod, at), .code = .T0023 }, "type '{f}' does not conform to protocol '{s}'; add an `impl {s} has {s} {{ ... }}`", .{ shown, pname, tname, pname });
        }
        return;
    }

    const owned = try t.gpa.dupe(Type, args);
    errdefer t.gpa.free(owned);
    try worklist.append(t.gpa, .{ .gid = gid, .args = owned });
}

/// Re-check one generic instance: substitute the template's params/ret to concrete
/// types and walk the template body with a substitution-seeded `BodyChecker` so every
/// node types concretely. Returns an owning `Instance` (name filled after the canonical
/// sort). The body is checked into `sc`, its module's unchecked scratch tables, and the
/// instance keeps its own copy of `span`, the template's decl range; the caller resets
/// that range.
fn recheck(t: *Typecheck, model: *const Model, gid: u32, args: []const Type, span: NodeRange, sc: NodeSlots) !Mono.Instance {
    const f = model.fns[gid];

    const params = try t.gpa.alloc(Type, f.params.len);
    errdefer t.gpa.free(params);
    for (f.params, 0..) |p, i| params[i] = t.substType(p, args);
    const ret = t.substType(f.ret, args);

    const args_owned = try t.gpa.dupe(Type, args);
    errdefer t.gpa.free(args_owned);

    // The BodyChecker seeds slot_types from `f.params` and `cur_ret` from `f.ret`;
    // hand it the SUBSTITUTED sig so the body types against concrete param/ret. Its
    // `subst` map resolves any generic-param type-ref inside the body to concrete,
    // and `typeOfCall` writes concrete type-arg node_types for nested generic calls.
    var fsub = f;
    fsub.params = params;
    fsub.ret = ret;

    var bc = t.bodyCheckerFor(model, fsub, sc);
    defer bc.deinit();
    bc.subst = .{ .names = f.generic_params, .types = args };
    // Pass C checked a bounded template once against its bounds, onto the module's slots.
    for (f.generic_bounds) |b| if (b != null) {
        if (t.gph_callees) |cs| bc.axiom_callees = cs[f.mod];
        break;
    };
    try bc.checkBody(gid, fsub);
    // The instance keeps only `span`, so a write outside it would be lost.
    if (builtin.is_test) {
        for (sc.node_types[0..span.start]) |ty| std.debug.assert(ty.kind == .invalid);
        for (sc.node_types[span.end..]) |ty| std.debug.assert(ty.kind == .invalid);
        for (sc.callees[0..span.start]) |c| std.debug.assert(c.tag == .none);
        for (sc.callees[span.end..]) |c| std.debug.assert(c.tag == .none);
    }
    const inst_nt = try t.gpa.dupe(Type, sc.node_types[span.start..span.end]);
    errdefer t.gpa.free(inst_nt);
    const inst_callees = try t.gpa.dupe(Callee, sc.callees[span.start..span.end]);
    errdefer t.gpa.free(inst_callees);
    // Merge the instance's diagnostics into the shared stream (sorted at the end).
    try t.sink.merge(&bc.sink);
    try t.instance_hints.appendSlice(t.gpa, bc.template_spellings.items);
    bc.template_spellings.clearRetainingCapacity();
    // A derivable-struct `==` reached ONLY from an unbounded generic body is seen for the
    // first time here (Pass C skips unbounded templates), so drain the structural-Eq
    // requests recorded during this re-check; `synthesizeDerives` dedups+sorts afterward.
    try t.derive_reqs.appendSlice(t.gpa, bc.derive_reqs.items);
    // A `descriptor_of[T]` reached ONLY from an unbounded generic body is first seen here;
    // drain its requests so the synthesis barrier still plans the descriptor + erased units.
    try t.descriptor_reqs.appendSlice(t.gpa, bc.descriptor_reqs.items);
    // A char conversion reached ONLY from an unbounded generic body is first seen here;
    // OR-merge its captured Result so the synthesis barrier still reifies the witness.
    if (t.conv_int_char_result == null) t.conv_int_char_result = bc.conv_int_char_result;
    if (t.conv_char_byte_result == null) t.conv_char_byte_result = bc.conv_char_byte_result;
    if (t.conv_float_int_result == null) t.conv_float_int_result = bc.conv_float_int_result;

    // Build this instance's resolved bound conformances (all satisfied — enqueue
    // gated them, so a bound's `conform_ty` is a concrete `structT`/`enumT`/scalar the
    // later `reifyApps` never rewrites). The (e) fingerprint fold folds their structural
    // identity so a sibling-conformance edit invalidates exactly this instance.
    const confs = try t.buildConformances(model, f, args);
    errdefer freeInstanceConformances(t.gpa, confs);
    const boxed = try bc.boxed.toOwnedSlice(t.gpa);

    return .{
        .template_gid = gid,
        .args = args_owned,
        .node_types = inst_nt,
        .callees = inst_callees,
        .node_base = span.start,
        .params = params,
        .ret = ret,
        .name = null, // minted in canonical order after the sort
        .mod = f.mod,
        .decl_node = f.decl_node,
        .conformances = confs,
        .boxed = boxed,
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
    /// The types this fn's `descriptor_of[T]()` sites demanded. OWNED (moved out of the
    /// `BodyChecker`); merged into `t.descriptor_reqs` in fn-id order by `checkBodies`.
    descriptor_reqs: []Type = &.{},
    /// The concrete `Result[char/byte, ConvErr]` this fn's `try_into` sites typed, OR-merged
    /// into `t.conv_*_result` by `checkBodies` (gate for the shared-witness synthesis).
    conv_int_char_result: ?Type = null,
    conv_char_byte_result: ?Type = null,
    conv_float_int_result: ?Type = null,
    /// This fn's boxed locals, ascending. OWNED (moved out of the `BodyChecker`); moved
    /// into `t.fn_boxed[fid]` by `checkBodies`.
    boxed: []u32 = &.{},
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
    defer for (slots) |*s| gpa.free(s.descriptor_reqs);
    defer for (slots) |*s| gpa.free(s.boxed);

    if (t.jobs.io) |io| {
        const Ctx = struct {
            t: *const Typecheck,
            model: *const Model,
            slots: []BodyResult,
            pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(bodyUnit)) {
                return .{ c.t, c.model, @as(u32, @intCast(i)), &c.slots[i] };
            }
        };
        Engine.chunkedFanOut(io, n, t.jobs.ncpu, t.jobs.threshold, Engine.Chunk.body.chunks_per_cpu, bodyUnit, Ctx{ .t = t, .model = model, .slots = slots });
    } else {
        for (slots, 0..) |*s, i| bodyUnit(t, model, @intCast(i), s);
    }

    for (slots) |s| if (s.err) |e| return e;

    // Snapshot the per-fn poison flags in fn-id order for the mono tail, once the
    // units have joined (serial, deterministic). A poisoned BOUNDED template is not
    // instantiated (see `enqueueInstance`). Freed at teardown.
    t.bound_poisoned = try gpa.alloc(bool, n);
    for (slots, 0..) |s, i| t.bound_poisoned[i] = s.poisoned;
    t.fn_boxed = try gpa.alloc([]u32, n);
    for (slots, t.fn_boxed) |*s, *bx| {
        bx.* = s.boxed;
        s.boxed = &.{};
    }

    // Merge each fn's derive requests into the program-wide list in FN-ID ORDER (the
    // deterministic collection order the synthesis barrier's dedup+sort depends on —
    // mirrors the poison snapshot above). PODs, so a plain concat.
    for (slots) |s| try t.derive_reqs.appendSlice(gpa, s.derive_reqs);
    for (slots) |s| try t.descriptor_reqs.appendSlice(gpa, s.descriptor_reqs);

    // OR-merge the captured conv Result types: every site interned the SAME
    // `Result[char/byte, ConvErr]`, so first-writer-wins is order-free.
    for (slots) |s| {
        if (t.conv_int_char_result == null) t.conv_int_char_result = s.conv_int_char_result;
        if (t.conv_char_byte_result == null) t.conv_char_byte_result = s.conv_char_byte_result;
        if (t.conv_float_int_result == null) t.conv_float_int_result = s.conv_float_int_result;
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
    if (f.kind == .builtin or f.kind == .import) return; // bodyless (`print` / an `extern`)
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
    // A bounded template's `type_var`-typed nodes must NOT land in the shared module
    // node_types (reify/codegen would choke on an ungrounded `type_var`). Type it into a
    // SCRATCH buffer so the module's slots stay `.invalid` for template nodes exactly as
    // before; the scratch is discarded (a template is never a codegen unit — only its
    // reified instances are, and each instance re-check writes its OWN node_types).
    var scratch: []Type = &.{};
    var tvars: []Type = &.{};
    defer if (scratch.len > 0) t.gpa.free(scratch);
    defer if (tvars.len > 0) t.gpa.free(tvars);
    if (f.isGeneric()) {
        scratch = t.gpa.alloc(Type, t.graph.mods[f.mod].tree.nodes.len) catch {
            out.err = error.OutOfMemory;
            return;
        };
        @memset(scratch, .invalid);
        // Resolve the template's OWN generic-param type-refs (a body annotation /
        // turbofish spelling a param, e.g. `return Map[K, V] { .. }` or `mp_find[K, V]`)
        // to their ordinal `type_var`s — the SAME mapping `decodeFnSig` used for the
        // signature. Without this the bound-as-axiom check has no substitution, so a
        // bounded template that constructs/calls with its own params fails "unknown type
        // K"; an unbounded template is skipped above and never reaches here.
        tvars = t.gpa.alloc(Type, f.generic_params.len) catch {
            out.err = error.OutOfMemory;
            return;
        };
        for (tvars, 0..) |*tv, i| tv.* = Type.typeVar(@intCast(i));
    }
    var bc = t.bodyCheckerFor(model, f, .{
        .node_types = if (f.isGeneric()) scratch else t.gph_node_types.?[f.mod],
        // Kept on the module slice even for a bounded template: the LSP reads its picks.
        .callees = t.gph_callees.?[f.mod],
    });
    defer bc.deinit();
    if (f.isGeneric()) {
        bc.subst = .{ .names = f.generic_params, .types = tvars };
        bc.abstract_template = true;
    }
    bc.checkBody(fid, f) catch |e| {
        out.err = e;
        return;
    };
    // Poison a bounded template whose bound-as-axiom check reported an error, so the
    // mono tail skips instantiating it (a definition error fires once, not per instance).
    // A warning (a constant `if` condition) leaves the body sound, and its instances must
    // still be minted or lower finds no callee.
    out.poisoned = f.isGeneric() and bc.errorsSince(0);
    // Transfer the finished sink into the slot; leave bc holding a fresh empty sink
    // so the `defer bc.deinit()` frees nothing it no longer owns.
    out.sink.deinit();
    out.sink = bc.sink;
    bc.sink = DiagnosticSink.init(bc.gpa);
    // Move this fn's derive requests into the slot (merged fn-id-ordered by
    // checkBodies). `toOwnedSlice` empties bc's list so the deferred `bc.deinit` frees
    // nothing it no longer owns.
    out.derive_reqs = bc.derive_reqs.toOwnedSlice(bc.gpa) catch |e| {
        out.err = e;
        return;
    };
    out.descriptor_reqs = bc.descriptor_reqs.toOwnedSlice(bc.gpa) catch |e| {
        out.err = e;
        return;
    };
    out.boxed = bc.boxed.toOwnedSlice(bc.gpa) catch |e| {
        out.err = e;
        return;
    };
    out.conv_int_char_result = bc.conv_int_char_result;
    out.conv_char_byte_result = bc.conv_char_byte_result;
    out.conv_float_int_result = bc.conv_float_int_result;
}

/// A `pub` fn must not expose a non-`pub` type: if any param/return type resolves
/// to a struct/enum whose decl is not `pub`, an importer naming the fn could not
/// name the type. Diagnose against the owning module + the offending type-ref.
fn checkPubSignatures(t: *Typecheck, fns: []const GraphFnInput) !void {
    for (fns, 0..) |gf, i| {
        // An `extern` (`.import`) only names C-ABI scalars (never a non-pub type), and
        // has no proto to re-walk for pub-exposure — skip it like the bodyless builtin.
        if (gf.kind != .user_fn or !gf.is_pub) continue;
        _ = t.gphSelect(gf.module);
        const f = t.fns.items[i];
        const decl = t.tree.nodes[f.decl_node.int()];
        const proto = Ast.protoAt(t.tree, decl.lhs.int());
        for (proto.params, f.params) |param_idx, pty| {
            try t.checkPubType(pty, t.fieldTypeSpan(param_idx), "function", gf.name);
        }
        if (proto.ret_type != Ast.none)
            try t.checkPubType(f.ret, t.spanOf(proto.ret_type), "function", gf.name);
    }

    // A `pub` FIELD of a `pub` struct or a `pub` enum variant PAYLOAD that names a
    // non-pub type leaks it across the boundary exactly as a fn param/return would (an
    // importer can read the field / destructure the variant but cannot name the type):
    // the locked rule "a pub signature naming a type forces that type pub" applies to
    // FIELD types too. A private field is unreachable outside, so it leaks nothing.
    // Per-type checking makes this transitive: a pub type embedded in another pub type
    // is itself checked.
    for (t.structs.items) |s| {
        if (s.decl_node == Ast.none or !s.pub_export or s.poisoned) continue;
        _ = t.gphSelect(s.mod);
        const field_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[s.decl_node.int()].lhs.int());
        for (s.field_types, 0..) |fty, fi| {
            if (fi < field_nodes.len and !t.tree.isPub(field_nodes[fi])) continue;
            const at = if (fi < field_nodes.len) t.fieldTypeSpan(field_nodes[fi]) else t.tokSpan(t.tree.nodes[s.decl_node.int()].main_token);
            try t.checkPubType(fty, at, "struct", s.name);
        }
    }
    for (t.enums.items) |e| {
        if (e.decl_node == Ast.none or !e.pub_export or e.poisoned) continue;
        _ = t.gphSelect(e.mod);
        const variant_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[e.decl_node.int()].lhs.int());
        for (e.variants, 0..) |v, vi| {
            const vnode: ?Ast.Node = if (vi < variant_nodes.len) t.tree.nodes[variant_nodes[vi].int()] else null;
            const payload: []const Ast.Index = if (vnode) |vn| switch (vn.tag) {
                .enum_variant_tuple, .enum_variant_struct => Ast.rangeSlice(t.tree, vn.lhs.int()),
                else => &.{},
            } else &.{};
            const name_at = t.tokSpan(if (vnode) |vn| vn.main_token else t.tree.nodes[e.decl_node.int()].main_token);
            for (v.field_types, 0..) |fty, k| {
                try t.checkPubType(fty, if (k < payload.len) t.fieldTypeSpan(payload[k]) else name_at, "enum", e.name);
            }
        }
    }
}

/// W0011 at each method the program declares where a bundled module declares one of the
/// same name on the same type, or the compiler provides one as a builtin, and the mark that
/// keeps it out of every bundled module's calls (`Method.shadows`). The program's calls reach
/// its method; std's reach std's own or the builtin. Only modules in the graph put rows in
/// the tables, so a std method counts exactly when its module is part of the program,
/// whether the program imports it directly or not. Runs before the mono tail, which copies
/// a template's mark onto its instance rows.
fn checkStdShadows(t: *Typecheck) !void {
    const pre = t.prelude;
    const char_id = if (pre) |p| p.char_struct else null;
    for (t.templates.items) |*own| {
        if (own.bundled) continue;
        const ty_name = if (own.recv_is_enum) t.enums.items[own.recv_ctor].name else t.structs.items[own.recv_ctor].name;
        const std_fn: ?u32 = for (t.templates.items) |other| {
            if (!other.bundled or other.recv_ctor != own.recv_ctor or other.recv_is_enum != own.recv_is_enum) continue;
            if (std.mem.eql(u8, other.name, own.name)) break other.fn_id;
        } else null;
        const fam: LayoutEngine.NativeEnumFamily = if (own.recv_is_enum and pre != null) pre.?.optResultFamily(own.recv_ctor) else .none;
        if (std_fn == null and optionResultMethod(fam, own.name) == null) continue;
        own.shadows = true;
        try t.reportStdShadow(own.fn_id, ty_name, std_fn);
    }
    for (t.methods.items) |*own| {
        if (own.bundled) continue;
        const std_fn: ?u32 = for (t.methods.items) |other| {
            if (other.bundled and Type.eql(other.recv, own.recv) and std.mem.eql(u8, other.name, own.name)) break other.fn_id;
        } else null;
        if (std_fn == null and !isBuiltinTypeMethod(own.recv, own.name, char_id)) continue;
        own.shadows = true;
        try t.reportStdShadow(own.fn_id, t.typeName(own.recv), std_fn);
    }
}

/// `std_fn` is the bundled method shadowed, or null for a builtin.
fn reportStdShadow(t: *Typecheck, own_fn: u32, ty_name: []const u8, std_fn: ?u32) !void {
    const f = t.fns.items[own_fn];
    _ = t.gphSelect(f.mod);
    const name_tok = t.tree.nodes[f.decl_node.int()].main_token;
    const span = t.tokSpan(name_tok);
    if (std_fn) |sf| {
        try t.sink.report(
            .{ .span = span, .code = .W0011 },
            "method '{s}' on '{s}' shadows the one in {s}: the program's calls reach this one, std's own calls keep std's; rename it to keep both",
            .{ t.nameText(name_tok), ty_name, t.graph.mods[t.fns.items[sf].mod].path },
        );
    } else {
        try t.sink.report(
            .{ .span = span, .code = .W0011 },
            "method '{s}' on '{s}' shadows the built-in one: the program's calls reach this one, std's own calls keep the built-in; rename it to keep both",
            .{ t.nameText(name_tok), ty_name },
        );
    }
}

/// W0013 at each method whose receiver struct has a field of function type with the same
/// name. Both stay reachable, by two rules a reader cannot see at the use: `x.m(..)` selects
/// the method first, and `x.m` as a value reads the field first. A field of another type
/// cannot be called, so it is never taken for the method. An associated fn is left out:
/// `x.m(..)` on one is T0018, an error that already names it.
fn checkFnFieldMethods(t: *Typecheck) !void {
    for (t.methods.items) |m| {
        if (m.is_static or m.recv.kind != .@"struct") continue;
        try t.reportFnFieldMethod(m.fn_id, m.recv.struct_id);
    }
    for (t.templates.items) |m| {
        if (!m.has_self or m.recv_is_enum) continue;
        try t.reportFnFieldMethod(m.fn_id, m.recv_ctor);
    }
}

fn reportFnFieldMethod(t: *Typecheck, fn_id: u32, struct_id: u32) !void {
    const s = t.structs.items[struct_id];
    const f = t.fns.items[fn_id];
    _ = t.gphSelect(f.mod);
    const name_tok = t.tree.nodes[f.decl_node.int()].main_token;
    const name = t.nameText(name_tok);
    const fi = for (s.field_names, 0..) |fname, i| {
        if (std.mem.eql(u8, fname, name)) break i;
    } else return;
    if (fi >= s.field_types.len or !s.field_types[fi].isFunc()) return;
    // A related span indexes the diagnostic's own module, so a field declared in another
    // module goes unlabelled.
    const related: ?Span = if (s.decl_node != Ast.none and s.mod == f.mod) blk: {
        const field_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[s.decl_node.int()].lhs.int());
        break :blk if (fi < field_nodes.len) t.tokSpan(t.tree.nodes[field_nodes[fi].int()].main_token) else null;
    } else null;
    const recv: u8 = if (s.name.len > 0 and std.ascii.isAlphabetic(s.name[0])) std.ascii.toLower(s.name[0]) else 'x';
    try t.sink.report(
        .{ .span = t.tokSpan(name_tok), .code = .W0013, .related = related },
        "method '{s}' on '{s}' has the same name as its field '{s}' of fn type: '{c}.{s}(..)' calls the method and '{c}.{s}' reads the field; rename one",
        .{ name, s.name, name, recv, name, recv, name },
    );
}

/// W0015 at each associated fn of an enum named like one of its variants, and the softer
/// W0016 at each such `self` method. A variant keeps its name (`E.A(..)` builds it), so the
/// associated fn is unreachable, while the method is reached only through a value.
fn checkVariantNamedFns(t: *Typecheck) !void {
    for (t.methods.items) |m| {
        if (m.recv.kind != .@"enum") continue;
        try t.reportVariantNamedFn(m.fn_id, m.recv.enum_id, m.is_static);
    }
    for (t.templates.items) |m| {
        if (!m.recv_is_enum) continue;
        try t.reportVariantNamedFn(m.fn_id, m.recv_ctor, !m.has_self);
    }
}

fn reportVariantNamedFn(t: *Typecheck, fn_id: u32, enum_id: u32, is_static: bool) !void {
    const e = t.enums.items[enum_id];
    const f = t.fns.items[fn_id];
    _ = t.gphSelect(f.mod);
    const name_tok = t.tree.nodes[f.decl_node.int()].main_token;
    const name = t.nameText(name_tok);
    const vi = for (e.variants, 0..) |v, i| {
        if (std.mem.eql(u8, v.name, name)) break i;
    } else return;
    // A related span indexes the diagnostic's own module, so a variant declared in another
    // module goes unlabelled.
    const related: ?Span = if (e.decl_node != Ast.none and e.mod == f.mod) blk: {
        const variant_nodes = Ast.rangeSlice(t.tree, t.tree.nodes[e.decl_node.int()].lhs.int());
        break :blk if (vi < variant_nodes.len) t.tokSpan(t.tree.nodes[variant_nodes[vi].int()].main_token) else null;
    } else null;
    const at: DiagnosticSink.Report = .{ .span = t.tokSpan(name_tok), .code = if (is_static) .W0015 else .W0016, .related = related };
    // Another module's enum is spelled as the impl's module imports it, so the hint names
    // the variant as the reader can write it.
    var qual: []const u8 = "";
    if (e.mod != f.mod) {
        var it = t.graph.mods[f.mod].namespaces.iterator();
        while (it.next()) |ns| if (ns.value_ptr.* == e.mod) {
            qual = ns.key_ptr.*;
            break;
        };
    }
    const dot: []const u8 = if (qual.len > 0) "." else "";
    if (is_static) {
        try t.sink.report(at, "associated fn '{s}' of '{s}{s}{s}' can never be called: '{s}{s}{s}.{s}' always names the variant '{s}'; rename the function", .{ name, qual, dot, e.name, qual, dot, e.name, name, name });
    } else {
        try t.sink.report(at, "method '{s}' on '{s}{s}{s}' has the name of its variant '{s}': '.{s}(..)' on a value calls the method and '{s}{s}{s}.{s}' is the variant; rename one", .{ name, qual, dot, e.name, name, name, qual, dot, e.name, name });
    }
}

/// Emit a coherence error if `ty` is a struct/enum whose declaration is not pub.
/// `owner_kind`/`owner_name` describe the exposing decl ("function" `lib.make`,
/// "struct" `lib.Outer`, "enum" `lib.E`).
fn checkPubType(t: *Typecheck, ty: Type, at: Span, owner_kind: []const u8, owner_name: []const u8) !void {
    const non_pub = switch (ty.kind) {
        .@"struct" => !t.structs.items[ty.struct_id].pub_export,
        .@"enum" => !t.enums.items[ty.enum_id].pub_export,
        else => false,
    };
    if (non_pub)
        try t.sink.report(.{ .span = at, .code = .T0009 }, "pub {s} '{s}' exposes non-pub type '{f}'", .{ owner_kind, owner_name, t.fmtType(ty) });
}

/// The type-ref of a `name: T` param or field, or the node itself when it is a bare
/// type (a tuple field / tuple-variant payload).
fn fieldTypeSpan(t: *const Typecheck, node: Ast.Index) Span {
    const n = t.tree.nodes[node.int()];
    if (n.tag == .param) return if (n.lhs != Ast.none) t.spanOf(n.lhs) else t.tokSpan(n.main_token);
    return t.spanOf(node);
}

pub fn tokSpan(t: *const Typecheck, tok: u32) Span {
    return refs.tokSpan(t, tok);
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
        const main_decl = tree.nodes[f.decl_node.int()];
        const main_tok = main_decl.main_token;
        if (!std.mem.eql(u8, tokens[main_tok].text(source), "main")) continue;
        const proto = Ast.protoAt(tree, main_decl.lhs.int());
        if (f.ret.kind != .int and f.ret.kind != .unit and f.ret.kind != .invalid) {
            // Select the entry module so the sink stamps this diagnostic with the
            // entry module's scope (gphSelect -> sink.setScope).
            _ = t.gphSelect(entry_mod);
            const at: Span = if (proto.ret_type != Ast.none) t.spanOf(proto.ret_type) else t.tokSpan(main_tok);
            try t.sink.report(.{ .span = at, .code = .T0010 }, "main must return int or ()", .{});
        }
        // A program receives no arguments, so nothing could pass one. Codegen refuses it
        // too; reporting it here makes `toy check` agree with a build.
        if (proto.params.len > 0) {
            _ = t.gphSelect(entry_mod);
            const at: Span = .{ .start = t.spanOf(proto.params[0]).start, .end = t.spanOf(proto.params[proto.params.len - 1]).end };
            try t.sink.report(.{ .span = at, .code = .T0073 }, "main takes no parameters", .{});
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
            if (pr.hashable) |id| if (std.mem.eql(u8, name, "Hashable")) return id;
        }
        return null;
    }
    if (n.tag == .field_access) {
        const target = refs.moduleOf(t, n.lhs) orelse return null;
        const name = t.nameText(n.main_token);
        const id = t.graph.mods[target].protocol_ids.get(name) orelse return null;
        return if (refs.exportsType(t.graph, target, name, t.protocols.items[id])) id else null;
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
        const target = refs.moduleOf(t, n.lhs) orelse return null;
        const member = t.nameText(n.main_token);
        if (t.graph.mods[target].struct_ids.get(member)) |id| return if (refs.exportsType(t.graph, target, member, t.structs.items[id])) Type.structT(id) else null;
        if (t.graph.mods[target].enum_ids.get(member)) |id| return if (refs.exportsType(t.graph, target, member, t.enums.items[id])) Type.enumT(id) else null;
        return null;
    }
    return null;
}

/// Ground a protocol-signature type to a conformance's concrete types. A protocol's
/// decoded sig uses `type_var(0)` for the `self` slot + any `Self`-typed param/return,
/// and `type_var(k>=1)` for the protocol's generic params (`protocol Into[U]` -> `U ==
/// type_var(1)`, ordinal-offset so `Self` keeps 0). Grounds `tv(0)` to `recv` and
/// `tv(k>=1)` to `protocol_args[k-1]`, through composite `App`s and fn types too: a
/// return spelled `Option[Item]` decodes to `App(Option, [type_var(1)])`, and a stray
/// `type_var(1)` left inside would read as an unrelated generic param of whatever
/// template the caller is checking. Re-interned Apps are transient query-only indices
/// (never fingerprinted). `ctx` is any holder of `composite` + `gpa` (the checker or the
/// body checker). The `k-1 < len` guard is defensive against an arity-mismatched impl
/// (already rejected before this is trusted for a witness).
pub fn groundProtoType(ctx: anytype, ty: Type, recv: Type, protocol_args: []const Type) Type {
    if (ty.isTypeVar()) {
        const ord = ty.typeVarOrd();
        if (ord == 0) return recv;
        return if (ord - 1 < protocol_args.len) protocol_args[ord - 1] else ty;
    }
    return refs.mapComposite(ctx, ty, struct {
        ctx: @TypeOf(ctx),
        recv: Type,
        protocol_args: []const Type,
        pub fn map(m: @This(), x: Type) Type {
            return groundProtoType(m.ctx, x, m.recv, m.protocol_args);
        }
    }{ .ctx = ctx, .recv = recv, .protocol_args = protocol_args });
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
    // Every use of a repeated name resolves to its first param, so a later one could
    // never be bound. The impl's own pattern is checked once, below, against `Self`.
    const impl_n: usize = if (recv_type != Ast.none and t.tree.nodes[recv_type.int()].tag == .type_app)
        Ast.rangeSlice(t.tree, t.tree.nodes[recv_type.int()].rhs.int()).len
    else
        0;
    for (proto.generic_params, 0..) |gp, i| {
        if (i < impl_n) continue;
        for (gnames[0..i], 0..) |prev, j| if (std.mem.eql(u8, prev, gnames[i])) {
            const span = t.tokSpan(t.tree.nodes[gp.int()].main_token);
            if (j < impl_n)
                try t.sink.report(.{ .span = span, .code = .T0053 }, "type parameter '{s}' is already the impl's; give the method's own a different name", .{gnames[i]})
            else
                try t.sink.report(.{ .span = span, .code = .T0053 }, "type parameter '{s}' is declared twice", .{gnames[i]});
            break;
        };
    }
    // Put the fn's generic params in scope BEFORE decoding the bound args, so a bound arg
    // referencing another param resolves to a `type_var` rather than an unknown type.
    t.cur_generic_params = gnames;
    defer t.cur_generic_params = &.{};
    const bounds = try t.decodeBounds(proto.generic_params);
    gbounds = bounds.ids;
    gbound_args = bounds.args;

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
        params[i] = t.typeFromNode(param.lhs);
    }
    const ret = refs.retTypeFromNode(t, proto.ret_type);
    const mut_mask = Modes.ofProto(t.tree, t.tokens, proto);
    try t.fns.append(t.gpa, .{ .decl_node = fn_idx, .kind = .user_fn, .params = params, .ret = ret, .mod = mod, .generic_params = gnames, .generic_bounds = gbounds, .generic_bound_args = gbound_args, .self_type = self_ty, .mut_mask = mut_mask });
    // Register the method into the program-wide table (SERIAL, fn-id order). The name
    // is BORROWED from source (like `Sig.name`); the table is frozen before Pass C.
    if (recv_type != Ast.none) {
        const name = t.nameText(decl.main_token);
        // An impl method with no leading `self` param is an ASSOCIATED function
        // (`fn new() -> Vec[T]`): it carries the impl receiver as `self_type` (so `Self`
        // and the receiver's generics resolve) but takes no `self` value. Classify off the
        // syntactic receiver marker (the leading param the parser named `self`), not off a
        // params[0]-equals-receiver type test — an associated fn whose first parameter
        // happens to be the receiver type (`fn combine(other: Box[T])`) is still self-less.
        const has_self = proto.params.len > 0 and Ast.isSelfParam(t.tree, t.tokens, t.source, proto.params[0]);
        // A static method's `mut` first param is a place param, not a receiver.
        const mut_self = has_self and Modes.isMut(mut_mask, 0);
        const nominal = self_ty.kind == .@"struct" or self_ty.kind == .@"enum";
        if (gnames.len > 0 and !self_ty.isApp() and !nominal and self_ty.kind != .invalid) {
            // Templates are keyed by a struct or enum id, which a builtin receiver has none
            // of. Reported here, not at a call, so a method never called is caught too; its
            // row stays so a call finds it rather than adding a no-such-method error.
            try t.sink.report(.{ .span = t.tokSpan(decl.main_token), .code = .T0013 }, "'{s}' declares its own type parameters, which a method of the builtin type '{f}' cannot have yet; make it a generic fn", .{ name, t.fmtType(self_ty) });
        }
        if (!self_ty.isApp() and gnames.len > 0 and nominal) {
            // A method with type params of its own on a non-generic struct or enum: a
            // template like a generic type's, its params bound per call, with one instance
            // row per instantiation on the receiver itself.
            try t.templates.append(t.gpa, .{
                .recv_ctor = self_ty.nominalId(),
                .recv_is_enum = self_ty.kind == .@"enum",
                .name = name,
                .fn_id = gid,
                .mut_self = mut_self,
                .has_self = has_self,
                .bundled = t.graph.mods[mod].bundled,
            });
        } else if (self_ty.isApp()) {
            // A generic receiver `impl Box[T]`: `self_ty` is a check-time `App`.
            // Record its ctor into the TEMPLATE table so `findGenericMethod` dispatches
            // off a concrete receiver `App`'s ctor without dereferencing the
            // (post-typecheck-freed) composite. A pattern no template can serve (see
            // `implPatternFault`) is rejected rather than minted as an uninstantiable
            // template.
            const e = t.composite.at(self_ty.appIdx());
            const tm: TemplateMethod = .{
                .recv_ctor = e.ctor,
                .recv_is_enum = e.ctor_is_enum,
                .name = name,
                .fn_id = gid,
                .mut_self = mut_self,
                .has_self = has_self,
                .bundled = t.graph.mods[mod].bundled,
            };
            // `decodeImplHeads` already reported the pattern, once for the impl.
            if (implPatternFault(e.args) != null) {
                try t.rejected_methods.append(t.gpa, tm);
                return;
            }
            try t.templates.append(t.gpa, tm);
        } else {
            try t.methods.append(t.gpa, .{
                .recv = self_ty,
                .name = name,
                .fn_id = gid,
                .mut_self = mut_self,
                .is_static = !has_self,
                .bundled = t.graph.mods[mod].bundled,
            });
        }
    }
}

/// The bound protocol id (null = unbounded) and protocol args of each generic param,
/// owned by the caller. The caller puts the params in scope first, so a bound arg naming
/// another param decodes to its `type_var`.
const Bounds = struct {
    ids: []?u32 = &.{},
    args: [][]const Type = &.{},

    fn deinit(b: Bounds, gpa: std.mem.Allocator) void {
        if (b.ids.len > 0) gpa.free(b.ids);
        for (b.args) |ba| if (ba.len > 0) gpa.free(@constCast(ba));
        if (b.args.len > 0) gpa.free(b.args);
    }
};

fn decodeBounds(t: *Typecheck, gparams: []const Ast.Index) !Bounds {
    if (gparams.len == 0) return .{};
    const ids = try t.gpa.alloc(?u32, gparams.len);
    errdefer t.gpa.free(ids);
    const args = try t.gpa.alloc([]const Type, gparams.len);
    for (args) |*ba| ba.* = &.{};
    errdefer {
        for (args) |ba| if (ba.len > 0) t.gpa.free(@constCast(ba));
        t.gpa.free(args);
    }
    for (gparams, 0..) |gp, i| {
        const bound = Ast.genericParamBound(t.tree, gp);
        ids[i] = if (bound) |bn| (t.protocolIdFromNode(bn) orelse blk: {
            const ref_tok = t.tree.nodes[Ast.protocolRefBase(t.tree, bn).int()].main_token;
            try t.sink.report(.{ .span = t.tokSpan(ref_tok), .code = .T0021 }, "'{s}' is not a declared protocol", .{t.nameText(ref_tok)});
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
                    if (aty.isApp() or aty.isFunc()) {
                        try t.sink.report(.{ .span = t.spanOf(an), .code = .T0013 }, "a generic-protocol bound argument must be a concrete non-composite type (composite protocol args are not yet supported)", .{});
                        av[k] = .invalid;
                    } else av[k] = aty;
                }
                args[i] = av;
            }
        }
    }
    return .{ .ids = ids, .args = args };
}

/// Check each generic impl's receiver pattern once, from the impl's own header rather
/// than from its methods, which an impl may not have: T0013 rejects a pattern no template
/// can serve, and an accepted one gets its `ImplHead`.
fn decodeImplHeads(t: *Typecheck, mods: []const GraphModuleInput) !void {
    for (mods, 0..) |_, mi| {
        const mod: u32 = @intCast(mi);
        _ = t.gphSelect(mod);
        if (t.tree.nodes.len == 0) continue;
        const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
        if (prog.tag != .program) continue;
        for (Ast.rangeSlice(t.tree, prog.lhs.int())) |d| {
            const decl = t.tree.nodes[d.int()];
            if (decl.tag != .impl_decl and decl.tag != .impl_has_decl) continue;
            const recv_node = decl.lhs;
            const recv_ref = t.tree.nodes[recv_node.int()];
            if (recv_ref.tag != .type_app) {
                // Resolved here too so an impl with no methods still reports its receiver;
                // the sink drops the repeat a method's own decode makes.
                if (t.typeFromNode(recv_node).kind == .invalid) try t.noteRejectedImpl(mod, recv_node, t.implRecvNominal(recv_node));
                continue;
            }
            const gparams = Ast.rangeSlice(t.tree, recv_ref.rhs.int());
            const gnames = try t.gpa.alloc([]const u8, gparams.len);
            var kept = false;
            defer if (!kept) t.gpa.free(gnames);
            for (gparams, gnames) |gp, *gn| gn.* = t.nameText(t.tree.nodes[gp.int()].main_token);
            t.cur_generic_params = gnames;
            defer t.cur_generic_params = &.{};
            const recv = t.typeFromNode(recv_node);
            if (!recv.isApp()) {
                try t.noteRejectedImpl(mod, recv_node, t.implRecvNominal(recv_node));
                continue;
            }
            const e = t.composite.at(recv.appIdx());
            if (implPatternFault(e.args)) |fault| {
                try t.reportImplPattern(recv_node, e, fault);
                try t.noteRejectedImpl(mod, recv_node, if (e.ctor_is_enum) Type.enumT(e.ctor) else Type.structT(e.ctor));
                continue;
            }
            const bounds = try t.decodeBounds(gparams);
            errdefer bounds.deinit(t.gpa);
            try t.impl_heads.append(t.gpa, .{ .mod = mod, .recv_node = recv_node, .recv = recv, .bounds = bounds.ids, .bound_args = bounds.args, .gnames = gnames });
            kept = true;
        }
    }
}

/// The type a rejected impl's receiver names (`Box` for `impl Box` or `impl Box[T, U]`),
/// so the conformance it would have given is still known as rejected; `.invalid` when the
/// name is no local struct or enum.
fn implRecvNominal(t: *Typecheck, recv_node: Ast.Index) Type {
    var n = t.tree.nodes[recv_node.int()];
    if (n.tag == .type_app) n = t.tree.nodes[n.lhs.int()];
    if (n.tag != .identifier) return .invalid;
    const name = t.nameText(n.main_token);
    if (t.activeStructMap().get(name)) |id| return Type.structT(id);
    if (t.activeEnumMap().get(name)) |id| return Type.enumT(id);
    return .invalid;
}

/// The head of the generic impl at receiver node `recv_node` of module `mod`, or null
/// when it has none (not generic, or rejected).
pub fn implHead(t: *const Typecheck, mod: u32, recv_node: Ast.Index) ?ImplHead {
    for (t.impl_heads.items) |h| if (h.mod == mod and h.recv_node == recv_node) return h;
    return null;
}

/// Why an impl's receiver pattern cannot be a template: a fully ground receiver
/// (`impl Box[int]`) is a concrete-type impl, which is coherence territory; a partly
/// concrete or repeated arg (`impl Pair[T, int]`, `impl Pair[T, T]`) would leave an impl
/// param that no receiver binds, since the impl's params are the pattern's args, in order.
const ImplPatternFault = enum { concrete, partial };

fn implPatternFault(args: []const Type) ?ImplPatternFault {
    var any_var = false;
    for (args) |a| if (a.isTypeVar()) {
        any_var = true;
    };
    if (!any_var) return .concrete;
    for (args, 0..) |a, i| if (!a.isTypeVar() or a.typeVarOrd() != i) return .partial;
    return null;
}

/// T0013 at an impl's receiver pattern. It names the kind of impl and suggests the
/// pattern spelled with the type's own params (`impl Cell[T]`).
fn reportImplPattern(t: *Typecheck, recv_type: Ast.Index, e: anytype, fault: ImplPatternFault) !void {
    const kind = if (t.implProtocolOf(recv_type) != null) "a conformance" else "an inherent";
    const sym_name = if (e.ctor_is_enum) t.enums.items[e.ctor].name else t.structs.items[e.ctor].name;
    const sym_params = if (e.ctor_is_enum) t.enums.items[e.ctor].generic_params else t.structs.items[e.ctor].generic_params;
    const pattern: ImplPattern = .{ .name = sym_name, .params = sym_params };
    switch (fault) {
        .concrete => try t.sink.report(.{ .span = t.spanOf(recv_type), .code = .T0013 }, "{s} 'impl' on a concrete type instance is not yet supported (use 'impl {f}')", .{ kind, pattern }),
        .partial => try t.sink.report(.{ .span = t.spanOf(recv_type), .code = .T0013 }, "{s} 'impl' whose pattern is partly concrete or repeats a name is not yet supported; name each type parameter once, as in 'impl {f}'", .{ kind, pattern }),
    }
}

const ImplPattern = struct {
    name: []const u8,
    params: []const []const u8,

    pub fn format(p: ImplPattern, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.print("{s}[", .{p.name});
        for (p.params, 0..) |n, i| {
            if (i != 0) try w.writeAll(", ");
            try w.writeAll(n);
        }
        try w.writeByte(']');
    }
};

/// The protocol ref of the conformance (`has P`) whose receiver type-ref is `recv_type`,
/// or null for an inherent impl. The fn table records only the receiver node, so this
/// finds the impl that owns it.
fn implProtocolOf(t: *const Typecheck, recv_type: Ast.Index) ?Ast.Index {
    const prog = t.tree.nodes[Ast.root(t.tree.nodes).int()];
    if (prog.tag != .program) return null;
    for (Ast.rangeSlice(t.tree, prog.lhs.int())) |d| {
        const decl = t.tree.nodes[d.int()];
        if (decl.tag == .impl_has_decl and decl.lhs == recv_type) return Ast.implProtocol(t.tree, decl);
    }
    return null;
}

/// Record the impl at receiver node `recv_node` of module `mod` as rejected, once.
pub fn noteRejectedImpl(t: *Typecheck, mod: u32, recv_node: Ast.Index, recv: Type) !void {
    if (t.implRejected(mod, recv_node)) return;
    const pid = if (t.implProtocolOf(recv_node)) |pr| t.protocolIdFromNode(pr) else null;
    try t.rejected_impls.append(t.gpa, .{ .mod = mod, .recv_node = recv_node, .recv = recv, .protocol = pid });
}

/// Whether the impl at receiver node `recv_node` of module `mod` was rejected.
pub fn implRejected(t: *const Typecheck, mod: u32, recv_node: Ast.Index) bool {
    for (t.rejected_impls.items) |r| if (r.mod == mod and r.recv_node == recv_node) return true;
    return false;
}

/// Decode a bodyless `extern fn` signature into the global fn table (kind `.import`).
/// Every param type and the return type are gated to the C-ABI types the backend
/// can marshal — `int` (any width), `rawptr`, `bool`, `float` (f64/C `double`), or `str`
/// (a `{ptr,len}` reg-pair spreading to two C args); a `void` return, spelled by omitting
/// `-> R`, and a `never` return are also fine.
/// A type outside that set is T0037 and poisoned to `.invalid`. No generics, no self,
/// no method registration.
fn decodeExternSig(t: *Typecheck, fn_idx: Ast.Index, mod: u32) !void {
    const decl = t.tree.nodes[fn_idx.int()];
    const proto = Ast.protoAt(t.tree, decl.lhs.int());

    const params = try t.gpa.alloc(Type, proto.params.len);
    for (proto.params, 0..) |param_idx, i| {
        const param = t.tree.nodes[param_idx.int()];
        const pty = t.typeFromNode(param.lhs);
        if (externTypeOk(pty)) {
            params[i] = pty;
        } else {
            try t.sink.report(.{ .span = t.fieldTypeSpan(param_idx), .code = .T0037 }, "extern function parameter/return type must be int, float, str, rawptr, or bool", .{});
            params[i] = .invalid;
        }
    }
    var ret = refs.retTypeFromNode(t, proto.ret_type);
    // A `void` extern omits `-> R` (ret == unit); an EXPLICIT return must be a C-ABI
    // scalar, or `never` for a C fn that does not return (`exit`, `abort`).
    if (proto.ret_type != Ast.none and ret.kind != .never and !externTypeOk(ret)) {
        try t.sink.report(.{ .span = t.spanOf(proto.ret_type), .code = .T0037 }, "extern function parameter/return type must be int, float, str, rawptr, or bool", .{});
        ret = .invalid;
    }
    try t.fns.append(t.gpa, .{ .decl_node = fn_idx, .kind = .import, .params = params, .ret = ret, .mod = mod });
}

/// The C-ABI type set an `extern` param/return may use: any integer width, `bool`,
/// an opaque `rawptr`, a `float`, or a `str`. A `str` is a `{ptr,len}` reg-pair the
/// backend already marshals to two consecutive integer registers, so an extern `str`
/// param spreads to a C `(const char* buf, size_t len)` pair — the only way to hand a
/// toy string to a raw C call in plain toy source (no address-of-local exists to
/// decompose it into a `rawptr`). A `float` is an f64 riding the V-register (NSRN)
/// sequence, matching C `double`. Poison (`.invalid`) is accepted so a prior error
/// does not cascade into a second T0037.
fn externTypeOk(ty: Type) bool {
    return ty.isInteger() or ty.kind == .bool or ty.isRawPtr() or ty.kind == .str or ty.kind == .float or ty.kind == .invalid;
}

/// Append a placeholder `(str) -> ret` sig for a bodyless builtin (`panic`/the intrinsics).
/// The real arg-checking is token-based in `BodyChecker`; the sig holds the id slot and,
/// for `panic`, the `never` return its call is typed with.
fn appendBuiltinSig(t: *Typecheck, ret: Type) !void {
    const params = try t.gpa.dupe(Type, &.{.str});
    try t.fns.append(t.gpa, .{ .decl_node = Ast.none, .kind = .builtin, .params = params, .ret = ret });
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

pub fn typeName(t: *const Typecheck, ty: Type) []const u8 {
    return refs.typeName(t, ty);
}

pub fn fmtType(t: *const Typecheck, ty: Type) refs.Fmt(*const Typecheck) {
    return refs.fmt(t, ty, .{});
}

pub fn nameText(t: *const Typecheck, tok: u32) []const u8 {
    return refs.nameText(t, tok);
}

pub fn byteOf(t: *const Typecheck, tok: u32) u32 {
    return refs.byteOf(t, tok);
}

pub fn spanOf(t: *const Typecheck, node: Ast.Index) Span {
    return refs.spanOf(t, node);
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

test "conformance across a multi-node template cycle is expression-order-independent" {
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
        if (self.tree.pub_bits.len != 0) gpa.free(@constCast(self.tree.pub_bits));
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
    return checkSourceAs(source, .user);
}

/// `.core` checks the source as a bundled `core/` module, where `extern fn` is legal.
fn checkSourceAs(source: []const u8, as: enum { user, core }) !Checked {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, source);
    errdefer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var g = try Graph.single(gpa, if (as == .core) "core/t" else "main", "", source, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    g.modules[0].bundled = as == .core;
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    errdefer res.deinit(gpa);
    const result = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    return .{ .tokens = tokens, .tree = tree, .resolve = res, .result = result, .source = source };
}

/// Test support for span tests in the checker's files: typecheck each case's source,
/// find the diagnostic whose message contains `key`, and require its span to cover
/// exactly `want`.
pub const SpanCase = struct { src: []const u8, key: []const u8, want: []const u8 };

pub fn expectSpans(cases: []const SpanCase) !void {
    const gpa = testing.allocator;
    for (cases) |c| {
        var r = try checkSource(c.src);
        defer r.deinit(gpa);
        const d = for (r.result.diags) |d| {
            if (std.mem.indexOf(u8, d.message, c.key) != null) break d;
        } else return error.TestExpectedDiagnostic;
        const sp = d.span();
        try testing.expectEqualStrings(c.want, c.src[sp.start..sp.end]);
    }
}

/// Test support for the checker's files: typecheck `src` and require exactly `n`
/// diagnostics coded `code`.
pub fn expectCodeCount(src: []const u8, code: codes.Code, n: usize) !void {
    const gpa = testing.allocator;
    var r = try checkSource(src);
    defer r.deinit(gpa);
    var seen: usize = 0;
    for (r.result.diags) |d| {
        if (d.code == code) seen += 1;
    }
    testing.expectEqual(n, seen) catch |e| {
        for (r.result.diags) |d| std.debug.print("{s}\n", .{d.message});
        return e;
    };
}

/// Test support for the checker's files: typecheck `src` and require its diagnostics to
/// carry exactly `want`'s codes, in order (none for a clean check).
pub fn expectDiagCodes(src: []const u8, want: []const codes.Code) !void {
    const gpa = testing.allocator;
    var r = try checkSource(src);
    defer r.deinit(gpa);
    const same = r.result.diags.len == want.len and for (r.result.diags, want) |d, w| {
        if (d.code != w) break false;
    } else true;
    if (same) return;
    std.debug.print("source:\n{s}\n", .{src});
    for (r.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    return error.TestExpectedDiagnostic;
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

const MonoShape = enum { generic_fn, own_generic, instances, late_template };

/// A program of `n` fns or instances in one module: `n` fns calling a generic fn, `n` fns
/// calling an own-generic method at three types, or one fn making `n` distinct instances of
/// a generic fn declared first or last.
fn monoShapeSource(gpa: std.mem.Allocator, shape: MonoShape, n: usize) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    switch (shape) {
        .generic_fn => {
            try out.appendSlice(gpa, "fn id[T](x: T) -> T { x }\n");
            for (0..n) |k| try out.print(gpa, "fn _pad{d}(a: int) -> int {{ id(a) + id({d}) }}\n", .{ k, k });
            try out.appendSlice(gpa, "fn main() -> int { return 0 }\n");
        },
        .own_generic => {
            try out.appendSlice(gpa, "struct Box[T] { v: T }\nimpl Box[T] {\n    fn pick[U](self, a: U, _b: U) -> U { a }\n}\n");
            for (0..n) |k| try out.print(gpa, "fn _pad{d}(b: Box[int]) -> int {{ b.pick({d}, 1) + (if b.pick(true, false) {{ 1 }} else {{ 0 }}) + b.pick(\"s\", \"t\").len() }}\n", .{ k, k });
            try out.appendSlice(gpa, "fn main() -> int { return 0 }\n");
        },
        .instances, .late_template => {
            if (shape == .instances) try out.appendSlice(gpa, "fn id[T](x: T) -> T { x }\n");
            for (0..n) |k| try out.print(gpa, "struct S{d} {{ v: int }}\n", .{k});
            try out.appendSlice(gpa, "fn main() -> int {\n    t := 0\n");
            for (0..n) |k| try out.print(gpa, "    t = t + id(S{d} {{ v: {d} }}).v\n", .{ k, k });
            try out.appendSlice(gpa, "    return t\n}\n");
            if (shape == .late_template) try out.appendSlice(gpa, "fn id[T](x: T) -> T { x }\n");
        },
    }
    return out.toOwnedSlice(gpa);
}

test "the monomorphization scans and instance tables at most double when the program doubles" {
    // A scan or a table the size of the module per fn or per instance doubles both their
    // count and their size, so 2N costs 4x (#41). Counting nodes pins it where a timing flakes.
    const gpa = testing.allocator;
    for ([_]MonoShape{ .generic_fn, .own_generic, .instances, .late_template }) |shape| {
        var scanned: [2]usize = undefined;
        var slots: [2]usize = undefined;
        for ([_]usize{ 40, 80 }, &scanned, &slots) |n, *out, *slot_out| {
            const src = try monoShapeSource(gpa, shape, n);
            defer gpa.free(src);
            var c = try checkSource(src);
            defer c.deinit(gpa);
            try testing.expectEqual(@as(usize, 0), c.result.diags.len);
            if (shape == .instances or shape == .late_template) try testing.expectEqual(n, c.result.instances.len);
            out.* = c.result.mono_scanned_nodes;
            slot_out.* = 0;
            for (c.result.instances) |inst| slot_out.* += inst.node_types.len + inst.callees.len;
        }
        errdefer std.debug.print("{s}: {d}/{d} nodes scanned, {d}/{d} table slots at N=40/80\n", .{ @tagName(shape), scanned[0], scanned[1], slots[0], slots[1] });
        try testing.expect(scanned[1] <= 2 * scanned[0]);
        try testing.expect(slots[1] <= 2 * slots[0]);
    }
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

test "a reified struct instance keeps its template name as source_name; a plain struct has none" {
    const gpa = testing.allocator;
    var c = try checkSource("struct Pair[T] { a: T, b: T }\nstruct Wrap[T] { p: Pair[T] }\nstruct P { x: int }\nfn main() -> int {\n w := Wrap{ p: Pair{ a: 1, b: 2 } }\n q := P{ x: 3 }\n return w.p.a + q.x\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var pairs: usize = 0;
    var wraps: usize = 0;
    for (c.result.layouts) |l| {
        if (std.mem.startsWith(u8, l.name, "Pair$")) {
            pairs += 1;
            try testing.expectEqualStrings("Pair", l.source_name.?);
        } else if (std.mem.startsWith(u8, l.name, "Wrap$")) {
            wraps += 1;
            try testing.expectEqualStrings("Wrap", l.source_name.?);
        } else if (std.mem.eql(u8, l.name, "P")) {
            try testing.expect(l.source_name == null);
        }
    }
    try testing.expectEqual(@as(usize, 1), pairs);
    try testing.expectEqual(@as(usize, 1), wraps);
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

/// Whether ANY diagnostic's message contains `needle` (order-independent — the check
/// harness merges diags in fn-id order, but the near-miss suffix is what we assert on).
fn anyDiagContains(c: Checked, needle: []const u8) bool {
    for (c.result.diags) |d| {
        if (std.mem.indexOf(u8, d.message, needle) != null) return true;
    }
    return false;
}

test "did-you-mean: a close field typo suggests the declared field; the base message is preserved" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Account { balance: int, owner: int }
        \\fn main() -> int {
        \\ a := Account{ balance: 1, owner: 2 }
        \\ return a.balnce
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(anyDiagContains(c, "no field 'balnce' in struct 'Account'"));
    try testing.expect(anyDiagContains(c, "did you mean 'balance'?"));
}

test "did-you-mean: a distant field name yields no suggestion (base message unchanged)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Account { balance: int, owner: int }
        \\fn main() -> int {
        \\ a := Account{ balance: 1, owner: 2 }
        \\ return a.zzzzzz
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(anyDiagContains(c, "no field 'zzzzzz' in struct 'Account'"));
    try testing.expect(!anyDiagContains(c, "did you mean"));
}

test "did-you-mean: a close enum-variant typo suggests the declared variant" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum Color { Red, Green, Blue }
        \\fn pick() -> Color { return Color.Gren }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(anyDiagContains(c, "enum 'Color' has no variant 'Gren'"));
    try testing.expect(anyDiagContains(c, "did you mean 'Green'?"));
}

test "did-you-mean: a close method typo suggests a method callable on the receiver" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Counter { n: int }
        \\impl Counter { fn increment(self) -> int { return self.n } }
        \\fn main() -> int {
        \\ c := Counter{ n: 0 }
        \\ return c.incrementt()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(anyDiagContains(c, "no method 'incrementt' on type 'Counter'"));
    try testing.expect(anyDiagContains(c, "did you mean 'increment'?"));
}

test "did-you-mean: a close type-name typo suggests a visible type (bare + type-app sites)" {
    const gpa = testing.allocator;
    var bare = try checkSource(
        \\struct Widget { n: int }
        \\fn main() -> int {
        \\ w: Widgett = Widget{ n: 1 }
        \\ return w.n
        \\}
        \\
    );
    defer bare.deinit(gpa);
    try testing.expect(anyDiagContains(bare, "unknown type 'Widgett'"));
    try testing.expect(anyDiagContains(bare, "did you mean 'Widget'?"));

    var app = try checkSource(
        \\struct Wrapper[T] { v: T }
        \\fn main() -> int {
        \\ w := Wrappr[int]{ v: 1 }
        \\ return 0
        \\}
        \\
    );
    defer app.deinit(gpa);
    try testing.expect(anyDiagContains(app, "unknown type 'Wrappr'"));
    try testing.expect(anyDiagContains(app, "did you mean 'Wrapper'?"));
}

test "did-you-mean: an ambiguous typo (equidistant from two candidates) yields no suggestion" {
    const gpa = testing.allocator;
    // `bat` is edit-distance 1 from both `bar` and `bat`... use two same-distance fields.
    var c = try checkSource(
        \\struct S { bar: int, cat: int }
        \\fn main() -> int {
        \\ s := S{ bar: 1, cat: 2 }
        \\ return s.bat
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(anyDiagContains(c, "no field 'bat' in struct 'S'"));
    try testing.expect(!anyDiagContains(c, "did you mean"));
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

test "decodeFnSig records one mut_mask bit per mut param; a static method's mut first param is not mut_self" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\struct B[T] { v: T }
        \\impl P {
        \\ fn m(mut self, mut d: int) { self.x = d }
        \\}
        \\impl B[T] {
        \\ fn fill(mut s: T, v: T) { s = v }
        \\}
        \\fn bump(a: int, mut n: int) { n = a }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const want = [_]struct { name: []const u8, mask: u64 }{
        .{ .name = "m", .mask = 0b11 },
        .{ .name = "fill", .mask = 0b01 },
        .{ .name = "bump", .mask = 0b10 },
        .{ .name = "main", .mask = 0 },
    };
    for (want) |w| {
        const gid = for (c.resolve.fns, 0..) |f, i| {
            const last = if (std.mem.lastIndexOfScalar(u8, f.name, '.')) |dot| f.name[dot + 1 ..] else f.name;
            if (f.decl_node != Ast.none and std.mem.eql(u8, last, w.name)) break i;
        } else return error.TestExpectedFn;
        try testing.expectEqual(w.mask, c.result.sigs[gid].mut_mask);
        // The modes are a calling convention: every param type stays by-value.
        for (c.result.sigs[gid].params) |pt| try testing.expect(pt.kind != .rawptr);
    }
    try testing.expectEqual(@as(usize, 1), c.result.methods.len);
    const m = c.result.methods[0];
    try testing.expect(m.mut_self and !m.is_static);
    try testing.expectEqual(@as(u64, 0b11), c.result.sigs[m.fn_id].mut_mask);
    try testing.expectEqual(Kind.@"struct", c.result.sigs[m.fn_id].params[0].kind);
    try testing.expectEqual(Kind.int, c.result.sigs[m.fn_id].params[1].kind);
    try testing.expectEqual(@as(usize, 1), c.result.templates.len);
    const fill = c.result.templates[0];
    try testing.expect(!fill.has_self and !fill.mut_self);
    try testing.expectEqual(@as(u64, 0b01), c.result.sigs[fill.fn_id].mut_mask);
}

/// Exactly one diagnostic, with `code`.
fn expectOnlyDiag(src: []const u8, code: codes.Code) !void {
    const gpa = testing.allocator;
    var c = try checkSource(src);
    defer c.deinit(gpa);
    if (c.result.diags.len == 1 and c.result.diags[0].code == code) return;
    std.debug.print("source:\n{s}\n", .{src});
    for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    return error.TestExpectedDiagnostic;
}

const mut_pre =
    \\struct P { x: int }
    \\struct T2(int, int)
    \\struct Q { p: P }
    \\struct Box[T] { v: T }
    \\impl Box[T] {
    \\ fn fill(mut b: Box[T], v: T) { b.v = v }
    \\ fn put(self, mut out: T) { out = self.v }
    \\}
    \\impl P {
    \\ fn take(self, mut n: int) { n = self.x }
    \\}
    \\fn bump(mut n: int) { n = n + 1 }
    \\fn swap[T](mut a: T, mut b: T) {
    \\ t := a
    \\ a = b
    \\ b = t
    \\}
    \\fn reset(mut f: fn(int) -> int) { f = fn(v: int) -> int { v } }
    \\fn one() -> int { 1 }
    \\fn inc(v: int) -> int { v + 1 }
    \\
;

test "a mut argument must be a mutable place (T0019)" {
    const gpa = testing.allocator;
    const bodies = [_][]const u8{
        "fn main() -> int {\n bump(1)\n return 0\n}\n",
        "fn main() -> int {\n bump(one())\n return 0\n}\n",
        "fn main() -> int {\n bump(P { x: 1 }.x)\n return 0\n}\n",
        "fn main() -> int {\n reset(inc)\n return 0\n}\n",
        "fn main() -> int {\n fn lf(v: int) -> int { v }\n reset(lf)\n return 0\n}\n",
        // Every call path: turbofish, inferred (pretyped), a method arg, a static method.
        "fn main() -> int {\n a := 1\n swap[int](a, 1)\n return a\n}\n",
        "fn main() -> int {\n a := 1\n swap(a, 1)\n return a\n}\n",
        "fn main() -> int {\n p := P { x: 1 }\n p.take(one())\n return 0\n}\n",
        "fn main() -> int {\n bx := Box[int] { v: 1 }\n bx.put(2)\n return 0\n}\n",
        "fn main() -> int {\n Box[int].fill(Box[int] { v: 1 }, 2)\n return 0\n}\n",
        // A bound method is a fresh closure, not storage, inside a method too.
        "struct G { x: int }\nimpl G { fn add(self, v: int) -> int { self.x + v } }\nfn main() -> int {\n g := G { x: 1 }\n reset(g.add)\n return 0\n}\n",
        "struct G { x: int }\nimpl G {\n fn add(self, v: int) -> int { self.x + v }\n fn me(self) { reset(self.add) }\n}\nfn main() -> int { return 0 }\n",
    };
    for (bodies) |body| {
        const src = try std.mem.concat(gpa, u8, &.{ mut_pre, body });
        defer gpa.free(src);
        try expectOnlyDiag(src, .T0019);
    }
}

test "a callee's signature in a diagnostic spells its mut params" {
    const gpa = testing.allocator;
    const src = try std.mem.concat(gpa, u8, &.{ mut_pre, "fn main() -> int {\n c := 1\n bump(c, 2)\n return c\n}\n" });
    defer gpa.free(src);
    try expectOnlyDiag(src, .T0039);
    var c = try checkSource(src);
    defer c.deinit(gpa);
    try testing.expectEqualStrings("expected 1 argument(s), got 2; 'bump' takes (mut int)", c.result.diags[0].message);
}

test "locals, params, forwarded mut params, field chains, tuple fields, loop vars and match bindings are places" {
    const gpa = testing.allocator;
    var c = try checkSource(mut_pre ++
        \\fn fwd(mut n: int, k: int) {
        \\ bump(n)
        \\ bump(k)
        \\}
        \\fn main() -> int {
        \\ c := 1
        \\ bump(c)
        \\ p := P { x: 1 }
        \\ bump(p.x)
        \\ q := Q { p: P { x: 1 } }
        \\ bump(q.p.x)
        \\ t := T2(1, 2)
        \\ bump(t.1)
        \\ fwd(c, 2)
        \\ for i in 0..3 { bump(i) }
        \\ o: Option[int] = .some(1)
        \\ r := match o {
        \\  .some(v) -> {
        \\   bump(v)
        \\   v
        \\  },
        \\  .none -> 0,
        \\ }
        \\ a := 1
        \\ b := 2
        \\ swap(a, b)
        \\ swap(a, a)
        \\ bx := Box[int] { v: 0 }
        \\ Box[int].fill(bx, 3)
        \\ out := 0
        \\ bx.put(out)
        \\ p.take(c)
        \\ return c + p.x + q.p.x + t.1 + r + a + b + out
        \\}
        \\
    );
    defer c.deinit(gpa);
    for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

/// The `nth` (node order) call whose callee names `name` (`f(..)`, `r.f(..)`, `f[T](..)`).
fn nthCallOf(c: *const Checked, name: []const u8, nth: usize) !Ast.Index {
    var seen: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .call) continue;
        var head = c.tree.nodes[n.lhs.int()];
        if (head.tag == .type_app) head = c.tree.nodes[head.lhs.int()];
        if (head.tag != .identifier and head.tag != .field_access) continue;
        if (!std.mem.eql(u8, c.tokens[head.main_token].text(c.source), name)) continue;
        if (seen == nth) return Ast.Index.from(@intCast(i));
        seen += 1;
    }
    return error.TestCallNotFound;
}

test "Modes.argMask agrees with every call shape" {
    const gpa = testing.allocator;
    var c = try checkSource(mut_pre ++
        \\fn main() -> int {
        \\ c := 1
        \\ bump(c)
        \\ a := 1
        \\ b := 2
        \\ swap[int](a, b)
        \\ swap(a, b)
        \\ p := P { x: 1 }
        \\ p.take(c)
        \\ bx := Box[int] { v: 0 }
        \\ Box[int].fill(bx, 3)
        \\ bx.put(c)
        \\ f := fn(v: int) -> int { v }
        \\ s := "abc"
        \\ return f(c) + s.len() + inc(c)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const body: Body.BodyView = .{ .tree = c.tree, .resolutions = c.resolve.resolutions[0], .nodes = .{ .types = c.result.node_types[0], .callees = c.result.callees[0] } };
    const prog: Body.ProgramView = .{ .sigs = c.result.sigs, .methods = c.result.methods, .instances = c.result.instances };
    const cases = [_]struct { name: []const u8, nth: usize, mask: u64 }{
        .{ .name = "bump", .nth = 0, .mask = 0b1 }, // direct
        .{ .name = "swap", .nth = 0, .mask = 0b11 }, // turbofish generic
        .{ .name = "swap", .nth = 1, .mask = 0b11 }, // inferred generic
        .{ .name = "take", .nth = 0, .mask = 0b1 }, // instance method: self's bit is shifted out
        .{ .name = "fill", .nth = 0, .mask = 0b01 }, // static generic-impl method: not shifted
        .{ .name = "put", .nth = 0, .mask = 0b1 }, // generic-type instance method
        .{ .name = "f", .nth = 0, .mask = 0 }, // a value call
        .{ .name = "len", .nth = 0, .mask = 0 }, // a builtin method
        .{ .name = "inc", .nth = 0, .mask = 0 }, // a direct call with no mut param
    };
    for (cases) |k| {
        const call = try nthCallOf(&c, k.name, k.nth);
        testing.expectEqual(k.mask, Modes.argMask(body, prog, call)) catch |e| {
            std.debug.print("call {s}#{d}\n", .{ k.name, k.nth });
            return e;
        };
    }
}

test "for over an iter(mut self) receiver needs a place (T0019, #29)" {
    const head =
        \\protocol Iterator[Item] { fn next(mut self) -> Option[Item] }
        \\struct Up[T] { i: int }
        \\impl Up[T] has Iterator[int] { fn next(mut self) -> Option[int] { Option[int].none } }
        \\impl Up[T] { fn iter(mut self) -> Up[T] { self } }
        \\struct Down[T] { i: int }
        \\impl Down[T] has Iterator[int] { fn next(mut self) -> Option[int] { Option[int].none } }
        \\impl Down[T] { fn iter(self) -> Down[T] { self } }
        \\struct H { u: Up[int] }
        \\fn up() -> Up[int] { Up[int] { i: 1 } }
        \\fn down() -> Down[int] { Down[int] { i: 1 } }
        \\
    ;
    try expectCodeCount(head ++ "fn main() { for x in up() { _ = x } }\n", .T0019, 1);
    try expectCodeCount(head ++ "fn main() { u := up()\n for x in u { _ = x } }\n", .T0019, 0);
    try expectCodeCount(head ++ "fn main() { h := H { u: up() }\n for x in h.u { _ = x } }\n", .T0019, 0);
    try expectCodeCount(head ++ "fn main() { for x in down() { _ = x } }\n", .T0019, 0);
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

test "builtinStrMethod recognizes str concat/len/byte_at + int/bool to_string, rejects the rest" {
    const cc = builtinStrMethod(Type.str, "concat").?;
    try testing.expectEqual(@as(usize, 1), cc.params.len);
    try testing.expectEqual(Kind.str, cc.params[0].kind);
    try testing.expectEqual(Kind.str, cc.ret.kind);
    const ln = builtinStrMethod(Type.str, "len").?;
    try testing.expectEqual(@as(usize, 0), ln.params.len);
    try testing.expectEqual(Kind.int, ln.ret.kind);
    const ba = builtinStrMethod(Type.str, "byte_at").?;
    try testing.expectEqual(@as(usize, 1), ba.params.len);
    try testing.expectEqual(Kind.int, ba.params[0].kind);
    try testing.expectEqual(Kind.int, ba.ret.kind);
    try testing.expectEqual(Kind.str, builtinStrMethod(Type.int, "to_string").?.ret.kind);
    try testing.expectEqual(Kind.str, builtinStrMethod(Type.bool, "to_string").?.ret.kind);
    try testing.expectEqual(Kind.str, builtinStrMethod(Type.str, "to_string").?.ret.kind);
    // `()` gained `to_string` (it is `self.display()` -> the const-data `"()"`).
    try testing.expectEqual(Kind.str, builtinStrMethod(Type.unit, "to_string").?.ret.kind);
    try testing.expect(builtinStrMethod(Type.int, "concat") == null);
    // struct/enum `to_string` is gated in the checker (may derive Display), not here.
    try testing.expect(builtinStrMethod(Type.structT(0), "to_string") == null);
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

test "Ref[int] and Ref[bool] reify to DISTINCT structs, each tagged native_family .ref" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\pub fn take(_a: Ref[int], _b: Ref[bool]) -> int { 0 }
        \\fn main() -> int { 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var ref_int: ?usize = null;
    var ref_bool: ?usize = null;
    for (c.result.layouts, 0..) |l, i| {
        if (std.mem.eql(u8, l.name, "Ref$int")) ref_int = i;
        if (std.mem.eql(u8, l.name, "Ref$bool")) ref_bool = i;
    }
    try testing.expect(ref_int != null);
    try testing.expect(ref_bool != null);
    try testing.expect(ref_int.? != ref_bool.?); // distinct reified struct ids
    try testing.expectEqual(LayoutEngine.NativeStructFamily.ref, c.result.layouts[ref_int.?].native_family);
    try testing.expectEqual(LayoutEngine.NativeStructFamily.ref, c.result.layouts[ref_bool.?].native_family);
    // Each box is an 8-byte cell pointer.
    try testing.expectEqual(@as(u32, 8), c.result.layouts[ref_int.?].size);
}

test "an associated fn's T0016 hint names its own type params too, and compiles" {
    const Case = struct { decl: []const u8, call: []const u8, message: []const u8, fixed: []const u8 };
    const cases = [_]Case{
        .{ .decl = "fn mk[V](_v: T) -> int { 0 }", .call = "Box.mk(1)", .message = "cannot infer type parameter 'V' for 'Box.mk'; add explicit type arguments, e.g. Box[int].mk[int]", .fixed = "Box[int].mk[int](1)" },
        .{ .decl = "fn mk[U, V](v: T, _u: U) -> Box[T] { Box { v: v } }", .call = "Box.mk(1, \"s\").v", .message = "cannot infer type parameter 'V' for 'Box.mk'; add explicit type arguments, e.g. Box[int].mk[str, int]", .fixed = "Box[int].mk[str, int](1, \"s\").v" },
    };
    for (cases) |k| {
        const pre = try std.fmt.allocPrint(testing.allocator, "struct Box[T] {{ v: T }}\nimpl Box[T] {{ {s} }}\nfn main() -> int {{ return ", .{k.decl});
        defer testing.allocator.free(pre);
        const src = try std.mem.concat(testing.allocator, u8, &.{ pre, k.call, " }\n" });
        defer testing.allocator.free(src);
        try expectOneDiag(src, .T0016, k.message);
        const fixed = try std.mem.concat(testing.allocator, u8, &.{ pre, k.fixed, " }\n" });
        defer testing.allocator.free(fixed);
        try expectNoErrors(fixed);
    }
}

test "an associated fn of a bare generic struct infers its type params from its value args" {
    const gpa = testing.allocator;
    const decls =
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\ fn new(v: T) -> Box[T] { Box { v: v } }
        \\ fn empty() -> Box[T] { panic("none") }
        \\ fn pair(a: T, _b: T) -> Box[T] { Box { v: a } }
        \\}
        \\
    ;
    var c = try checkSource(decls ++
        \\fn main() -> int {
        \\ b := Box.new(5)
        \\ x: Box[int] = b
        \\ f := Box.new(true)
        \\ if f.v { return x.v }
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const cases = [_]struct { stmt: []const u8, code: codes.Code, message: []const u8 }{
        // The inferred `T` is the arg's own type, so a mismatched annotation is still caught.
        .{ .stmt = "_y: Box[bool] = Box.new(5)", .code = .T0041, .message = "cannot bind Box[int] to '_y' of type Box[bool]" },
        // No value arg binds `T`: the turbofish is still required.
        .{ .stmt = "_e := Box.empty()", .code = .T0016, .message = "cannot infer type parameter 'T' for 'Box.empty'; add explicit type arguments, e.g. Box[int].empty" },
        .{ .stmt = "_p := Box.pair(1, true)", .code = .T0015, .message = "conflicting types for type parameter 'T': int vs bool" },
        // The count is checked first, as for a generic fn: a turbofish would not help.
        .{ .stmt = "_q := Box.new(1, 2)", .code = .T0039, .message = "expected 1 argument(s), got 2; 'new' takes (T)" },
        .{ .stmt = "_r := Box.new()", .code = .T0039, .message = "expected 1 argument(s), got 0; 'new' takes (T)" },
        .{ .stmt = "_s := Box[int].nope(5)", .code = .T0018, .message = "'Box' has no associated function 'nope'" },
    };
    for (cases) |case| {
        const src = try std.mem.concat(gpa, u8, &.{ decls, "fn main() -> int {\n ", case.stmt, "\n return 0\n}\n" });
        defer gpa.free(src);
        var d = try checkSource(src);
        defer d.deinit(gpa);
        errdefer for (d.result.diags) |dg| std.debug.print("got {s}: {s}\n", .{ @tagName(dg.code), dg.message });
        try testing.expectEqual(@as(usize, 1), d.result.diags.len);
        try testing.expectEqual(case.code, d.result.diags[0].code);
        try testing.expectEqualStrings(case.message, d.result.diags[0].message);
    }
}

test "a field of a call result is no store target, and a user Vec has no index" {
    const gpa = testing.allocator;
    const decls =
        \\struct P { x: int }
        \\struct H { xs: Vec[int] }
        \\struct Vec[T] { n: int }
        \\fn id[T](x: T) -> T { x }
        \\
    ;
    const cases = [_]struct { stmt: []const u8, code: codes.Code, message: []const u8 }{
        // The `Vec` here is the user's, which has no element primitives to lower to.
        .{ .stmt = "ps[0].x = 5", .code = .T0057, .message = "cannot index a value of type 'Vec[P]'" },
        .{ .stmt = "xs[1] = 7", .code = .T0057, .message = "cannot index a value of type 'Vec[int]'" },
        .{ .stmt = "h.xs[0] = 7", .code = .T0057, .message = "cannot index a value of type 'Vec[int]'" },
        .{ .stmt = "i = xs[0]", .code = .T0057, .message = "cannot index a value of type 'Vec[int]'" },
        // Any other temporary is no place either.
        .{ .stmt = "id[H](h).xs = xs", .code = .T0075, .message = "cannot assign to a field of a temporary value, such as a call result; it is a copy, so the write would be lost" },
    };
    for (cases) |case| {
        const src = try std.mem.concat(gpa, u8, &.{ decls, "fn g(ps: Vec[P], xs: Vec[int], h: H) -> int {\n i := 0\n ", case.stmt, "\n return i + h.xs.n + ps.n + xs.n\n}\nfn main() -> int { 0 }\n" });
        defer gpa.free(src);
        var d = try checkSource(src);
        defer d.deinit(gpa);
        errdefer for (d.result.diags) |dg| std.debug.print("got {s}: {s}\n", .{ @tagName(dg.code), dg.message });
        const got = for (d.result.diags) |dg| {
            if (dg.code == case.code) break dg;
        } else return error.TestExpectedDiagnostic;
        try testing.expectEqualStrings(case.message, got.message);
    }
}

test "an associated fn of a non-generic struct or enum is called on its type as a static method" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct K { n: int }
        \\impl K {
        \\ fn make(n: int) -> K { K { n: n } }
        \\ fn get(self) -> int { self.n }
        \\}
        \\enum E { A, B }
        \\impl E { fn first() -> E { E.B } }
        \\enum F { Mk(int) }
        \\impl F { fn Mk(x: int) -> F { F.Mk(x + 1) } }
        \\fn main() -> int {
        \\ k := K.make(1)
        \\ _e := E.first()
        \\ _f := F.Mk(2)
        \\ return k.get()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The variant keeps its name, which W0015 points out at `fn Mk`.
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.W0015, c.result.diags[0].code);
    const make = try calleeMethod(&c, "make", 0);
    try testing.expect(make.is_static and make.instance == null and make.recv.kind == .@"struct");
    const first = try calleeMethod(&c, "first", 0);
    try testing.expect(first.is_static and first.recv.kind == .@"enum");
    // A variant keeps its name: `F.Mk(2)` constructs, it does not call the fn.
    try testing.expect((try calleeView(&c, "Mk", 1)) == .none);
}

test "an associated fn called on a value, or a method on a type, is T0018 naming how to reach each" {
    const gpa = testing.allocator;
    const decls =
        \\struct H { run: fn(int) -> int, n: int }
        \\impl H {
        \\ fn run(x: int) -> int { x + 1 }
        \\ fn make(n: int) -> H { H { run: fn(x) { x }, n: n } }
        \\ fn get(self) -> int { self.n }
        \\}
        \\enum E { A }
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn render(self) -> int { 1 } }
        \\
    ;
    const cases = [_]struct { stmt: []const u8, code: codes.Code, message: []const u8 }{
        // The field is not called instead; the message says how to reach each.
        .{ .stmt = "_a := h.run(2)", .code = .T0018, .message = "'run' is both an associated function of 'H' and a field of fn type: 'H.run(..)' calls the function and 'h.run' reads the field" },
        .{ .stmt = "_b := h.make(3)", .code = .T0018, .message = "'make' is an associated function; call it as 'H.make(..)'" },
        .{ .stmt = "_c := H.get()", .code = .T0018, .message = "'get' is an instance method, not an associated function; call it on a value" },
        .{ .stmt = "_d := H.nope(1)", .code = .T0018, .message = "'H' has no associated function 'nope'" },
        .{ .stmt = "_e := H.make(1, 2)", .code = .T0039, .message = "expected 1 argument(s), got 2; 'make' takes (int)" },
        .{ .stmt = "_f := E.nope()", .code = .T0061, .message = "enum 'E' has no variant 'nope'" },
        .{ .stmt = "_g := Box.render(Box { v: 7 })", .code = .T0018, .message = "'render' is an instance method, not an associated function; call it on a value" },
        .{ .stmt = "_h := Box[int].render(Box { v: 7 })", .code = .T0018, .message = "'render' is an instance method, not an associated function; call it on a value" },
    };
    for (cases) |case| {
        const src = try std.mem.concat(gpa, u8, &.{ decls, "fn main() -> int {\n h := H.make(1)\n ", case.stmt, "\n return h.n\n}\n" });
        defer gpa.free(src);
        var d = try checkSource(src);
        defer d.deinit(gpa);
        errdefer for (d.result.diags) |dg| std.debug.print("got {s}: {s}\n", .{ @tagName(dg.code), dg.message });
        try testing.expectEqual(@as(usize, 1), d.result.diags.len);
        try testing.expectEqual(case.code, d.result.diags[0].code);
        try testing.expectEqualStrings(case.message, d.result.diags[0].message);
    }
}

test "an associated fn `Vec[int].new()` mints one instance + a static reified-dispatch Method; an instance method's recv stays its self param" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Vec[T] { n: int }
        \\impl Vec[T] {
        \\ fn new() -> Vec[T] { Vec[T] { n: 0 } }
        \\ fn bump(self) -> int { self.n }
        \\}
        \\fn main() -> int {
        \\ a := Vec[int].new()
        \\ return a.bump()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);

    // The reified `Vec[int]` struct (the `new` receiver + the `bump` self type).
    var vec_int: ?u32 = null;
    for (c.result.layouts, 0..) |l, i| {
        if (std.mem.eql(u8, l.name, "Vec$int")) vec_int = @intCast(i);
    }
    try testing.expect(vec_int != null);
    const vec_int_ty = Type.structT(vec_int.?);

    // Exactly one reified-dispatch entry for `new`: static, receiver = reified `Vec[int]`.
    var new_entries: usize = 0;
    var saw_static_new = false;
    var saw_instance_bump = false;
    for (c.result.methods) |m| {
        if (m.instance == null) continue;
        if (std.mem.eql(u8, m.name, StdNames.method_new)) {
            new_entries += 1;
            try testing.expect(m.is_static);
            try testing.expect(Type.eql(m.recv, vec_int_ty));
            // The associated `new` instance has NO params — its receiver is reified from
            // the impl's self_type, not read off a (non-existent) self param.
            try testing.expectEqual(@as(usize, 0), c.result.instances[m.instance.?].params.len);
            saw_static_new = true;
        }
        if (std.mem.eql(u8, m.name, "bump")) {
            // An instance method keeps `is_static == false` and its recv is the self param.
            try testing.expect(!m.is_static);
            try testing.expect(Type.eql(m.recv, c.result.instances[m.instance.?].params[0]));
            try testing.expect(Type.eql(m.recv, vec_int_ty));
            saw_instance_bump = true;
        }
    }
    try testing.expectEqual(@as(usize, 1), new_entries);
    try testing.expect(saw_static_new);
    try testing.expect(saw_instance_bump);
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

test "Option native methods type to bool (predicates) / T (expect) with zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := Option[int].some(7)
        \\ p := x.is_some()
        \\ q := x.is_none()
        \\ u := x.expect("no value")
        \\ d := x.or(0)
        \\ if p { if q { return u } }
        \\ return d
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_some").?);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_none").?);
    try testing.expectEqual(Kind.int, methodCallKind(c, "expect").?);
    try testing.expectEqual(Kind.int, methodCallKind(c, "or").?);
}

test "Result native methods type to bool (predicates) / T (expect) with zero diags" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ x := Result[int, str].ok(7)
        \\ p := x.is_ok()
        \\ q := x.is_error()
        \\ u := x.expect("no value")
        \\ d := x.or(0)
        \\ if p { if q { return u } }
        \\ return d
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_ok").?);
    try testing.expectEqual(Kind.bool, methodCallKind(c, "is_error").?);
    // `expect`/`or` yield the FIRST type-arg (the `ok` payload), int here.
    try testing.expectEqual(Kind.int, methodCallKind(c, "expect").?);
    try testing.expectEqual(Kind.int, methodCallKind(c, "or").?);
}

test "expect takes exactly one str message" {
    for ([_][]const u8{ "x.expect()", "x.expect(1)", "x.expect(\"a\", \"b\")" }) |call| {
        var buf: [128]u8 = undefined;
        const src = try std.fmt.bufPrint(&buf, "fn main() -> int {{\n x := Option[int].some(7)\n return {s}\n}}\n", .{call});
        try testing.expectEqual(@as(usize, 1), try checkDiagCount(src));
    }
}

test "unwrap, unwrap_or, is_err and map_err are not methods of Option or Result" {
    const cases = [_][]const u8{
        "fn main() -> int {\n x := Option[int].some(7)\n return x.unwrap()\n}\n",
        "fn main() -> int {\n x := Option[int].some(7)\n return x.unwrap_or(0)\n}\n",
        "fn main() -> int {\n x := Result[int, str].ok(7)\n return if x.is_err() { 1 } else { 0 }\n}\n",
        "fn main() -> int {\n x := Result[int, str].ok(7)\n _y := x.map_err(fn(e: str) { e.len() })\n return 0\n}\n",
    };
    const gpa = testing.allocator;
    for (cases, [_][]const u8{ "unwrap", "unwrap_or", "is_err", "map_err" }) |src, name| {
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        const d = c.result.diags[0];
        try testing.expectEqual(codes.Code.T0018, d.code);
        var buf: [64]u8 = undefined;
        try testing.expect(std.mem.startsWith(u8, d.message, try std.fmt.bufPrint(&buf, "no method '{s}' on type", .{name})));
        try testing.expect(std.mem.indexOf(u8, d.message, "std/") == null);
    }
}

test "or with a wrong-typed default is exactly one diagnostic" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn main() -> int {
        \\ x := Option[int].some(7)
        \\ return x.or("s")
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

test "a std/option method without the import is T0018 naming std/option" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum Shade { light, dark }
        \\fn main() -> int {
        \\ o := Option[int].some(7)
        \\ r: Result[int, str] = Result.err("bad")
        \\ _a := o.map(fn(v: int) { v + 1 })
        \\ _b := r.map_error(fn(e: str) { e.len() })
        \\ _c := Shade.dark.map(fn(v: int) { v })
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 3), c.result.diags.len);
    const hint = "add 'import std/option'";
    for (c.result.diags) |d| try testing.expectEqual(codes.Code.T0018, d.code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "'map' on type 'Option[int]'") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, hint) != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[1].message, "'map_error' on type 'Result[int, str]'") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[1].message, hint) != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[2].message, hint) == null);
}

test "std/str methods do not shadow the builtin str methods" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\impl str {
        \\    fn contains(self, sub: str) -> bool { self.len() >= sub.len() }
        \\    fn split(self, _sep: str) -> int { self.len() }
        \\    fn chars(self) -> int { self.byte_at(0) }
        \\}
        \\fn main() -> int {
        \\    s := "a,b"
        \\    n := s.len()
        \\    if s.contains(",") { return s.split(",") + s.chars() + n }
        \\    return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Three `len` calls in the bodies, then main's `s.len()`.
    for (0..4) |nth| try testing.expectEqual(Callee.View{ .builtin = .str_len }, try calleeView(&c, "len", nth));
    try testing.expectEqual(Callee.View{ .builtin = .str_byte_at }, try calleeView(&c, "byte_at", 0));
    inline for (.{ "contains", "split", "chars" }) |name|
        try testing.expectEqualStrings(name, (try calleeMethod(&c, name, 0)).name);
    try expectNoPending(&c.result);
}

test "expect on a str (aggregate) payload is accepted — no diagnostic" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn main() -> int {
        \\ x := Option[str].some("hi")
        \\ s := x.expect("no value")
        \\ return 0
        \\}
        \\
    ));
}

test "or on a str (aggregate) payload is accepted — no diagnostic" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn main() -> int {
        \\ x := Option[str].some("hi")
        \\ s := x.or("d")
        \\ return 0
        \\}
        \\
    ));
}

test "expect on a struct (aggregate) payload is accepted — no diagnostic" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ x := Option[P].some(P { x: 40 })
        \\ p := x.expect("no value")
        \\ return 0
        \\}
        \\
    ));
}

test "or on a struct (aggregate) payload is accepted — no diagnostic" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct P { x: int }
        \\fn main() -> int {
        \\ d := P { x: 2 }
        \\ p := Option[P].none.or(d)
        \\ return 0
        \\}
        \\
    ));
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

test "a protocol method's fn-typed param grounds Self for the coherence signature check" {
    const gpa = testing.allocator;
    var ok = try checkSource(
        \\protocol Apply { fn app(self, f: fn(Self) -> int) -> int }
        \\pub struct S { x: int }
        \\impl S has Apply { fn app(self, _f: fn(S) -> int) -> int { self.x } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
    var bad = try checkSource(
        \\protocol Apply { fn app(self, f: fn(Self) -> int) -> int }
        \\pub struct S { x: int }
        \\impl S has Apply { fn app(self, _f: fn(int) -> int) -> int { self.x } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer bad.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), bad.result.diags.len);
    try testing.expectEqual(codes.Code.T0024, bad.result.diags[0].code);
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
    // recipes, both `.hash` kind. No container/descriptor is in play, so no trace unit is
    // synthesized (the vestigial natural-derive Trace fixpoint ignores str-only aggregates).
    // (Name = struct id 0, Person = struct id 1.)
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
        .{ .protocol = hash_pid, .recv = Type.bool },
        .{ .protocol = hash_pid, .recv = Type.unit },
    };
    var memo: std.AutoHashMapUnmanaged(u64, bool) = .empty;
    defer memo.deinit(gpa);
    var co: Composite = .{};
    defer co.deinit(gpa);
    const off = try conform.firstNonConformingField(&structs, &enums, &confs, .{}, Type.structT(0), hash_pid, &memo, gpa, &co, &.{});
    try testing.expect(off != null);
    try testing.expectEqualStrings("name", off.?.name);
    try testing.expect(Type.eql(Type.str, off.?.ty));
}

test "`P{..}.to_string()` on an all-Display-fields struct DERIVES exactly one recipe" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\fn main() {
        \\ s := P{ x: 4, y: 2 }.to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // `to_string` on a Display arg records ONE structural Display derive
    // (struct id 0 -> `Display$display$s0`), no diagnostic, ret str (the built rendering),
    // `self`-only (1 param).
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Display$display$s0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.display, c.result.derives[0].kind);
    try testing.expectEqual(Kind.str, c.result.derives[0].ret.kind);
    try testing.expectEqual(@as(usize, 1), c.result.derives[0].params.len);
}

test "`(enum value).to_string()` derives one Display recipe (Display$display$e0)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { A(int), B }
        \\fn main() {
        \\ s := E.A(3).to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqualStrings("Display$display$e0", c.result.derives[0].name.?);
    try testing.expectEqual(Derive.Kind.display, c.result.derives[0].kind);
}

test "`to_string` recurses through a NESTED aggregate (two Display recipes)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Point { x: int, y: int }
        \\struct Line { from: Point, to: Point }
        \\fn main() {
        \\ s := Line{ from: Point{ x: 0, y: 0 }, to: Point{ x: 4, y: 2 } }.to_string()
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
        \\ s := q.v.to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // `q.v.to_string()` renders an int (a prelude conformance, no derive); `Q` itself is
    // never rendered, so no `Display$display$s*` unit is synthesized.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "explicit `impl P has Display` OVERRIDES the derive (zero synthetic units)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Display {
        \\ fn display(self) -> str { return "x" }
        \\}
        \\fn main() {
        \\ s := P{ x: 1 }.to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The explicit impl wins at `resolveConformanceMethod`/`findConformance`, so no structural
    // derive fires.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "`x.to_string()` on a struct records one Display derive (to_string == display)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\fn main() {
        \\ s := P{ x: 4, y: 2 }.to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // to_string reuses the SAME `conformsTo(display)` gate print uses: it records ONE
    // structural Display derive and types the call `str`.
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqual(Derive.Kind.display, c.result.derives[0].kind);
    try testing.expectEqual(Kind.str, c.result.derives[0].ret.kind);
}

test "`to_string` on a struct with a non-Display (float) field is T0031 (shared gate)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Bad { f: float }
        \\fn main() {
        \\ s := Bad{ f: 1.5 }.to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len >= 1);
    try testing.expectEqual(codes.Code.T0031, c.result.diags[0].code);
    // The message uses the `to_string` verb (print's would mislead).
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "to_string") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "float") != null);
}

test "a unit-returning custom `impl Display` is a clean T0024 (ret must be str)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Display {
        \\ fn display(self) { }
        \\}
        \\fn main() {
        \\ s := P{ x: 1 }.to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len >= 1);
    try testing.expectEqual(codes.Code.T0024, c.result.diags[0].code);
}

test "`\"..\".to_string()` types clean and derives nothing (str path unchanged)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() {
        \\ s := "hi".to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "`'A'.to_string()` derives ONE shared char Display witness (UTF-8 encoder, not inlined)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() {
        \\ s := 'A'.to_string()
        \\}
        \\
    );
    defer c.deinit(gpa);
    // char no longer carries a bare Display row, so displaying it records a normal structural
    // Display derive: ONE `Display$display$s<charId>` recipe whose emitter is overridden to the
    // UTF-8 encoder. Every char-display site CALLs it (vs the former per-site inline encoder
    // that overflowed the frame past ~6 displays). Suppression ("A" not "char(65)") is upheld
    // by the overriding emitter — guarded end-to-end by tests/corpora/language-features/io/char_suppress.toy.
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

fn expectChecksClean(source: []const u8) !void {
    var c = try checkSource(source);
    defer c.deinit(testing.allocator);
    for (c.result.diags) |d| std.debug.print("unexpected: {s}\n", .{d.message});
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "a bounded call grounds a protocol's Option[X] return through the App" {
    try expectChecksClean(
        \\protocol Tagged[X] { fn tag(self, n: X) -> Option[X] }
        \\struct P { x: int }
        \\impl P has Tagged[int] { fn tag(self, n: int) -> Option[int] { Option[int].some(n + self.x) } }
        \\fn via[T has Tagged[int]](x: T, n: int) -> Option[int] { x.tag(n) }
        \\fn local[T has Tagged[int]](x: T, n: int) -> int {
        \\ r := x.tag(n)
        \\ return match r { .some(v) -> v, .none -> 0 }
        \\}
        \\fn main() -> int { return match via(P { x: 1 }, 2) { .some(v) -> v + local(P { x: 1 }, 2), .none -> 0 } }
        \\
    );
}

test "a bounded call grounds a protocol's Option[X] param through the App" {
    try expectChecksClean(
        \\protocol Take[X] { fn take(self, n: Option[X]) -> int }
        \\struct P { x: int }
        \\impl P has Take[int] { fn take(self, n: Option[int]) -> int { match n { .some(v) -> v + self.x, .none -> 0 } } }
        \\fn via[T has Take[int]](x: T, n: int) -> int { x.take(Option[int].some(n)) }
        \\fn main() -> int { return via(P { x: 1 }, 2) }
        \\
    );
}

test "an explicit-args bounded call grounds a protocol's Option[X] return" {
    try expectChecksClean(
        \\protocol Tagged[X] { fn tag(self, n: X) -> Option[X] }
        \\struct P { x: int }
        \\impl P has Tagged[int] { fn tag(self, n: int) -> Option[int] { Option[int].some(n + self.x) } }
        \\fn via[T has Tagged[int]](x: T, n: int) -> Option[int] { x.tag[int](n) }
        \\fn main() -> int { return match via(P { x: 1 }, 2) { .some(v) -> v, .none -> 0 } }
        \\
    );
}

test "a bounded call grounds Option[Self] to the receiver when it is not the first param" {
    try expectChecksClean(
        \\protocol Wrap { fn wrap(self) -> Option[Self] }
        \\struct P { x: int }
        \\impl P has Wrap { fn wrap(self) -> Option[P] { Option[P].some(self) } }
        \\fn via[U, T has Wrap](_u: U, x: T) -> int {
        \\ w: Option[T] = x.wrap()
        \\ return match w { .some(_) -> 3, .none -> 0 }
        \\}
        \\fn main() -> int { return via(true, P { x: 1 }) }
        \\
    );
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
    // A user protocol is not structurally derivable → suggest the `impl`.
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "impl Q has Doubler") != null);
    // The non-conforming instantiation is SKIPPED (no twice$Q instance minted).
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
}

test "a non-conforming type at a DERIVABLE bound suggests structural derivation" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Bad { f: float }
        \\fn needs_display[T has Display](x: T) -> str { x.to_string() }
        \\fn main() -> int {
        \\  s := needs_display(Bad{ f: 1.5 })
        \\  return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(codes.Code.T0023, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "Display") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "derives automatically") != null);
}

test "a scalar at a derivable bound gets no 'every field' hint; a float's Display names std/float" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn needs_display[T has Display](x: T) -> str { x.to_string() }
        \\fn needs_hash[T has Hash](x: T) -> int { x.hash() }
        \\fn main() -> int {
        \\  s := needs_display(1.5)
        \\  h := needs_hash(2.5)
        \\  return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), c.result.diags.len);
    for (c.result.diags) |d| {
        try testing.expectEqual(codes.Code.T0023, d.code);
        try testing.expect(std.mem.indexOf(u8, d.message, "every field") == null);
    }
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "add 'import std/float'") != null);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[1].message, "impl float has Hash") != null);
}

test "a program's own protocol named Display gets no std/float hint" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Display { fn display(self) -> str }
        \\fn needs[T has Display](_x: T) -> int { 0 }
        \\fn main() -> int { return needs(1.5) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0023, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "std/float") == null);
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

test "a warning in a bounded body does not poison the template: it is still instantiated" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Twice { fn twice(self) -> Self }
        \\impl int has Twice { fn twice(self) -> int { self * 2 } }
        \\fn dbl[T has Twice](o: T) -> T { if false { o } else { o.twice() } }
        \\fn main() -> int { dbl(6) }
        \\
    );
    defer c.deinit(gpa);
    for (c.result.diags) |d| try testing.expect(d.severity != .err);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
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

test "a generic receiver with two conformances is T0025 wherever its method is selected by name" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[T] has Into[bool] { fn into(self) -> bool { true } }
        \\impl Box[T] has Into[int] { fn into(self) -> int { 7 } }
        \\fn call(b: Box[int]) -> int { return b.into() }
        \\fn bound(b: Box[int]) -> int {
        \\ f := b.into
        \\ return f()
        \\}
        \\fn main() -> int {
        \\ b := Box { v: 1 }
        \\ n: int = b.into[int]()
        \\ t: bool = b.into[bool]()
        \\ return call(b) + bound(b) + n + (if t { 1 } else { 0 })
        \\}
        \\
    );
    defer c.deinit(gpa);
    var n25: usize = 0;
    for (c.result.diags) |d| {
        try testing.expectEqual(codes.Code.T0025, d.code);
        n25 += 1;
        const bool_at = std.mem.indexOf(u8, d.message, "Into[bool]") orelse return error.TestUnexpectedResult;
        const int_at = std.mem.indexOf(u8, d.message, "Into[int]") orelse return error.TestUnexpectedResult;
        try testing.expect(bool_at < int_at);
    }
    try testing.expectEqual(@as(usize, 2), n25);
}

/// The message of `src`'s only T0025, which `c` holds once checked.
fn onlyT0025(c: *const Checked) ![]const u8 {
    var msg: ?[]const u8 = null;
    for (c.result.diags) |d| if (d.severity == .err) {
        try testing.expectEqual(codes.Code.T0025, d.code);
        try testing.expect(msg == null);
        msg = d.message;
    };
    return msg orelse error.TestUnexpectedResult;
}

fn expectNoErrors(src: []const u8) !void {
    var c = try checkSource(src);
    defer c.deinit(testing.allocator);
    for (c.result.diags) |d| if (d.severity == .err) {
        std.debug.print("following the hint gave: {s}\n{s}\n", .{ d.message, src });
        return error.TestUnexpectedResult;
    };
}

test "every T0025 hint compiles when it is followed literally" {
    const gpa = testing.allocator;
    const Case = struct {
        decls: []const u8,
        /// The call in `main` that T0025 reports, on the local `p`, and how it reads
        /// through a bounded fn (`{f}` stands for its name).
        p: []const u8,
        call: []const u8,
        via: []const u8 = "",
        explicit: usize,
        bounded: bool,
    };
    const cases = [_]Case{
        .{
            .decls = "struct P { x: int }\nimpl P has Into[int] { fn into(self) -> int { self.x } }\nimpl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }\n",
            .p = "P { x: 1 }",
            .call = "p.into()",
            .explicit = 2,
            .bounded = false,
        },
        .{
            .decls = "struct Box[T] { v: T }\nimpl Box[T] has Into[T] { fn into(self) -> T { self.v } }\nimpl Box[T] has Into[Option[T]] { fn into(self) -> Option[T] { .some(self.v) } }\n",
            .p = "Box { v: Box { v: 1 } }",
            .call = "p.into()",
            .explicit = 2,
            .bounded = false,
        },
        .{
            .decls = "protocol Show { fn render(self) -> int }\nprotocol Debug { fn render(self) -> int }\nstruct P { x: int }\nimpl P has Show { fn render(self) -> int { 1 } }\nimpl P has Debug { fn render(self) -> int { 2 } }\n",
            .p = "P { x: 1 }",
            .call = "p.render()",
            .via = "{f}(p)",
            .explicit = 0,
            .bounded = true,
        },
        .{
            .decls = "protocol Show { fn render(self, k: int) -> int }\nprotocol Debug { fn render(self, k: int) -> int }\nimpl int has Show { fn render(self, k: int) -> int { self + k } }\nimpl int has Debug { fn render(self, k: int) -> int { self - k } }\n",
            .p = "4",
            .call = "p.render(1)",
            .via = "{f}(p, 1)",
            .explicit = 0,
            .bounded = true,
        },
        .{
            .decls = "protocol Tagged[X] { fn tag(mut self, n: X, s: str) -> X }\nprotocol Named[X] { fn tag(mut self, n: X, s: str) -> X }\nstruct P { x: int }\nimpl P has Tagged[int] { fn tag(mut self, n: int, _s: str) -> int { n } }\nimpl P has Named[int] { fn tag(mut self, _n: int, _s: str) -> int { 0 } }\n",
            .p = "P { x: 1 }",
            .call = "p.tag(1, \"s\")",
            .via = "{f}(p, 1, \"s\")",
            .explicit = 0,
            .bounded = true,
        },
        .{
            .decls = "protocol Conv { fn into(self) -> int }\nstruct P { x: int }\nimpl P has Into[int] { fn into(self) -> int { self.x } }\nimpl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }\nimpl P has Conv { fn into(self) -> int { 5 } }\n",
            .p = "P { x: 1 }",
            .call = "p.into()",
            .via = "{f}(p)",
            .explicit = 2,
            .bounded = true,
        },
    };
    for (cases) |k| {
        const src = try std.fmt.allocPrint(gpa, "{s}fn main() -> int {{\n p := {s}\n _r := {s}\n return 0\n}}\n", .{ k.decls, k.p, k.call });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        const msg = try onlyT0025(&c);
        const member = k.call[2..std.mem.indexOfScalar(u8, k.call, '(').?];
        // Each `.m[..]()` the hint offers, in place of the call's own `.m()`.
        var explicit: usize = 0;
        var rest = msg;
        const dot_m = try std.fmt.allocPrint(gpa, ".{s}[", .{member});
        defer gpa.free(dot_m);
        while (std.mem.indexOf(u8, rest, dot_m)) |at| {
            const end = std.mem.indexOfPos(u8, rest, at, "]()").? + 3;
            const fixed = try std.fmt.allocPrint(gpa, "{s}fn main() -> int {{\n p := {s}\n _r := p{s}\n return 0\n}}\n", .{ k.decls, k.p, rest[at..end] });
            defer gpa.free(fixed);
            try expectNoErrors(fixed);
            explicit += 1;
            rest = rest[end..];
        }
        try testing.expectEqual(k.explicit, explicit);
        // The bounded fn it offers, declared and called in place of the method.
        const fn_at = std.mem.indexOf(u8, msg, "e.g. 'fn ");
        try testing.expectEqual(k.bounded, fn_at != null);
        if (fn_at) |at| {
            const text = msg[at + 6 .. msg.len - 1];
            const name = text[3..std.mem.indexOfScalar(u8, text, '[').?];
            const call = try std.mem.replaceOwned(u8, gpa, k.via, "{f}", name);
            defer gpa.free(call);
            const fixed = try std.fmt.allocPrint(gpa, "{s}{s}\nfn main() -> int {{\n p := {s}\n _r := {s}\n return 0\n}}\n", .{ k.decls, text, k.p, call });
            defer gpa.free(fixed);
            try expectNoErrors(fixed);
        }
    }
}

test "every T0025 hint at a method value compiles when it is followed literally" {
    const gpa = testing.allocator;
    const Case = struct { decls: []const u8, p: []const u8, value: []const u8, offered: usize };
    const into = "impl P has Into[int] { fn into(self) -> int { self.x } }\nimpl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }\n";
    const cases = [_]Case{
        .{ .decls = "struct P { x: int }\n" ++ into, .p = "P { x: 1 }", .value = "p.into", .offered = 2 },
        .{ .decls = "struct P { x: int }\n" ++ into, .p = "P { x: 1 }", .value = "p.into[int]", .offered = 2 },
        .{ .decls = "struct Box[T] { v: T }\nimpl Box[T] has Into[int] { fn into(self) -> int { 7 } }\nimpl Box[T] has Into[bool] { fn into(self) -> bool { true } }\n", .p = "Box { v: 1 }", .value = "p.into", .offered = 2 },
        .{ .decls = "struct Box[T] { v: T }\nimpl Box[T] has Into[int] { fn into(self) -> int { 7 } }\nimpl Box[T] has Into[bool] { fn into(self) -> bool { true } }\n", .p = "Box { v: 1 }", .value = "p.into[bool]", .offered = 2 },
        .{ .decls = "protocol Tagged[X] { fn tag(self, n: X, s: str) -> X }\nstruct P { x: int }\nimpl P has Tagged[int] { fn tag(self, n: int, _s: str) -> int { n } }\nimpl P has Tagged[bool] { fn tag(self, n: bool, _s: str) -> bool { n } }\n", .p = "P { x: 1 }", .value = "p.tag", .offered = 2 },
    };
    for (cases) |k| {
        const src = try std.fmt.allocPrint(gpa, "{s}fn main() -> int {{\n p := {s}\n _f := {s}\n return 0\n}}\n", .{ k.decls, k.p, k.value });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        const msg = try onlyT0025(&c);
        var offered: usize = 0;
        var rest = msg;
        while (std.mem.indexOf(u8, rest, "'fn(")) |at| {
            const end = std.mem.indexOfPos(u8, rest, at, "}'").? + 1;
            const fixed = try std.fmt.allocPrint(gpa, "{s}fn main() -> int {{\n p := {s}\n _f := {s}\n return 0\n}}\n", .{ k.decls, k.p, rest[at + 1 .. end] });
            defer gpa.free(fixed);
            try expectNoErrors(fixed);
            offered += 1;
            rest = rest[end..];
        }
        errdefer std.debug.print("{s}\n", .{msg});
        try testing.expectEqual(k.offered, offered);
    }
}

test "an unbounded template's per-instance T0025 and T0016 hints read as the template does" {
    const gpa = testing.allocator;
    const box = "struct Box[T] { v: T }\nimpl Box[T] has Into[T] { fn into(self) -> T { self.v } }\nimpl Box[T] has Into[Option[T]] { fn into(self) -> Option[T] { .some(self.v) } }\n";
    const pick = "fn pick[A, B](a: A) -> A { a }\n";
    const Case = struct { src: []const u8, code: codes.Code, hint: []const u8, fixed: []const u8 };
    const cases = [_]Case{
        .{
            .src = box ++ "fn h[T](b: Box[T]) -> int {\n _x := b.into()\n 0\n}\nfn main() -> int { return h(Box { v: 1 }) + h(Box { v: \"s\" }) }\n",
            .code = .T0025,
            .hint = "(Into[T] and Into[Option[T]]); add explicit protocol type arguments, e.g. .into[T]() or .into[Option[T]]()",
            .fixed = box ++ "fn h[T](b: Box[T]) -> int {\n _x := b.into[T]()\n 0\n}\nfn main() -> int { return h(Box { v: 1 }) + h(Box { v: \"s\" }) }\n",
        },
        .{
            .src = pick ++ "fn g[T](t: T) -> T { pick(t) }\nfn main() -> int {\n x := g(1)\n y := g(\"s\")\n return x + y.len()\n}\n",
            .code = .T0016,
            .hint = "e.g. pick[T, int](..)",
            .fixed = pick ++ "fn g[T](t: T) -> T { pick[T, int](t) }\nfn main() -> int {\n x := g(1)\n y := g(\"s\")\n return x + y.len()\n}\n",
        },
        // A concrete arg that one instance's binding happens to equal still reads as written.
        .{
            .src = pick ++ "fn g[T](t: T) -> T {\n _p := pick(5)\n t\n}\nfn main() -> int {\n x := g(1)\n y := g(\"s\")\n return x + y.len()\n}\n",
            .code = .T0016,
            .hint = "e.g. pick[int, int](..)",
            .fixed = pick ++ "fn g[T](t: T) -> T {\n _p := pick[int, int](5)\n t\n}\nfn main() -> int {\n x := g(1)\n y := g(\"s\")\n return x + y.len()\n}\n",
        },
    };
    for (cases) |k| {
        var c = try checkSource(k.src);
        defer c.deinit(gpa);
        errdefer for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
        var n: usize = 0;
        for (c.result.diags) |d| if (d.severity == .err) {
            try testing.expectEqual(k.code, d.code);
            try testing.expect(std.mem.endsWith(u8, d.message, k.hint));
            n += 1;
        };
        try testing.expectEqual(@as(usize, 1), n);
        try expectNoErrors(k.fixed);
    }
}

test "a site that fails in every instance of a template reports once" {
    const gpa = testing.allocator;
    const box = "struct Box[T] { v: T }\n";
    const Case = struct { src: []const u8, code: codes.Code, message: []const u8 };
    const cases = [_]Case{
        // `int` is the declared type and also one instance's `T`, so the instances
        // spell their templates differently; the site still reports once.
        .{
            .src = box ++ "fn f[T](x: T) -> int {\n _y: int = Box { v: x }\n 0\n}\nfn main() -> int { return f(1) + f(true) }\n",
            .code = .T0041,
            .message = "cannot bind Box[int] to '_y' of type int",
        },
        .{
            .src = "fn f[T](v: T) -> int {\n _g: fn(T) -> int = fn(_x) { true }\n 0\n}\nfn main() -> int { return f(1) + f(\"s\") }\n",
            .code = .T0042,
            .message = "lambda has type fn(int) -> bool, but fn(int) -> int is expected: its value is bool, not int",
        },
        // Polymorphic recursion: one instance per nesting depth (and a T0017 where the
        // depth runs out), one report.
        .{
            .src = box ++ "fn grow[T](x: T, n: int) -> int {\n if n == 0 { return 0 }\n _bad: int = Box { v: x }\n grow(Box { v: x }, n - 1)\n}\nfn main() -> int { return grow(1, 3) }\n",
            .code = .T0041,
            .message = "cannot bind Box[int] to '_bad' of type int",
        },
    };
    for (cases) |k| {
        var c = try checkSource(k.src);
        defer c.deinit(gpa);
        errdefer for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
        var n: usize = 0;
        for (c.result.diags) |d| if (d.severity == .err and d.code != .T0017) {
            try testing.expectEqual(k.code, d.code);
            try testing.expectEqualStrings(k.message, d.message);
            n += 1;
        };
        try testing.expectEqual(@as(usize, 1), n);
    }
}

test "T0025 on a generic type offers a bounded fn, and following it compiles" {
    const gpa = testing.allocator;
    const decls =
        \\protocol Show { fn render(self) -> int }
        \\protocol Debug { fn render(self) -> int }
        \\struct Box[T] { v: T }
        \\impl Box[T] has Show { fn render(self) -> int { 1 } }
        \\impl Box[T] has Debug { fn render(self) -> int { 2 } }
        \\
    ;
    var c = try checkSource(decls ++ "fn main() -> int { return Box { v: 1 }.render() }\n");
    defer c.deinit(gpa);
    const msg = try onlyT0025(&c);
    try testing.expect(std.mem.endsWith(u8, msg, "(Show and Debug); call it through a fn bounded by 'Show', e.g. 'fn render_via[T has Show](x: T) -> int { x.render() }'"));
    try expectNoErrors(decls ++ "fn render_via[T has Show](x: T) -> int { x.render() }\nfn main() -> int { return render_via(Box { v: 1 }) }\n");
}

test "a call on a bounded param takes the bound's conformance in each instance" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Show { fn render(self) -> int }
        \\protocol Debug { fn render(self) -> int }
        \\struct P { x: int }
        \\impl P has Show { fn render(self) -> int { 1 } }
        \\impl P has Debug { fn render(self) -> int { 2 } }
        \\impl P has Into[int] { fn into(self) -> int { self.x } }
        \\impl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }
        \\fn use_show[T has Show](t: T) -> int { t.render() }
        \\fn use_debug[T has Debug](t: T) -> int { t.render() }
        \\fn as_int[T has Into[int]](t: T) -> int { t.into() }
        \\fn conv[T has Into[U], U](t: T) -> U { t.into() }
        \\fn main() -> int {
        \\ p := P { x: 3 }
        \\ return use_show(p) + use_debug(p) + as_int(p) + (if conv[P, bool](p) { 1 } else { 0 })
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Each instance's call selects the witness of its own bound's conformance.
    var seen: usize = 0;
    for (c.result.instances) |inst| {
        const full = c.result.sigs[inst.template_gid].name;
        const tname = full[(std.mem.lastIndexOfScalar(u8, full, '.') orelse return error.TestUnexpectedResult) + 1 ..];
        for (inst.callees) |cl| switch (cl.view()) {
            .method => |mi| {
                const m = c.result.methods[mi];
                const proto = c.result.protocols[m.protocol_id.?];
                const want: []const u8 = if (std.mem.eql(u8, tname, "use_show")) "Show" else if (std.mem.eql(u8, tname, "use_debug")) "Debug" else "Into";
                try testing.expectEqualStrings(want, proto.name);
                if (std.mem.eql(u8, tname, "as_int")) try testing.expectEqual(Type.int.kind, m.protocol_args[0].kind);
                if (std.mem.eql(u8, tname, "conv")) try testing.expectEqual(Type.bool.kind, m.protocol_args[0].kind);
                seen += 1;
            },
            else => {},
        };
    }
    try testing.expectEqual(@as(usize, 4), seen);
}

test "two bounds of one generic protocol take their own args at one instance type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\impl P has Into[int] { fn into(self) -> int { self.x } }
        \\impl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }
        \\fn f[A has Into[int], B has Into[bool]](a: A, b: B) -> int {
        \\ n: int = a.into()
        \\ ok: bool = b.into()
        \\ return if ok { n } else { 0 }
        \\}
        \\fn main() -> int {
        \\ p := P { x: 5 }
        \\ return f(p, p)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var kinds: [2]@TypeOf(Type.int.kind) = undefined;
    var seen: usize = 0;
    for (c.result.instances) |inst| for (inst.callees) |cl| switch (cl.view()) {
        .method => |mi| {
            kinds[seen] = c.result.methods[mi].protocol_args[0].kind;
            seen += 1;
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 2), seen);
    try testing.expectEqual(Type.int.kind, kinds[0]);
    try testing.expectEqual(Type.bool.kind, kinds[1]);
}

test "the T0016 hint, in place of the callee, gets past T0001 and T0016 at every site" {
    const gpa = testing.allocator;
    const decls =
        \\enum Either[A, B] { left(A), right(B) }
        \\struct Pair[A, B] { a: A }
        \\impl Pair[A, B] {
        \\    fn make(a: A) -> Pair[A, B] { Pair[A, B] { a: a } }
        \\    fn with[U](a: A, _u: U) -> Pair[A, B] { Pair[A, B] { a: a } }
        \\    fn conv[U, V](self, _v: V) -> Option[U] { Option[U].none }
        \\}
        \\fn pick[A, B](a: A) -> A { a }
        \\
    ;
    const Case = struct { stmt: []const u8, callee: []const u8, hint: []const u8 };
    const cases = [_]Case{
        .{ .stmt = "_r := Either.left(true)", .callee = "Either.left", .hint = "Either[bool, int].left" },
        .{ .stmt = "_r := Either.right(\"s\")", .callee = "Either.right", .hint = "Either[int, str].right" },
        .{ .stmt = "_r := Pair { a: true }", .callee = "Pair", .hint = "Pair[bool, int]{ .. }" },
        .{ .stmt = "_r := pick(true)", .callee = "pick", .hint = "pick[bool, int](..)" },
        .{ .stmt = "_r := Pair.make(true)", .callee = "Pair.make", .hint = "Pair[bool, int].make" },
        .{ .stmt = "_r := Pair.with[str](true, \"s\")", .callee = "Pair.with", .hint = "Pair[int, int].with" },
        .{ .stmt = "_r := Pair[int, int] { a: 1 }.conv(true)", .callee = "conv", .hint = "conv[int, bool](..)" },
    };
    for (cases) |k| {
        const src = try std.mem.concat(gpa, u8, &.{ decls, "fn main() -> int {\n ", k.stmt, "\n return 0\n}\n" });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        var msg: ?[]const u8 = null;
        for (c.result.diags) |d| if (d.code == codes.Code.T0016) {
            msg = d.message;
        };
        const m = msg orelse {
            for (c.result.diags) |d| std.debug.print("{s}: got {s}: {s}\n", .{ k.stmt, @tagName(d.code), d.message });
            return error.TestExpectedDiagnostic;
        };
        const text = m[std.mem.indexOf(u8, m, "e.g. ").? + 5 ..];
        errdefer std.debug.print("{s}: hint '{s}'\n", .{ k.stmt, text });
        try testing.expectEqualStrings(k.hint, text);
        // The hint names the callee with its type args; `{ .. }` and `(..)` stand for the
        // rest of the expression, which stays as written.
        const callee_text = if (std.mem.endsWith(u8, text, "{ .. }")) text[0 .. text.len - 6] else if (std.mem.endsWith(u8, text, "(..)")) text[0 .. text.len - 4] else text;
        const fixed_stmt = try std.mem.replaceOwned(u8, gpa, k.stmt, k.callee, callee_text);
        defer gpa.free(fixed_stmt);
        const fixed = try std.mem.concat(gpa, u8, &.{ decls, "fn main() -> int {\n ", fixed_stmt, "\n return 0\n}\n" });
        defer gpa.free(fixed);
        var f = try checkSource(fixed);
        defer f.deinit(gpa);
        for (f.result.diags) |d| {
            errdefer std.debug.print("{s}: got {s}: {s}\n", .{ fixed_stmt, @tagName(d.code), d.message });
            try testing.expect(d.code != codes.Code.T0001 and d.code != codes.Code.T0016);
        }
    }
}

test "pickGenericMethod is ambiguous only between conformances" {
    const inherent: TemplateMethod = .{ .recv_ctor = 0, .recv_is_enum = false, .name = "m", .fn_id = 0 };
    var a = inherent;
    a.fn_id = 1;
    a.protocol_id = 0;
    var b = inherent;
    b.fn_id = 2;
    b.protocol_id = 1;
    var std_m = inherent;
    std_m.fn_id = 3;
    std_m.bundled = true;
    var other = a;
    other.recv_ctor = 1;
    const Case = struct { table: []const TemplateMethod, want: std.meta.Tag(TemplatePick), fn_id: u32 = 0 };
    const cases = [_]Case{
        .{ .table = &.{}, .want = .none },
        .{ .table = &.{a}, .want = .one, .fn_id = 1 },
        .{ .table = &.{ a, b }, .want = .ambiguous },
        .{ .table = &.{ a, other, inherent }, .want = .one, .fn_id = 1 },
        .{ .table = &.{ std_m, a }, .want = .one, .fn_id = 1 },
        .{ .table = &.{ inherent, b, other }, .want = .one, .fn_id = 0 },
        .{ .table = &.{ inherent, a, b }, .want = .ambiguous },
    };
    for (cases) |k| {
        const got = pickGenericMethod(k.table, 0, false, "m", .program);
        try testing.expectEqual(k.want, std.meta.activeTag(got));
        if (got == .one) try testing.expectEqual(k.fn_id, got.one.fn_id);
    }
    try testing.expect(pickGenericMethod(&.{ a, b }, 0, true, "m", .program) == .none);
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

/// The layout ids of every reified function-type pair in a checked program.
fn fnPairIds(c: *const Checked, out: []u32) []u32 {
    var n: usize = 0;
    for (c.result.layouts, 0..) |l, i| if (l.fn_sig != null) {
        out[n] = @intCast(i);
        n += 1;
    };
    return out[0..n];
}

test "fn types are assignable iff structurally equal, aliases included" {
    const gpa = testing.allocator;
    var ok = try checkSource(
        \\type F = fn(int) -> int
        \\pub fn a(g: fn(int) -> int) -> int {
        \\ h: F = g
        \\ k: fn(int) -> int = h
        \\ return 0
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
    try testing.expect(Type.eql(identTypeOf(&ok, "g"), identTypeOf(&ok, "h")));

    var bad = try checkSource(
        \\pub fn keep(g: fn(int) -> int) -> int { return 0 }
        \\pub fn a(g: fn(int) -> bool, u: fn(int, int) -> int) -> int {
        \\ h: fn(int) -> int = g
        \\ v: fn(int) -> int = u
        \\ w: fn() = g
        \\ return keep(g)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer bad.deinit(gpa);
    try testing.expectEqual(@as(usize, 4), bad.result.diags.len);
    try testing.expectEqual(codes.Code.T0041, bad.result.diags[0].code);
    try testing.expectEqualStrings("cannot bind fn(int) -> bool to 'h' of type fn(int) -> int", bad.result.diags[0].message);
    try testing.expectEqualStrings("cannot bind fn(int, int) -> int to 'v' of type fn(int) -> int", bad.result.diags[1].message);
    try testing.expectEqualStrings("cannot bind fn(int) -> bool to 'w' of type fn()", bad.result.diags[2].message);
    try testing.expectEqual(codes.Code.T0040, bad.result.diags[3].code);
    try testing.expectEqualStrings("argument 1: expected fn(int) -> int, got fn(int) -> bool; 'keep' takes (fn(int) -> int)", bad.result.diags[3].message);
}

test "a fn-typed param, field and generic-struct arg reify to one 16-byte pair per signature" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\struct Holder[T] { f: fn(Box[T]) -> int }
        \\pub struct H { run: fn(int) -> int }
        \\pub fn p(g: fn(int) -> int, b: Box[fn(int) -> int], h: H) -> int { return 0 }
        \\pub fn q(g: fn(Box[int]) -> int) -> int { return 0 }
        \\pub fn r(h: Holder[int]) -> int { return 0 }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var buf: [8]u32 = undefined;
    const pairs = fnPairIds(&c, &buf);
    try testing.expectEqual(@as(usize, 2), pairs.len);
    for (pairs) |id| {
        const l = c.result.layouts[id];
        try testing.expectEqual(@as(u32, 16), l.size);
        try testing.expectEqual(@as(u32, 8), l.@"align");
        try testing.expectEqualSlices(u32, &.{ 0, 8 }, l.offsets);
        try testing.expect(Type.eql(l.field_types[0], Type.int));
        try testing.expect(Type.eql(l.field_types[1], Type.rawptr));
        try testing.expectEqualStrings("code", l.field_names[0]);
        try testing.expectEqualStrings("env", l.field_names[1]);
        try testing.expectEqual(LayoutEngine.NativeStructFamily.none, l.native_family);
    }
    const sig_of = struct {
        fn f(cc: *const Checked, name: []const u8) Sig {
            for (cc.result.sigs) |sg| if (std.mem.eql(u8, sg.name, name)) return sg;
            unreachable;
        }
    }.f;
    const p = sig_of(&c, "main.p");
    const int_pair = p.params[0];
    try testing.expect(LayoutEngine.isFnStruct(int_pair, c.result.layouts));
    const ip = c.result.layouts[int_pair.struct_id].fn_sig.?;
    try testing.expectEqualSlices(Type, &.{Type.int}, ip.params);
    try testing.expect(Type.eql(ip.ret, Type.int));
    try testing.expectEqualStrings("fn$int$int", c.result.layouts[int_pair.struct_id].name);
    // The same signature spelled as a Box arg and as a struct field is the same pair.
    const box = c.result.layouts[p.params[1].struct_id];
    try testing.expect(Type.eql(box.field_types[0], int_pair));
    const hl = c.result.layouts[p.params[2].struct_id];
    try testing.expect(Type.eql(hl.field_types[0], int_pair));
    // A written fn(Box[int]) -> int and Holder[T]'s fn(Box[T]) -> int at T = int ground
    // to one pair.
    const q = sig_of(&c, "main.q");
    const holder = c.result.layouts[sig_of(&c, "main.r").params[0].struct_id];
    try testing.expect(LayoutEngine.isFnStruct(q.params[0], c.result.layouts));
    try testing.expect(Type.eql(holder.field_types[0], q.params[0]));
    const bp = c.result.layouts[q.params[0].struct_id].fn_sig.?;
    try testing.expectEqualStrings("Box$int", c.result.layouts[bp.params[0].struct_id].name);
}

test "a struct field of fn type lays out 16 bytes at offset 8" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\pub struct S { id: int, run: fn(int) -> int }
        \\pub fn f(s: S) -> int { return s.id }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const s = for (c.result.layouts) |l| {
        if (std.mem.eql(u8, l.name, "S")) break l;
    } else unreachable;
    try testing.expectEqualSlices(u32, &.{ 0, 8 }, s.offsets);
    try testing.expectEqual(@as(u32, 24), s.size);
    try testing.expect(LayoutEngine.isFnStruct(s.field_types[1], c.result.layouts));
}

test "no .func survives reify" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\pub enum E { a(fn() -> int), b }
        \\pub struct S { run: fn(int) -> int }
        \\fn apply[T](f: fn(T) -> T, x: T) -> T { return x }
        \\pub fn use(g: fn(int) -> int, s: S, e: E) -> int {
        \\ h: fn(int) -> int = g
        \\ return apply[int](h, 1) + apply(s.run, 2)
        \\}
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    const no_func = struct {
        fn ok(t: Type) bool {
            return t.kind != .func and t.kind != .app and t.kind != .type_var;
        }
    }.ok;
    for (c.result.node_types) |mnt| for (mnt) |ty| try testing.expect(no_func(ty));
    for (c.result.instances) |inst| {
        for (inst.node_types) |ty| try testing.expect(no_func(ty));
        for (inst.args) |a| try testing.expect(no_func(a));
        for (inst.params) |pp| try testing.expect(no_func(pp));
        try testing.expect(no_func(inst.ret));
    }
    for (c.result.sigs) |sg| {
        if (sg.isGenericTemplate()) continue;
        for (sg.params) |pp| try testing.expect(no_func(pp));
        try testing.expect(no_func(sg.ret));
    }
    // Layout tables also hold the (never-lowered) prelude templates, whose payloads keep
    // their type_var patterns; none of them holds a fn type.
    for (c.result.layouts) |l| for (l.field_types) |ft| try testing.expect(ft.kind != .func);
    for (c.result.enum_layouts) |el| for (el.variants) |v| for (v.field_types) |ft| try testing.expect(ft.kind != .func);
    // The template keeps its fn-typed param, so it still reads as a template.
    for (c.result.sigs) |sg| if (std.mem.eql(u8, sg.name, "main.apply")) try testing.expect(sg.isGenericTemplate());
}

test "a struct with a fn field derives Display but no Eq or Hash" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\pub struct S { id: int, run: fn(int) -> int }
        \\pub fn eq(a: S, b: S) -> bool { return a == b }
        \\pub fn h(a: S) -> int { return a.hash() }
        \\pub fn d(a: S) -> str { return a.to_string() }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0029, c.result.diags[0].code);
    try testing.expectEqual(codes.Code.T0030, c.result.diags[1].code);
    for (c.result.diags) |dg| try testing.expect(std.mem.indexOf(u8, dg.message, "'fn(int) -> int'") != null);
    try testing.expectEqual(@as(usize, 1), c.result.derives.len);
    try testing.expectEqual(Derive.Kind.display, c.result.derives[0].kind);
}

/// The `nth` (node order) value-position identifier spelled `name` that resolves to a fn.
fn fnIdent(c: *const Checked, name: []const u8, nth: usize) !Ast.Index {
    var seen: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .identifier or c.resolve.resolutions[0][i] != .func) continue;
        if (!std.mem.eql(u8, c.tokens[n.main_token].text(c.source), name)) continue;
        if (seen == nth) return Ast.Index.from(@intCast(i));
        seen += 1;
    }
    return error.TestIdentNotFound;
}

/// The `nth` (node order) `.call` node.
fn nthCall(c: *const Checked, nth: usize) !Ast.Node {
    var seen: usize = 0;
    for (c.tree.nodes) |n| {
        if (n.tag != .call) continue;
        if (seen == nth) return n;
        seen += 1;
    }
    return error.TestCallNotFound;
}

fn hasCode(c: *const Checked, code: codes.Code) bool {
    for (c.result.diags) |d| if (d.code == code) return true;
    return false;
}

test "a named fn in value position types as its fn type and records fn_ref" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn g(x: int) -> int { return x }
        \\fn main() -> int {
        \\ h := g
        \\ return h(1)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const g = try fnIdent(&c, "g", 0);
    const ty = c.result.node_types[0][g.int()];
    try testing.expect(LayoutEngine.isFnStruct(ty, c.result.layouts));
    const fs = c.result.layouts[ty.struct_id].fn_sig.?;
    try testing.expectEqual(@as(usize, 1), fs.params.len);
    try testing.expectEqual(Kind.int, fs.params[0].kind);
    try testing.expectEqual(Kind.int, fs.ret.kind);
    const gid = c.resolve.resolutions[0][g.int()].func;
    try testing.expectEqual(Callee.View{ .fn_ref = gid }, c.result.callees[0][g.int()].view());
}

test "calling a fn-typed local records value at callSlot" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn g(x: int) -> int { return x }
        \\fn main() -> int {
        \\ h := g
        \\ return h(1)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const call = try nthCall(&c, 0);
    const slot = Callee.callSlot(c.tree, call).?;
    try testing.expectEqual(call.lhs, slot);
    try testing.expectEqual(Callee.View.value, c.result.callees[0][slot.int()].view());
}

test "a direct call records its fn as a direct callee, not a fn_ref" {
    const gpa = testing.allocator;
    var c = try checkSource("fn g(x: int) -> int { return x }\nfn main() -> int { return g(1) }\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const call = try nthCall(&c, 0);
    const g = c.resolve.resolutions[0][call.lhs.int()].func;
    try testing.expectEqual(Callee.View{ .direct = g }, c.result.callees[0][call.lhs.int()].view());
    try testing.expectEqual(@as(usize, 0), c.result.adapters.len);
}

test "s.f(1) picks the method when both exist, else the fn-typed field" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct S { run: fn(int) -> int, go: fn(int) -> int }
        \\impl S { fn run(self, x: int) -> int { return x + 100 } }
        \\fn dbl(x: int) -> int { return x * 2 }
        \\fn main() -> int {
        \\ s := S { run: dbl, go: dbl }
        \\ return s.run(1) + s.go(3)
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The documented rule, which W0013 points out at the method.
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.W0013, c.result.diags[0].code);
    try testing.expectEqualStrings("run", (try calleeMethod(&c, "run", 0)).name);
    try testing.expectEqual(Callee.View.value, try calleeView(&c, "go", 0));
    const go = try memberSlot(&c, "go", 0);
    try testing.expect(LayoutEngine.isFnStruct(c.result.node_types[0][go.int()], c.result.layouts));
}

test "a generic struct's fn-typed field is called through its substituted type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { f: fn(T) -> T, v: T }
        \\fn dbl(x: int) -> int { return x * 2 }
        \\fn main() -> int {
        \\ b := Box[int] { f: dbl, v: 4 }
        \\ return b.f(b.v)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(Callee.View.value, try calleeView(&c, "f", 0));
}

test "calling a local poisoned by an earlier error adds no second error" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() -> int {\n f := 1 + true\n return f(1)\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0044, c.result.diags[0].code);
    // A local that is no function is still reported.
    var d = try checkSource("fn main() -> int {\n f := 1\n return f(1)\n}\n");
    defer d.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), d.result.diags.len);
    try testing.expect(std.mem.indexOf(u8, d.result.diags[0].message, "called value is not a function") != null);
}

test "a call whose callee is a call or an index is checked, not silently poisoned" {
    const gpa = testing.allocator;
    {
        var c = try checkSource("fn get() -> int { return 1 }\nfn main() -> int { return get()(1) }\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "called value is not a function") != null);
    }
    {
        var c = try checkSource("fn get() -> int { return 1 }\nfn main() -> int { return 1(2) }\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    }
    {
        var c = try checkSource(
            \\fn inc(x: int) -> int { return x + 1 }
            \\fn get() -> fn(int) -> int { return inc }
            \\fn main() -> int { return get()(2) }
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);
        // The outer call's slot is the inner call node.
        const outer = for (c.tree.nodes) |n| {
            if (n.tag == .call and c.tree.nodes[n.lhs.int()].tag == .call) break n;
        } else return error.TestCallNotFound;
        try testing.expectEqual(outer.lhs, Callee.callSlot(c.tree, outer).?);
        try testing.expectEqual(Callee.View.value, c.result.callees[0][outer.lhs.int()].view());
    }
    {
        var c = try checkSource(
            \\fn inc(x: int) -> int { return x + 1 }
            \\fn get() -> fn(int) -> int { return inc }
            \\fn main() -> int { return get()(true) }
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0040, c.result.diags[0].code);
    }
}

test "== on fn values is bool; f.to_string() is str; different fn types T0044; f.hash() T0018; f < g T0027" {
    const gpa = testing.allocator;
    {
        var c = try checkSource(
            \\fn inc(x: int) -> int { return x + 1 }
            \\fn dec(x: int) -> int { return x - 1 }
            \\fn main() -> int {
            \\ f := inc
            \\ s: str = f.to_string()
            \\ if f == inc { return 1 }
            \\ if f != dec { return 2 }
            \\ return 0
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    }
    const cases = [_]struct { body: []const u8, code: codes.Code }{
        .{ .body = " b := inc == not\n", .code = .T0044 },
        .{ .body = " h := inc.hash()\n", .code = .T0018 },
        .{ .body = " b := inc < dec\n", .code = .T0027 },
    };
    for (cases) |cs| {
        const src = try std.mem.concat(gpa, u8, &.{
            "fn inc(x: int) -> int { return x + 1 }\nfn dec(x: int) -> int { return x - 1 }\nfn not(b: bool) -> bool { return !b }\nfn main() {\n",
            cs.body,
            "}\n",
        });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try testing.expect(hasCode(&c, cs.code));
    }
}

test "apply[T](f: fn(T) -> T, x: T) instantiates once on T=int" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn apply[T](f: fn(T) -> T, x: T) -> T { return f(x) }
        \\fn inc(x: int) -> int { return x + 1 }
        \\fn dec(x: int) -> int { return x - 1 }
        \\fn main() -> int { return apply(inc, 1) + apply(dec, 5) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances[0].args.len);
    try testing.expectEqual(Kind.int, c.result.instances[0].args[0].kind);
}

test "call[T](f: fn() -> T) infers T from the fn's ret" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn call[T](f: fn() -> T) -> T { return f() }
        \\fn seven() -> int { return 7 }
        \\fn yes() -> bool { return true }
        \\fn main() -> int {
        \\ if call(yes) { return call(seven) }
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
    var kinds: [2]Kind = undefined;
    for (c.result.instances, 0..) |inst, i| kinds[i] = inst.args[0].kind;
    try testing.expect((kinds[0] == .int and kinds[1] == .bool) or (kinds[0] == .bool and kinds[1] == .int));
}

test "adapters are the deduped, sorted fn_refs, including refs inside an instance body" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn a1(x: int) -> int { return x }
        \\fn b2(x: int) -> int { return x }
        \\fn c3(x: int) -> int { return x }
        \\fn use[T](v: T) -> T {
        \\ f := a1
        \\ f(1)
        \\ return v
        \\}
        \\fn main() -> int {
        \\ g := c3
        \\ h := b2
        \\ k := c3
        \\ return g(1) + h(2) + k(3) + use(4)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 3), c.result.adapters.len);
    for (c.result.adapters[1..], c.result.adapters[0 .. c.result.adapters.len - 1]) |cur, prev| try testing.expect(prev.key < cur.key);
    for (c.result.adapters) |a| {
        try testing.expectEqual(Closure.AdapterKind.fn_value, a.kind);
        try testing.expect(std.mem.endsWith(u8, a.name, "$fnval"));
        try testing.expect(std.mem.startsWith(u8, a.name, a.target.name));
        try testing.expectEqual(@as(usize, 1), a.sig.params.len);
    }
    // `a1` is named only inside the generic body, so only the instance slice holds it.
    try testing.expectEqualStrings("main.a1", c.result.adapters[0].target.name);
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

test "an all-() reified struct instance (Box[()]) is rejected as empty" {
    // `()` is now a permitted zero-sized type-arg, so `Box{ v: nop() }` infers `Box[()]`
    // and reifies to a struct whose only field is `()` — an all-zero-sized aggregate, the
    // same hazard the empty-struct guard exists for, so it is rejected as empty (T0005).
    const gpa = testing.allocator;
    var c = try checkSource("struct Box[T] { v: T }\nfn nop() {}\nfn main() -> int {\n c := Box{ v: nop() }\n return 0\n}\n");
    defer c.deinit(gpa);
    var saw_empty = false;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0005) saw_empty = true;
    }
    try testing.expect(saw_empty);
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
    // A flat param's leaf IS the whole arg, so the message names the real types.
    try testing.expect(std.mem.indexOf(u8, saw.?.message, "int vs bool") != null);
    // A conflicting call is rejected, so it mints no instance.
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
}

test "a container-hidden bare-call conflict names the LEAF types, not the container ctor" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Wrap[T] { v: T }
        \\fn pair[T](_a: Wrap[T], _b: Wrap[T]) -> int { 0 }
        \\fn main() -> int { return pair(Wrap[int]{ v: 1 }, Wrap[bool]{ v: true }) }
        \\
    );
    defer c.deinit(gpa);
    var saw: ?DiagnosticSink.Diagnostic = null;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0015) saw = d;
    }
    try testing.expect(saw != null);
    // The clash is int vs bool inside the two Wrap args, not "Wrap vs Wrap".
    try testing.expect(std.mem.indexOf(u8, saw.?.message, "int vs bool") != null);
    // Two distinct source args ⇒ a related (earlier) span is set.
    try testing.expect(saw.?.related != DiagnosticSink.NO_RELATED);
    try testing.expectEqual(@as(usize, 0), c.result.instances.len);
}

test "an intra-arg conflict names the LEAF types and collapses to a single caret" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Both[K, V] { x: K, y: V }
        \\fn single[T](_m: Both[T, T]) -> int { 0 }
        \\fn main() -> int { return single(Both[int, bool]{ x: 1, y: true }) }
        \\
    );
    defer c.deinit(gpa);
    var saw: ?DiagnosticSink.Diagnostic = null;
    for (c.result.diags) |d| {
        if (d.code == codes.Code.T0015) saw = d;
    }
    try testing.expect(saw != null);
    try testing.expect(std.mem.indexOf(u8, saw.?.message, "int vs bool") != null);
    // One source arg ⇒ a single caret, no self-referential "previously defined here".
    try testing.expect(saw.?.related == DiagnosticSink.NO_RELATED);
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

test "an arg that takes its type from its slot takes what an earlier arg binds" {
    const gpa = testing.allocator;
    var c = try checkSource("enum Opt[T] { some(T), none }\nfn pair[U](_a: U, _b: Opt[U]) -> int { return 2 }\nfn main() -> int {\n t := 5\n return pair(t, Opt.none)\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    try testing.expectEqualStrings("main.pair$int", c.result.instances[0].name.?);
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

test "value-poison: a builtin used as a value reports T0049 and poisons the node" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n p := panic\n}\n");
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len >= 1);
    try testing.expectEqual(codes.Code.T0049, c.result.diags[0].code);
    try testing.expectEqual(Kind.invalid, identTypeOf(&c, "panic").kind);
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
    try testing.expectEqualStrings("argument 2: expected int, got bool; 'add' takes (int, int)", c.result.diags[0].message);
    try testing.expectEqual(codes.Code.T0040, c.result.diags[0].code);
    try testing.expect(c.result.diags[0].related != DiagnosticSink.NO_RELATED);
}

test "call arity mismatch carries the signature and a defined-here span (T0039)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        "fn add(a: int, b: int) -> int { return a + b }\nfn f() {\n x := add(1)\n return\n}\n",
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0039, c.result.diags[0].code);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "'add' takes (int, int)") != null);
    try testing.expect(c.result.diags[0].related != DiagnosticSink.NO_RELATED);
}

test "piped T0039 counts the piped value and points at the failing |> step" {
    const gpa = testing.allocator;
    const src = "fn add(a: int, b: int) -> int { return a + b }\nfn f() {\n x := 1 |> add\n return\n}\n";
    var c = try checkSource(src);
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    const d = c.result.diags[0];
    try testing.expectEqual(codes.Code.T0039, d.code);
    try testing.expectEqualStrings("expected 2 argument(s), got 1 (the piped value is argument 1); 'add' takes (int, int)", d.message);
    try testing.expect(d.related != DiagnosticSink.NO_RELATED);
    try testing.expectEqual(@as(u32, @intCast(std.mem.indexOf(u8, src, "|> add").?)), d.byte_offset);
}

test "piped T0040 marks argument 1 as the piped value and covers the lhs" {
    const gpa = testing.allocator;
    const src = "fn neg(a: int) -> int { return 0 - a }\nfn f() {\n x := !true |> neg\n return\n}\n";
    var c = try checkSource(src);
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    const d = c.result.diags[0];
    try testing.expectEqual(codes.Code.T0040, d.code);
    try testing.expectEqualStrings("argument 1 (the piped value): expected int, got bool; 'neg' takes (int)", d.message);
    const lhs = std.mem.indexOf(u8, src, "!true").?;
    try testing.expectEqual(@as(u32, @intCast(lhs)), d.byte_offset);
    try testing.expectEqual(@as(u32, @intCast(lhs + "!true".len)), d.end);
}

test "generic, method and variant count errors carry the piped hint" {
    const gpa = testing.allocator;
    const cases = [_]struct { body: []const u8, want: []const u8 }{
        .{ .body = "x := 1 |> pair", .want = "expected 2 argument(s), got 1 (the piped value is argument 1); 'pair' takes (T, T)" },
        .{ .body = "x := 1 |> pair[int]", .want = "expected 2 argument(s), got 1 (the piped value is argument 1); 'pair' takes (int, int)" },
        .{ .body = "x := true |> pair[int](2)", .want = "argument 1 (the piped value): expected int, got bool; 'pair' takes (int, int)" },
        .{ .body = "b := Box[int]{ x: 1 }\n x := true |> b.put", .want = "argument 1 (the piped value): expected int, got bool; 'put' takes (int)" },
        .{ .body = "x := 1 |> Shape.circle(2)", .want = "variant 'Shape.circle' expects 1 value(s), got 2 (the piped value is argument 1)" },
        // A plain call through the same paths carries no hint.
        .{ .body = "x := pair(1)", .want = "expected 2 argument(s), got 1; 'pair' takes (T, T)" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa,
            \\struct Box[T] {{ x: T }}
            \\impl Box[T] {{
            \\ fn put(self, y: T) -> T {{ return y }}
            \\}}
            \\enum Shape {{ circle(int) }}
            \\fn pair[T](a: T, b: T) -> T {{ return a }}
            \\fn f() {{
            \\ {s}
            \\ return
            \\}}
            \\
        , .{cs.body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer std.debug.print("case: {s}\n", .{cs.body});
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqualStrings(cs.want, c.result.diags[0].message);
    }
}

test "a hole pipe names the '_' as the piped value and needs no count hint" {
    const gpa = testing.allocator;
    const cases = [_]struct { body: []const u8, code: codes.Code, want: []const u8, span: ?[]const u8 = null }{
        .{ .body = "x := true |> add(1, _)", .code = .T0040, .want = "argument 2 (the piped value): expected int, got bool; 'add' takes (int, int)" },
        .{ .body = "x := true |> add(_, 1)", .code = .T0040, .want = "argument 1 (the piped value): expected int, got bool; 'add' takes (int, int)" },
        .{ .body = "x := 1 |> add(true, _)", .code = .T0040, .want = "argument 1: expected int, got bool; 'add' takes (int, int)" },
        .{ .body = "x := 1 |> add(1, 2, _)", .code = .T0039, .want = "expected 2 argument(s), got 3; 'add' takes (int, int)", .span = "|> add(1, 2, _)" },
        // A `_` first builds the same tree as a prepended value, yet it is written out.
        .{ .body = "x := 1 |> add(_)", .code = .T0039, .want = "expected 2 argument(s), got 1; 'add' takes (int, int)", .span = "|> add(_)" },
        .{ .body = "x := 1 |> add(_, 2, 3)", .code = .T0039, .want = "expected 2 argument(s), got 3; 'add' takes (int, int)", .span = "|> add(_, 2, 3)" },
        .{ .body = "x := 1 |> add(2, 3)", .code = .T0039, .want = "expected 2 argument(s), got 3 (the piped value is argument 1); 'add' takes (int, int)", .span = "|> add(2, 3)" },
        .{ .body = "x: bool = 1 |> add(2, _)", .code = .T0041, .want = "cannot bind int to 'x' of type bool", .span = "1 |> add(2, _)" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa, "fn add(a: int, b: int) -> int {{ return a + b }}\nfn f() {{\n {s}\n return\n}}\n", .{cs.body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer std.debug.print("case: {s}\n", .{cs.body});
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(cs.code, c.result.diags[0].code);
        try testing.expectEqualStrings(cs.want, c.result.diags[0].message);
        if (cs.span) |want_span| {
            const sp = c.result.diags[0].span();
            try testing.expectEqualStrings(want_span, src[sp.start..sp.end]);
        }
    }
}

test "a method pipe's errors carry no piped-value hint" {
    const gpa = testing.allocator;
    const cases = [_]struct { body: []const u8, code: codes.Code, want: []const u8 }{
        .{ .body = "_x := c |> .add(1, 2)", .code = .T0039, .want = "expected 1 argument(s), got 2; 'add' takes (int)" },
        .{ .body = "_x := c |> .add(true)", .code = .T0040, .want = "argument 1: expected int, got bool; 'add' takes (int)" },
        .{ .body = "c |> .nope()", .code = .T0018, .want = "no method 'nope' on type 'C'" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa, "struct C {{ n: int }}\nimpl C {{ fn add(self, k: int) -> int {{ self.n + k }} }}\nfn f() {{\n c := C {{ n: 1 }}\n {s}\n return\n}}\n", .{cs.body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer std.debug.print("case: {s}\n", .{cs.body});
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(cs.code, c.result.diags[0].code);
        try testing.expectEqualStrings(cs.want, c.result.diags[0].message);
    }
}

test "a mut self method pipe needs a place and marks it mutated" {
    const gpa = testing.allocator;
    const pre =
        \\struct C { n: int }
        \\impl C { fn bump(mut self) { self.n = self.n + 1 } }
        \\fn mk() -> C { C { n: 0 } }
        \\
    ;
    const temp = pre ++ "fn main() -> int {\n mk() |> .bump()\n return 0\n}\n";
    try expectOnlyDiag(temp, .T0019);
    try expectOneDiag(temp, .T0019, "cannot call mutating method 'bump' on a temporary; the receiver must be a mutable place (a local, a Vec element, or a field of either)");
    var c = try checkSource(pre ++ "fn main() -> int {\n c := C { n: 0 }\n f := fn() { c |> .bump() }\n f()\n return c.n\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqualSlices(u32, &.{0}, try expectBoxedOnce(&c, 1));
}

test "protocol args on a generic receiver pick its conformance, written or piped" {
    const gpa = testing.allocator;
    const pre =
        \\struct Box[T] { v: T }
        \\protocol Scale[U] { fn scale(self, k: U) -> U }
        \\impl Box[T] has Scale[int] { fn scale(self, k: int) -> int { k * 2 } }
        \\impl Box[T] has Scale[bool] { fn scale(self, k: bool) -> bool { k } }
        \\
    ;
    for ([_][]const u8{
        pre ++ "fn main() -> int {\n b := Box { v: 1 }\n return if b.scale[bool](true) { b |> .scale[int](3) } else { 0 }\n}\n",
        pre ++ "fn main() -> int {\n b := Box { v: 1 }\n return if b |> .scale[bool](true) { b.scale[int](3) } else { 0 }\n}\n",
    }) |src| {
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try expectOwnGenericOk(&c);
        const first = try instanceNameOf(&c, "scale", 0);
        const second = try instanceNameOf(&c, "scale", 1);
        try testing.expect(std.mem.indexOf(u8, first, "Scale$bool") != null);
        try testing.expect(std.mem.indexOf(u8, second, "Scale$int") != null);
    }
    try expectOneDiag(pre ++ "fn main() -> int {\n b := Box { v: 1 }\n return b |> .nope[int](3)\n}\n", .T0018, "no method 'nope' on type 'Box[int]'");
    try expectOneDiag(pre ++ "fn main() -> int {\n b := Box { v: 1 }\n return b.scale[str](3)\n}\n", .T0018, "no method 'scale' on type 'Box[int]' for the given protocol type arguments");
}

test "explicit own type args on a generic receiver pick one instance per call" {
    const gpa = testing.allocator;
    var c = try checkSource(own_box_pre ++
        \\fn main() -> int {
        \\ b := Box { v: 20 }
        \\ return b.tag[bool]() + b.tag[int]() + b.pick[str]("a", "b").len() + Box[int].with[bool](1, true).get()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    // `U` appears in no value param of `tag`, so only the explicit args tell its calls apart.
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "tag", 0), "Box.tag$int$bool"));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "tag", 1), "Box.tag$int$int"));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "pick", 0), "Box.pick$int$str"));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "with", 0), "Box.with$int$bool"));
    try testing.expect((try calleeMethod(&c, "with", 0)).is_static);
}

test "explicit type args a method cannot take are each one error" {
    const pre = own_box_pre ++ "struct K { v: int }\nimpl K { fn make() -> K { K { v: 1 } } }\nfn main() -> int {\n b := Box { v: 2 }\n";
    const cases = [_]struct { body: []const u8, code: codes.Code, want: []const u8 }{
        .{ .body = "return b.tag[int, bool]()", .code = .T0054, .want = "'tag' takes 1 type argument(s) of its own, got 2" },
        .{ .body = "return b.get[int]()", .code = .T0018, .want = "no method 'get' on type 'Box[int]' for the given protocol type arguments" },
        .{ .body = "return b.with[int](1, 2).get()", .code = .T0018, .want = "'with' is an associated function; call it as 'Box[..].with(..)'" },
        .{ .body = "return Box.with[bool](1, true).get()", .code = .T0016, .want = "cannot infer type parameter for 'Box.with'; add explicit type arguments, e.g. Box[int].with" },
        .{ .body = "return Box[int].tag[int]()", .code = .T0018, .want = "'tag' is an instance method, not an associated function; call it on a value" },
        .{ .body = "return K.make[int]().v", .code = .T0054, .want = "'make' is not generic; drop the type arguments" },
        .{ .body = "return b.pick[str](\"a\", 2).len()", .code = .T0040, .want = "argument 2: expected str, got int; 'pick' takes (str, str)" },
    };
    for (cases) |case| {
        const src = try std.fmt.allocPrint(testing.allocator, "{s} {s}\n}}\n", .{ pre, case.body });
        defer testing.allocator.free(src);
        errdefer std.debug.print("case: {s}\n", .{case.body});
        try expectOnlyDiag(src, case.code);
        try expectOneDiag(src, case.code, case.want);
    }
}

test "a method pipe from a type name is one 'not a value' error" {
    const gpa = testing.allocator;
    const cases = [_]struct { body: []const u8, want: []const u8 }{
        .{ .body = "_s := Shape |> .circle(3)", .want = "type 'Shape' is not a value, so it cannot be piped; use 'Shape.circle' directly" },
        .{ .body = "_u := Shape |> .sq", .want = "type 'Shape' is not a value, so it cannot be piped; use 'Shape.sq' directly" },
        .{ .body = "_o: Option[int] = Option |> .some(5)", .want = "type 'Option' is not a value, so it cannot be piped; use 'Option.some' directly" },
        .{ .body = "_c := C |> .make(true)", .want = "type 'C' is not a value, so it cannot be piped; use 'C.make' directly" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa, "enum Shape {{ circle(int), sq }}\nstruct C {{ n: int }}\nimpl C {{ fn make(k: int) -> C {{ C {{ n: k }} }} }}\nstruct Box[T] {{ v: T }}\nimpl Box[T] {{ fn make(v: T) -> Box[T] {{ Box {{ v: v }} }} }}\nfn main() -> int {{\n {s}\n return 0\n}}\n", .{cs.body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer for (c.result.diags) |d| std.debug.print("{s}: got {s}: {s}\n", .{ cs.body, @tagName(d.code), d.message });
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqualStrings(cs.want, c.result.diags[0].message);
    }
    var ok = try checkSource("struct C { n: int }\nimpl C { fn get(self) -> int { self.n } }\nfn main() -> int {\n c := C { n: 1 }\n return c |> .get\n}\n");
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
}

test "call-shape checks: one coded diagnostic per failing call, on every call path" {
    const gpa = testing.allocator;
    const cases = [_]struct { body: []const u8, code: codes.Code, want: []const u8 }{
        // The call's expected type is its result's, so a mismatched count must not also
        // judge a literal arg against it (no extra out-of-range error).
        .{ .body = "_x: int8 = small(300)", .code = .T0039, .want = "expected 2 argument(s), got 1; 'small' takes (int, int)" },
        .{ .body = "_x := b.put(1, 2)", .code = .T0039, .want = "expected 1 argument(s), got 2; 'put' takes (int)" },
        .{ .body = "_x := Box[int].make(true)", .code = .T0040, .want = "argument 1: expected int, got bool; 'make' takes (int)" },
        .{ .body = "_x := 1.eq(true)", .code = .T0040, .want = "argument 1: expected int, got bool; 'eq' takes (int)" },
        .{ .body = "_x := \"a\".len(1)", .code = .T0039, .want = "expected 0 argument(s), got 1; 'len' takes ()" },
        .{ .body = "_x := pair[int](1, true)", .code = .T0040, .want = "argument 2: expected int, got bool; 'pair' takes (int, int)" },
        .{ .body = "_x := b.x |> pair[int](true)", .code = .T0040, .want = "argument 2: expected int, got bool; 'pair' takes (int, int)" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa,
            \\struct Box[T] {{ x: T }}
            \\impl Box[T] {{
            \\ fn put(self, y: T) -> T {{ return y }}
            \\ fn make(x: T) -> Box[T] {{ return Box[T]{{ x: x }} }}
            \\}}
            \\fn small(a: int, b: int) -> int8 {{ return 1 }}
            \\fn pair[T](a: T, b: T) -> T {{ return a }}
            \\fn f() {{
            \\ b := Box[int]{{ x: 1 }}
            \\ {s}
            \\ return
            \\}}
            \\
        , .{cs.body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer std.debug.print("case: {s}\n", .{cs.body});
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(cs.code, c.result.diags[0].code);
        try testing.expectEqualStrings(cs.want, c.result.diags[0].message);
    }
}

test "a non-exhaustive match lists MULTIPLE missing variants in one diagnostic" {
    const gpa = testing.allocator;
    var c = try checkSource(
        "enum Color { Red, Green, Blue }\nfn name(c: Color) -> int {\n match c { .Red -> 0 }\n}\nfn main() -> int { return name(Color.Red) }\n",
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "missing variants 'Green', 'Blue'") != null);
}

test "a non-exhaustive match with ONE missing variant keeps the singular wording" {
    const gpa = testing.allocator;
    var c = try checkSource(
        "enum S { A, B }\nfn f(s: S) -> int {\n match s { .A -> 1 }\n}\nfn main() -> int { return f(S.A) }\n",
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "missing variant 'B'") != null);
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

test ":= from a () call binds (zero-sized)" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
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

test "() local binds (zero-sized)" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
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
    // The diagnostic spans the offending return type.
    try testing.expectEqual(@as(u32, 13), c.result.diags[0].byte_offset);
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

test "an unimported std type in a type-app base position hints the module to import" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f(v: Vec[int]) -> int {\n return 0\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("unknown type 'Vec'; add 'import std/vec'", c.result.diags[0].message);
}

test "an unimported std type as a bare type name hints the module to import" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f(s: Set) -> int {\n return 0\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("unknown type 'Set'; add 'import std/set'", c.result.diags[0].message);
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

test "() parameter type is allowed (zero-sized)" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
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

test "to_string of an int typechecks clean (Display, no arg-type error)" {
    // `int` conforms to the prelude `Display`, so `42.to_string()` is clean (no diagnostic)
    // and derives nothing (int has a prelude Display conformance, so `findConformance` wins
    // over the structural path).
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n s := 42.to_string()\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.derives.len);
}

test "panic with no arguments is an arity error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn main() {\n panic()\n}\n",
    ));
}

test "panic is never: a fn tail, an if or match arm, or a value after it checks clean" {
    const gpa = testing.allocator;
    const bodies = [_][]const u8{
        " panic(\"bad\")",
        " if n > 0 { return n } else { panic(\"negative\") }",
        " if n > 0 { return n }\n panic(\"negative\")",
        " x := if n > 0 { n } else { panic(\"negative\") }\n return x",
        " return if n > 0 { n } else { panic(\"negative\") }",
        " match n { 0 -> 10, _ -> panic(\"bad\"), }",
        " return panic(\"bad\")",
    };
    for (bodies) |body| {
        const src = try std.fmt.allocPrint(gpa, "fn pick(n: int) -> int {{\n{s}\n}}\nfn main() -> int {{\n return pick(3)\n}}\n", .{body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        if (c.result.diags.len != 0) {
            std.debug.print("source:\n{s}\n", .{src});
            for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
            return error.TestUnexpectedResult;
        }
    }
}

test "a bad panic call is still never, so it reports only its own error" {
    try expectOnlyDiag("fn f() -> int {\n panic(1)\n}\nfn main() -> int {\n return f()\n}\n", .T0040);
    // The undeclared `nope` is resolve's error, outside this check; the poisoned call adds none.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount("fn f() -> int {\n panic(nope)\n}\nfn main() -> int {\n return f()\n}\n"));
    try testing.expectEqual(@as(usize, 1), try checkDiagCount("fn f() -> int {\n panic()\n}\nfn main() -> int {\n return f()\n}\n"));
}

test "code after a panic is still unreachable" {
    try expectOnlyDiag("fn f() -> int {\n panic(\"bad\")\n return 0\n}\nfn main() -> int {\n return f()\n}\n", .W0005);
}

test "a -> never fn, method, local fn, lambda or fn value diverges in a value arm" {
    const bodies = [_][]const u8{
        "fn fail(m: str) -> never { panic(m) }\nfn pick(n: int) -> int {\n x := if n > 0 { n } else { fail(\"neg\") }\n x\n}\n",
        "fn fail(m: str) -> never { panic(m) }\nfn pick(n: int) -> int {\n match n { 0 -> 10, _ -> fail(\"bad\"), }\n}\n",
        "fn spin() -> never { loop {} }\nfn wrap() -> never { spin() }\nfn pick(n: int) -> int {\n if n > 0 { return n }\n wrap()\n}\n",
        "struct S { a: int }\nimpl S { fn boom(self) -> never { panic(\"boom\") } }\nfn pick(n: int) -> int {\n if n > 0 { n } else { S { a: n }.boom() }\n}\n",
        "fn fail[T](_x: T) -> never { panic(\"generic\") }\nfn pick(n: int) -> int {\n if n > 0 { n } else { fail(true) }\n}\n",
        "protocol Boom { fn boom(self) -> never }\nstruct S { a: int }\nimpl S has Boom { fn boom(self) -> never { panic(\"b\") } }\nfn pick(n: int) -> int {\n if n > 0 { n } else { S { a: n }.boom() }\n}\n",
        "fn call(n: int, g: fn(str) -> never) -> int {\n if n > 0 { n } else { g(\"neg\") }\n}\nfn pick(n: int) -> int {\n fn local(m: str) -> never { panic(m) }\n lam := fn(m: str) -> never { panic(m) }\n call(n, local) + call(n, lam)\n}\n",
    };
    for (bodies) |body| {
        const src = try std.fmt.allocPrint(testing.allocator, "{s}fn main() -> int {{\n return pick(3)\n}}\n", .{body});
        defer testing.allocator.free(src);
        try expectDiagCodes(src, &.{});
    }
}

test "a -> never fn that can return is an error" {
    try expectDiagCodes("fn bad(x: int) -> never {\n if x > 0 { panic(\"p\") }\n}\nfn main() {\n bad(1)\n}\n", &.{.T0043});
    try expectDiagCodes("fn bad(x: int) -> never {\n if x > 0 { return }\n panic(\"p\")\n}\nfn main() {\n bad(1)\n}\n", &.{.T0042});
    try expectDiagCodes("fn bad() -> never { 1 }\nfn main() {\n bad()\n}\n", &.{.T0042});
    try expectDiagCodes("fn main() {\n fn local(n: int) -> never { if n > 0 { panic(\"x\") } }\n local(1)\n}\n", &.{.T0043});
    try expectDiagCodes("fn main() {\n lam := fn(n: int) -> never { if n > 0 { panic(\"x\") } }\n lam(1)\n}\n", &.{.T0043});
}

test "never is spellable only as a return type" {
    try expectDiagCodes("fn f(_x: never) {}\nfn main() {}\n", &.{.T0001});
    try expectDiagCodes("struct S { a: never }\nfn main() {}\n", &.{.T0001});
    try expectDiagCodes("fn main() {\n _x: never = panic(\"p\")\n}\n", &.{.T0001});
}

test "a user type named never shadows the builtin in every position" {
    try expectDiagCodes("struct never { a: int }\nfn mk() -> never { never { a: 3 } }\nfn take(x: never) -> int { x.a }\nfn main() -> int {\n return take(mk())\n}\n", &.{});
    try expectDiagCodes("type never = int\nfn mk() -> never { 3 }\nfn main() -> int {\n return mk()\n}\n", &.{});
}

test "an extern and the seeded panic both return never" {
    const gpa = testing.allocator;
    var c = try checkSourceAs(
        \\extern fn abort() -> never
        \\fn pick(n: int) -> int {
        \\ if n > 0 { n } else { abort() }
        \\}
        \\fn main() -> int {
        \\ return pick(3)
        \\}
        \\
    , .core);
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.resolve.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    for ([_][]const u8{ "abort", "panic" }) |name| {
        const id = for (c.resolve.fns, 0..) |f, i| {
            if (std.mem.endsWith(u8, f.name, name)) break i;
        } else return error.TestExpectedFn;
        try testing.expectEqual(Kind.never, c.result.sigs[id].ret.kind);
    }
}

test "a user fn named panic keeps its declared type" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount("fn panic(_s: str) -> int { 7 }\nfn main() -> int {\n return panic(\"x\")\n}\n"));
    try expectOnlyDiag("fn panic(_s: str) -> int { 7 }\nfn g() -> bool {\n return panic(\"x\")\n}\nfn main() {\n g()\n}\n", .T0042);
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
        "fn f() -> int {\n i := 0\n loop {\n if i >= 5 { break i } else { break i > 0 }\n }\n}\n",
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
        "fn f() {\n break\n}\n",
    ));
}

test "continue outside a loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n continue\n}\n",
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

test "builtinConvMethod int↔float verdicts" {
    // int -> float (into): signed or sub-64-bit unsigned -> lossless scvtf.
    try testing.expectEqual(ConvKind.int_to_float, builtinConvMethod(Type.int, Type.float, "into", null).?.kind);
    try testing.expectEqual(Type.float, builtinConvMethod(Type.int, Type.float, "into", null).?.target);
    try testing.expectEqual(ConvKind.int_to_float, builtinConvMethod(Type.uint8, Type.float, "into", null).?.kind);
    // A 64-bit unsigned (>= 2^63 would scvtf negative) is deferred (T0018), not miscompiled.
    try testing.expect(builtinConvMethod(Type.uint, Type.float, "into", null) == null);
    // float -> int (try_into): target restricted to plain `int`; a narrower/unsigned target defers.
    try testing.expectEqual(ConvKind.float_to_int, builtinConvMethod(Type.float, Type.int, "try_into", null).?.kind);
    try testing.expectEqual(Type.int, builtinConvMethod(Type.float, Type.int, "try_into", null).?.target);
    try testing.expect(builtinConvMethod(Type.float, Type.uint8, "try_into", null) == null);
    // float.into() to int and int.try_into() to float are NOT this surface.
    try testing.expect(builtinConvMethod(Type.float, Type.int, "into", null) == null);
    try testing.expect(builtinConvMethod(Type.int, Type.float, "try_into", null) == null);
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

test "chained `wide.try_into().expect(\"no value\")` leaks the int target to try_into" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\  wide: uint = 200
        \\  back: uint8 = wide.try_into().expect("no value")
        \\  return if back == 200 { 42 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);

    var saw_try_into = false;
    var saw_expect = false;
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
        } else if (std.mem.eql(u8, member, "expect")) {
            saw_expect = true;
            try testing.expectEqual(Type.uint8, ty);
        }
    }
    try testing.expect(saw_try_into and saw_expect);
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

test "declaration and instantiation diagnostics span their subject" {
    try expectSpans(&.{
        .{ .src = "struct A { x: int }\nfn g(_a: A[int]) -> int { return 0 }\n", .key = "is not generic", .want = "A[int]" },
        .{ .src = "struct Box[T] { v: T }\nfn g(_b: Box[int, bool]) -> int { return 0 }\n", .key = "expects 1 type argument", .want = "[int, bool]" },
        .{ .src = "struct P { x: int }\npub fn mk(p: P) -> int { return p.x }\n", .key = "exposes non-pub", .want = "P" },
        .{ .src = "struct P { x: int }\npub struct S { pub f: P }\n", .key = "exposes non-pub", .want = "P" },
        .{ .src = "struct P { x: int }\npub enum E { v(int, P) }\n", .key = "exposes non-pub", .want = "P" },
        .{ .src = "fn main() -> bool { return true }\n", .key = "main must return", .want = "bool" },
        .{ .src = "protocol Show { fn show(self) -> int }\nfn h[T has Show](_t: T) -> int { return 0 }\nfn main() -> int { return h(3) }\n", .key = "does not conform", .want = "h(3)" },
        .{ .src = "protocol Conv[U] { fn conv(self) -> U }\nstruct Box[T] { v: T }\nfn h[T has Conv[Box[int]]](_t: T) -> int { return 0 }\n", .key = "bound argument must be", .want = "Box[int]" },
        .{ .src = "struct Box[T] { v: T }\nimpl Box[int] { fn get(self) -> int { return 1 } }\n", .key = "concrete type instance", .want = "Box[int]" },
    });
}

fn isCallMember(tree: Ast.Tree, fa: Ast.Index) bool {
    for (tree.nodes) |n| {
        if (n.tag != .call) continue;
        if (Callee.memberNode(tree, n)) |m| if (m == fa) return true;
    }
    return false;
}

/// The `nth` (node order) method-call member node spelled `member`.
fn memberSlot(c: *const Checked, member: []const u8, nth: usize) !Ast.Index {
    var seen: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .field_access) continue;
        if (!std.mem.eql(u8, c.tokens[n.main_token].text(c.source), member)) continue;
        const idx = Ast.Index.from(@intCast(i));
        if (!isCallMember(c.tree, idx)) continue;
        if (seen == nth) return idx;
        seen += 1;
    }
    return error.TestMemberNotFound;
}

fn calleeView(c: *const Checked, member: []const u8, nth: usize) !Callee.View {
    return c.result.callees[0][(try memberSlot(c, member, nth)).int()].view();
}

fn calleeMethod(c: *const Checked, member: []const u8, nth: usize) !Method {
    return switch (try calleeView(c, member, nth)) {
        .method => |i| c.result.methods[i],
        else => error.TestExpectedMethodCallee,
    };
}

fn expectNoPending(r: *const GraphResult) !void {
    for (r.callees) |cs| for (cs) |x| try testing.expect(!x.isPending());
    for (r.instances) |inst| for (inst.callees) |x| try testing.expect(!x.isPending());
}

test "pickMethodIndex selects what resolveConformanceMethod returns" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Into[U] { fn into(self) -> U }
        \\protocol Tag { fn tag(self) -> int }
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, a: int) -> int { self.n + a } }
        \\impl Acc has Into[int] { fn into(self) -> int { self.n } }
        \\impl Acc has Into[bool] { fn into(self) -> bool { self.n > 0 } }
        \\impl Acc has Tag { fn tag(self) -> int { self.n } }
        \\fn main() -> int { return 0 }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const ms = c.result.methods;
    const plus = findMethodIndexNamed(ms, "plus").?;
    const tag = findMethodIndexNamed(ms, "tag").?;
    const into = findMethodIndexNamed(ms, "into").?;
    const acc = ms[plus].recv;
    const tag_pid = ms[tag].protocol_id.?;
    const into_pid = ms[into].protocol_id.?;

    const Case = struct { name: []const u8, pid: ?u32 = null, args: ?[]const Type = null, want: std.meta.Tag(MethodIndexPick) };
    const cases = [_]Case{
        .{ .name = "plus", .want = .one },
        .{ .name = "tag", .want = .one },
        .{ .name = "into", .want = .ambiguous },
        .{ .name = "into", .args = &.{Type.int}, .want = .one },
        .{ .name = "into", .args = &.{Type.bool}, .want = .one },
        .{ .name = "nope", .want = .none },
        .{ .name = "tag", .pid = tag_pid, .want = .one },
        .{ .name = "tag", .pid = into_pid, .want = .none },
        .{ .name = "plus", .pid = tag_pid, .want = .none },
    };
    for (cases) |k| {
        const idx = pickMethodIndex(ms, acc, k.name, k.pid, k.args, .program);
        const whole = resolveConformanceMethod(ms, acc, k.name, k.pid, k.args);
        try testing.expectEqual(k.want, std.meta.activeTag(idx));
        switch (idx) {
            .none => try testing.expect(whole == .none),
            .ambiguous => try testing.expect(whole == .ambiguous),
            .one => |i| {
                try testing.expect(std.meta.eql(ms[i], whole.one));
                try testing.expectEqualStrings(k.name, ms[i].name);
                if (k.args) |a| try testing.expect(eqlTypeVec(ms[i].protocol_args, a));
            },
        }
    }
}

fn findMethodIndexNamed(ms: []const Method, name: []const u8) ?u32 {
    for (ms, 0..) |m, i| if (std.mem.eql(u8, m.name, name)) return @intCast(i);
    return null;
}

/// Every source-method callee shape in one program; the body of
/// tests/corpora/language-features/methods/dispatch_shapes.toy, shared by the callee tests
/// here and in types_graph.zig.
pub const dispatch_shapes_src =
    \\protocol Doubler {
    \\    fn dbl(self) -> int
    \\}
    \\protocol Into[U] {
    \\    fn into(self) -> U
    \\}
    \\protocol Twice {
    \\    fn twice(self) -> int
    \\}
    \\struct Acc { n: int }
    \\impl Acc {
    \\    fn plus(self, a: int, b: int) -> int { self.n + a + b }
    \\    fn bump(mut self) { self.n = self.n + 1 }
    \\    fn shade(self) -> Shade { if self.n > 1 { Shade.dark } else { Shade.light } }
    \\}
    \\impl Acc has Doubler {
    \\    fn dbl(self) -> int { self.n * 2 }
    \\}
    \\impl Acc has Into[int] {
    \\    fn into(self) -> int { self.n }
    \\}
    \\impl Acc has Into[bool] {
    \\    fn into(self) -> bool { self.n > 0 }
    \\}
    \\impl int has Twice {
    \\    fn twice(self) -> int { self * 2 }
    \\}
    \\struct Box[T] { v: T }
    \\impl Box[T] {
    \\    fn make(v: T) -> Box[T] { Box[T] { v: v } }
    \\    fn get(self) -> T { self.v }
    \\}
    \\struct Key { a: int, b: bool }
    \\enum Shade { light, dark }
    \\fn twice[T has Doubler](v: T) -> int { v.dbl() + v.dbl() }
    \\fn main() -> int {
    \\    acc := Acc { n: 1 }
    \\    acc.bump()
    \\    if acc.plus(1, 2) != 5 { return 1 }
    \\    piped := 1 |> acc.plus(2)
    \\    if piped != 5 { return 2 }
    \\    if acc.into[int]() != 2 { return 3 }
    \\    if !acc.into[bool]() { return 4 }
    \\    if twice(acc) != 8 { return 5 }
    \\    if (3).twice() != 6 { return 6 }
    \\    b := Box[int].make(7)
    \\    if b.get() != 7 { return 7 }
    \\    k := Key { a: 1, b: true }
    \\    if k.hash() == 0 { return 8 }
    \\    if k.to_string().len() == 0 { return 9 }
    \\    if Shade.dark.to_string().len() == 0 { return 10 }
    \\    lit := match acc.shade() { .light -> 0, .dark -> 1 }
    \\    if lit != 1 { return 14 }
    \\    o: Option[int] = Option.some(1)
    \\    if o.to_string().len() == 0 { return 11 }
    \\    c := 'x'
    \\    cp: int = c.into()
    \\    if cp != 120 { return 12 }
    \\    if c.to_string().len() != 1 { return 13 }
    \\    return 42
    \\}
    \\
;

test "every dispatch branch records its callee" {
    const gpa = testing.allocator;
    {
        var c = try checkSource(dispatch_shapes_src);
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);

        for ([_]usize{ 0, 1 }) |nth| {
            const m = try calleeMethod(&c, "plus", nth);
            try testing.expectEqualStrings("plus", m.name);
            try testing.expect(m.instance == null);
        }
        try testing.expect((try calleeMethod(&c, "bump", 0)).mut_self);
        const into_int = try calleeMethod(&c, "into", 0);
        try testing.expect(eqlTypeVec(into_int.protocol_args, &.{Type.int}));
        const into_bool = try calleeMethod(&c, "into", 1);
        try testing.expect(eqlTypeVec(into_bool.protocol_args, &.{Type.bool}));
        try testing.expectEqualStrings("twice", (try calleeMethod(&c, "twice", 0)).name);
        try testing.expectEqualStrings("shade", (try calleeMethod(&c, "shade", 0)).name);
        try testing.expect((try calleeMethod(&c, "get", 0)).instance != null);
        const make = try calleeMethod(&c, "make", 0);
        try testing.expect(make.is_static and make.instance != null);
        try testing.expect((try calleeMethod(&c, "hash", 0)).derive != null);
        // k, Shade.dark, Option[int], char
        for (0..4) |nth| try testing.expectEqual(Callee.View{ .builtin = .display_to_string }, try calleeView(&c, "to_string", nth));
        try testing.expectEqual(Callee.View{ .builtin = .conv_into }, try calleeView(&c, "into", 2));
        for (0..4) |nth| try testing.expectEqual(Callee.View{ .builtin = .str_len }, try calleeView(&c, "len", nth));
        try expectNoPending(&c.result);
    }
    {
        var c = try checkSource(
            \\fn main() -> int {
            \\    s := "abc"
            \\    t := s.concat("de")
            \\    if t.len() != 5 { return 1 }
            \\    if t.byte_at(1) != 98 { return 2 }
            \\    if !s.eq("abc") { return 3 }
            \\    if !(7).eq(7) { return 4 }
            \\    if !true.eq(true) { return 5 }
            \\    if !().eq(()) { return 6 }
            \\    h := s.hash() + (3).hash() + false.hash() + ().hash()
            \\    if h == 0 { return 7 }
            \\    if (12).to_string().len() != 2 { return 8 }
            \\    if true.to_string().len() != 4 { return 9 }
            \\    if s.to_string().len() != 3 { return 10 }
            \\    if ().to_string().len() == 0 { return 11 }
            \\    some: Option[int] = Option.some(40)
            \\    none: Option[int] = Option.none
            \\    if !some.is_some() { return 12 }
            \\    if !none.is_none() { return 13 }
            \\    ok: Result[int, bool] = Result.ok(1)
            \\    err: Result[int, bool] = Result.err(false)
            \\    if !ok.is_ok() { return 14 }
            \\    if !err.is_error() { return 15 }
            \\    small: uint8 = 5
            \\    w: uint = small.into()
            \\    big: uint = 300
            \\    back: uint8 = big.try_into().or(0)
            \\    if back != 0 { return 16 }
            \\    if w != 5 { return 17 }
            \\    return some.expect("no value") + none.or(2)
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);
        const Want = struct { member: []const u8, count: usize, k: Callee.Builtin };
        const wants = [_]Want{
            .{ .member = "concat", .count = 1, .k = .str_concat },
            .{ .member = "len", .count = 5, .k = .str_len },
            .{ .member = "byte_at", .count = 1, .k = .str_byte_at },
            .{ .member = "eq", .count = 4, .k = .scalar_eq },
            .{ .member = "hash", .count = 4, .k = .scalar_hash },
            .{ .member = "to_string", .count = 4, .k = .scalar_to_string },
            .{ .member = "is_some", .count = 1, .k = .opt_is_tag0 },
            .{ .member = "is_ok", .count = 1, .k = .opt_is_tag0 },
            .{ .member = "is_none", .count = 1, .k = .opt_is_tag1 },
            .{ .member = "is_error", .count = 1, .k = .opt_is_tag1 },
            .{ .member = "into", .count = 1, .k = .conv_into },
            .{ .member = "try_into", .count = 1, .k = .conv_try_into },
            .{ .member = "expect", .count = 1, .k = .opt_expect },
            .{ .member = "or", .count = 2, .k = .opt_or },
        };
        for (wants) |w| {
            for (0..w.count) |nth| try testing.expectEqual(Callee.View{ .builtin = w.k }, try calleeView(&c, w.member, nth));
            try testing.expectError(error.TestMemberNotFound, memberSlot(&c, w.member, w.count));
        }
        try expectNoPending(&c.result);
    }
}

test "a direct cmp: a builtin on an Ord scalar, the derived witness on an aggregate" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int }
        \\fn main() -> int {
        \\    a := (1).cmp(2)
        \\    b := "a".cmp("b")
        \\    c := true.cmp(false)
        \\    w: uint8 = 3
        \\    d := w.cmp(4)
        \\    e := P { x: 1 }.cmp(P { x: 2 })
        \\    return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    for (0..4) |nth| try testing.expectEqual(Callee.View{ .builtin = .scalar_cmp }, try calleeView(&c, "cmp", nth));
    try testing.expect((try calleeMethod(&c, "cmp", 4)).derive != null);
    try expectNoPending(&c.result);
}

test "template spans: the module slice holds the abstract pick, each instance slice its own" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\protocol Doubler { fn dbl(self) -> int }
        \\struct Acc { n: int }
        \\impl Acc has Doubler { fn dbl(self) -> int { self.n * 2 } }
        \\struct Pt { x: int, y: int }
        \\fn hashit[T](x: T) -> int { x.hash() }
        \\fn twice[T has Doubler](v: T) -> int { v.dbl() + v.dbl() }
        \\fn main() -> int { hashit(Pt { x: 1, y: 2 }) + hashit(7) + twice(Acc { n: 1 }) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);

    const hash_fa = (try memberSlot(&c, "hash", 0)).int();
    try testing.expectEqual(Callee.View.none, c.result.callees[0][hash_fa].view());
    const dbl_fa = (try memberSlot(&c, "dbl", 0)).int();
    const abstract = switch (c.result.callees[0][dbl_fa].view()) {
        .protocol => |p| p,
        else => return error.TestExpectedProtocolCallee,
    };
    try testing.expectEqual(@as(u16, 0), abstract.method);

    var saw_int = false;
    var saw_pt = false;
    var saw_acc = false;
    for (c.result.instances) |inst| {
        const name = c.resolve.fns[inst.template_gid].name;
        if (std.mem.endsWith(u8, name, "hashit")) {
            const v = inst.nodes().callee(Ast.Index.from(hash_fa)).view();
            if (inst.args[0].kind == .int) {
                try testing.expectEqual(Callee.View{ .builtin = .scalar_hash }, v);
                saw_int = true;
            } else {
                try testing.expect(c.result.methods[v.method].derive != null);
                saw_pt = true;
            }
        } else if (std.mem.endsWith(u8, name, "twice")) {
            const mi = inst.nodes().callee(Ast.Index.from(dbl_fa)).view().method;
            try testing.expectEqual(@as(?u32, abstract.protocol), c.result.methods[mi].protocol_id);
            saw_acc = true;
        }
    }
    try testing.expect(saw_int and saw_pt and saw_acc);
    try expectNoPending(&c.result);
}

test "argument errors keep the selection; selection failures leave none" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, a: int) -> int { self.n + a } }
        \\struct Key { a: int }
        \\fn main() -> int {
        \\  acc := Acc { n: 1 }
        \\  a := acc.plus(true)
        \\  b := acc.nope()
        \\  c := "s".len(1)
        \\  k := Key { a: 1 }
        \\  d := k.hash(1)
        \\  e := k.to_string(1)
        \\  return a + b + c + d + e.len()
        \\}
        \\
    );
    defer c.deinit(gpa);
    var t0018 = false;
    for (c.result.diags) |d| {
        if (d.code == .T0018) t0018 = true;
    }
    try testing.expect(t0018);
    try testing.expect(c.result.diags.len >= 5);
    try testing.expectEqualStrings("plus", (try calleeMethod(&c, "plus", 0)).name);
    try testing.expectEqual(Callee.View.none, try calleeView(&c, "nope", 0));
    try testing.expectEqual(Callee.View{ .builtin = .str_len }, try calleeView(&c, "len", 0));
    try testing.expect((try calleeMethod(&c, "hash", 0)).derive != null);
    try testing.expectEqual(Callee.View{ .builtin = .display_to_string }, try calleeView(&c, "to_string", 0));
    try expectNoPending(&c.result);
}

/// The `nth` (node order) `.lambda` node.
fn nthLambda(c: *const Checked, nth: usize) !Ast.Index {
    var seen: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .lambda) continue;
        if (seen == nth) return Ast.Index.from(@intCast(i));
        seen += 1;
    }
    return error.TestLambdaNotFound;
}

/// The type the checker gave param `pi` of lambda `lam`.
fn lambdaParamType(c: *const Checked, lam: Ast.Index, pi: usize) Type {
    const proto = Ast.protoAt(c.tree, c.tree.nodes[lam.int()].lhs.int());
    return c.result.node_types[0][proto.params[pi].int()];
}

/// The reified signature of lambda `lam`'s value.
fn lambdaSig(c: *const Checked, lam: Ast.Index) LayoutEngine.FnSig {
    const t = c.result.node_types[0][lam.int()];
    std.debug.assert(LayoutEngine.isFnStruct(t, c.result.layouts));
    return c.result.layouts[t.struct_id].fn_sig.?;
}

test "a lambda takes its param types from the expected fn type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct H { run: fn(int) -> int }
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\ fn with(self, v: T) -> Box[T] { Box { v: v } }
        \\}
        \\fn apply(f: fn(int) -> int) -> int { return f(1) }
        \\fn ret() -> fn(bool) -> bool { return fn(b) { !b } }
        \\fn tail() -> fn(str) -> int { fn(s) { s.len() } }
        \\fn main() -> int {
        \\ f: fn(int) -> int = fn(x) { x + 1 }
        \\ n := apply(fn(y) { y * 2 })
        \\ h := H { run: fn(z) { z } }
        \\ b := Box { v: f }
        \\ b2 := b.with(fn(w) { w - 1 })
        \\ if ret()(true) { return 0 }
        \\ return f(n) + h.run(1) + b2.v(3) + tail()("ab")
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const want = [_]Type{ Type.bool, Type.str, Type.int, Type.int, Type.int, Type.int };
    for (want, 0..) |w, i| {
        const lam = try nthLambda(&c, i);
        try testing.expect(Type.eql(w, lambdaParamType(&c, lam, 0)));
        // The expected fn type's return is the lambda's too.
        const sig = lambdaSig(&c, lam);
        try testing.expect(Type.eql(if (i == 0) Type.bool else Type.int, sig.ret));
    }
}

test "an unannotated param with no expected type is T0050 (incl. a generic call arg)" {
    const gpa = testing.allocator;
    const srcs = [_][]const u8{
        "fn main() -> int {\n f := fn(x) { x }\n return f(1)\n}\n",
        "fn apply1[T](f: fn(T) -> T) -> int { 0 }\nfn main() -> int { return apply1(fn(x) { x }) }\n",
        "fn main() -> int { return fn(x) { x }(1) }\n",
    };
    for (srcs) |src| {
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try testing.expect(hasCode(&c, .T0050));
        for (c.result.diags) |d| if (d.code == .T0050) try testing.expect(std.mem.indexOf(u8, d.message, "'x'") != null);
    }
    // An annotated param needs nothing from the context.
    var ok = try checkSource("fn apply[T](f: fn(T) -> T, x: T) -> T { f(x) }\nfn main() -> int { return apply(fn(x: int) { x * 2 }, 21) }\n");
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
    // Another arg of the generic call binds what the lambda's param needs.
    var bound = try checkSource("fn apply[T](f: fn(T) -> T, x: T) -> T { f(x) }\nfn main() -> int { return apply(fn(x) { x }, 1) }\n");
    defer bound.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), bound.result.diags.len);
}

test "T0050 names a generic call only when one expected the lambda" {
    const gpa = testing.allocator;
    const cases = [_]struct { src: []const u8, generic: bool }{
        .{ .src = "fn main() -> int {\n f := fn(x) { x }\n return f(1)\n}\n", .generic = false },
        .{ .src = "fn apply1[T](f: fn(T) -> T) -> int { 0 }\nfn main() -> int { return apply1(fn(x) { x }) }\n", .generic = true },
    };
    for (cases) |cs| {
        var c = try checkSource(cs.src);
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0050, c.result.diags[0].code);
        try testing.expectEqual(cs.generic, std.mem.indexOf(u8, c.result.diags[0].message, "generic call") != null);
    }
}

test "a '_' capture takes its param type from the position of its '_'" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn put(mut self, v: T) { self.v = v } }
        \\enum Shape { circle(int) }
        \\struct Pt(int, int)
        \\fn add(a: int, b: int) -> int { a + b }
        \\fn pick(a: int, _b: bool) -> int { a }
        \\fn main() -> int {
        \\ inc := add(1, _)
        \\ acc := Acc { n: 1 }
        \\ p := acc.plus(_)
        \\ g: fn(int, bool) -> int = pick
        \\ h := g(1, _)
        \\ b := Box[str] { v: "a" }
        \\ put := b.put(_)
        \\ put("b")
        \\ circle := Shape.circle(_)
        \\ _s := circle(1)
        \\ at := Pt(_, 4)
        \\ _q := at(1)
        \\ o: Option[bool] = Option.none
        \\ d := o.or(_)
        \\ _r := d(true)
        \\ return inc(1) + p(1) + h(true)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const want = [_]Type{ Type.int, Type.int, Type.bool, Type.str, Type.int, Type.int, Type.bool };
    for (want, 0..) |w, i| {
        const lam = try nthLambda(&c, i);
        try testing.expect(Ast.isCaptureLambda(c.tree, c.tokens, lam));
        try testing.expect(Type.eql(w, lambdaParamType(&c, lam, 0)));
        try testing.expect(Type.eql(w, lambdaSig(&c, lam).params[0]));
    }
    try testing.expect(Type.eql(Type.int, lambdaSig(&c, try nthLambda(&c, 0)).ret));
}

test "an expected fn type gives a capture its param type before the call does" {
    try expectOnlyDiag("fn add(a: int, b: int) -> int { a + b }\nfn main() -> int {\n h: fn(int8) -> int = add(1, _)\n return h(1)\n}\n", .T0040);
    try expectOneDiag("fn add(a: int, b: int) -> int { a + b }\nfn main() -> int {\n h: fn(int8) -> int = add(1, _)\n return h(1)\n}\n", .T0040, "argument 2: expected int, got int8; 'add' takes (int, int)");
}

test "a capture in a bare generic call is typed by its own call; a written lambda takes the type the other args bind" {
    const gpa = testing.allocator;
    const pre = "fn add(a: int, b: int) -> int { a + b }\nfn twice[T](f: fn(T) -> T, x: T) -> T { f(f(x)) }\n";
    var ok = try checkSource(pre ++ "fn main() -> int { return twice(add(1, _), 5) }\n");
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
    var lam = try checkSource(pre ++ "fn main() -> int { return twice(fn(v) { add(1, v) }, 5) }\n");
    defer lam.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lam.result.diags.len);
    try testing.expect(Type.eql(Type.int, lambdaParamType(&lam, try nthLambda(&lam, 0), 0)));
}

test "a capture whose '_' no position types is one diagnostic" {
    const gpa = testing.allocator;
    const pre = "fn pair[T](a: T, _b: T) -> T { a }\nfn id[T](a: T) -> T { a }\nenum Two[T] { both(T, T) }\n";
    const cases = [_]struct { body: []const u8, codes: []const codes.Code }{
        .{ .body = "_q := pair(1, _)", .codes = &.{} },
        .{ .body = "_s := pair(_, 1)", .codes = &.{} },
        .{ .body = "_r := id(_)", .codes = &.{.T0016} },
        .{ .body = "_k := pair[int](1, _)", .codes = &.{} },
        .{ .body = "_p := pair(if true { 1 } else { 2 }, _)", .codes = &.{.W0007} },
        // A warning in the body does not stand in for the T0050: it would leave the
        // capture poisoned with no error.
        .{ .body = "_t := Two.both(if true { 1 } else { 2 }, _)", .codes = &.{ .W0007, .T0050 } },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa, "{s}fn main() -> int {{\n {s}\n return 0\n}}\n", .{ pre, cs.body });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer for (c.result.diags) |d| std.debug.print("{s}: got {s}: {s}\n", .{ cs.body, @tagName(d.code), d.message });
        try testing.expectEqual(cs.codes.len, c.result.diags.len);
        for (cs.codes) |code| try testing.expect(hasCode(&c, code));
        for (c.result.diags) |d| if (d.code == .T0050)
            try testing.expect(std.mem.startsWith(u8, d.message, "cannot infer the type of '_'"));
    }
}

const own_box_pre =
    \\struct Box[T] { v: T }
    \\impl Box[T] {
    \\ fn map[U](self, f: fn(T) -> U) -> Box[U] { Box { v: f(self.v) } }
    \\ fn fold[A](self, init: A, f: fn(A, T) -> A) -> A { f(init, self.v) }
    \\ fn tag[U](self) -> int { 0 }
    \\ fn pick[U](self, a: U, _b: U) -> U { a }
    \\ fn with[U](v: T, _tag: U) -> Box[T] { Box { v: v } }
    \\ fn get(self) -> T { self.v }
    \\}
    \\
;

fn expectOwnGenericOk(c: *const Checked) !void {
    for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try expectNoPending(&c.result);
}

fn instanceNameOf(c: *const Checked, member: []const u8, nth: usize) ![]const u8 {
    const m = try calleeMethod(c, member, nth);
    return c.result.instances[m.instance orelse return error.TestExpectedInstanceRow].name.?;
}

test "a method's own type param binds from a lambda's return" {
    const gpa = testing.allocator;
    var c = try checkSource(own_box_pre ++
        \\fn main() -> int {
        \\ b := Box { v: 20 }
        \\ c := b.map(fn(x) { x > 1 })
        \\ return if c.get() { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    try testing.expect(Type.eql(Type.int, lambdaParamType(&c, try nthLambda(&c, 0), 0)));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "map", 0), "Box.map$int$bool"));
}

test "two instances of one own-generic method on one receiver seal to their own rows" {
    const gpa = testing.allocator;
    var c = try checkSource(own_box_pre ++
        \\fn main() -> int {
        \\ b := Box { v: 20 }
        \\ c := b.map(fn(x) { x + 1 })
        \\ d := b.map(fn(_x) { "s" })
        \\ return c.get() + d.get().len()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    const first = try calleeView(&c, "map", 0);
    const second = try calleeView(&c, "map", 1);
    try testing.expect(first.method != second.method);
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "map", 0), "Box.map$int$int"));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "map", 1), "Box.map$int$str"));
}

test "an own type param fixed by a non-lambda arg types the lambda" {
    const gpa = testing.allocator;
    var c = try checkSource(own_box_pre ++
        \\fn main() -> int {
        \\ b := Box { v: 20 }
        \\ return b.fold(0, fn(acc, x) { acc + x })
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    const lam = try nthLambda(&c, 0);
    try testing.expect(Type.eql(Type.int, lambdaParamType(&c, lam, 0)));
    try testing.expect(Type.eql(Type.int, lambdaParamType(&c, lam, 1)));
}

test "a later lambda sees what an earlier lambda bound" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn g[T, U](f: fn(int) -> T, h: fn(T) -> U) -> U { h(f(1)) }
        \\fn main() -> int {
        \\ return if g(fn(x) { x + 1 }, fn(y) { y > 1 }) { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    try testing.expect(Type.eql(Type.int, lambdaParamType(&c, try nthLambda(&c, 1), 0)));
}

test "a lambda in a bare generic call takes T from a later or an earlier arg" {
    const gpa = testing.allocator;
    const pre = "fn apply[T](f: fn(T) -> T, x: T) -> T { f(x) }\nfn apply2[T](x: T, f: fn(T) -> T) -> T { f(x) }\n";
    for ([_][]const u8{ "apply(fn(x) { x * 7 }, 1)", "apply2(1, fn(x) { x * 7 })" }) |call| {
        const src = try std.fmt.allocPrint(gpa, "{s}fn main() -> int {{ return {s} }}\n", .{ pre, call });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try expectOwnGenericOk(&c);
        try testing.expect(Type.eql(Type.int, lambdaParamType(&c, try nthLambda(&c, 0), 0)));
    }
}

test "a call's expected fn type still types a lambda no other arg fixes" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn id[T](a: T) -> T { a }
        \\fn main() -> int {
        \\ g: fn(int) -> int = id(fn(x) { x + 1 })
        \\ return g(1)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    try testing.expect(Type.eql(Type.int, lambdaParamType(&c, try nthLambda(&c, 0), 0)));
}

test "a method's own type param no arg fixes is T0016, and explicit args fix it" {
    const gpa = testing.allocator;
    const src = own_box_pre ++ "fn main() -> int {\n b := Box { v: 20 }\n return b.tag()\n}\n";
    try expectOnlyDiag(src, .T0016);
    try expectOneDiag(src, .T0016, "cannot infer type parameter 'U' of method 'tag' from its arguments; add explicit type arguments, e.g. tag[int](..)");
    var c = try checkSource(own_box_pre ++ "fn main() -> int {\n b := Box { v: 20 }\n return b.tag[int]()\n}\n");
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
}

test "conflicting own-param args are one T0015 naming both" {
    const gpa = testing.allocator;
    const src = own_box_pre ++ "fn main() -> int {\n b := Box { v: 20 }\n return b.pick(1, true)\n}\n";
    try expectOnlyDiag(src, .T0015);
    var c = try checkSource(src);
    defer c.deinit(gpa);
    const d = c.result.diags[0];
    const sp = d.span();
    try testing.expectEqualStrings("true", src[sp.start..sp.end]);
    const rel = d.relatedSpan().?;
    try testing.expectEqualStrings("1", src[rel.start..rel.end]);
}

test "a pass-1 conflict types the waiting lambda once" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn tri[T](a: T, b: T, f: fn(T) -> T) -> T { f(a) }
        \\fn main() -> int {
        \\ r := tri(1, true, fn(x) { x })
        \\ return r
        \\}
        \\
    );
    defer c.deinit(gpa);
    errdefer for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    try testing.expectEqual(@as(usize, 2), c.result.diags.len);
    try testing.expect(hasCode(&c, .T0015));
    try testing.expect(hasCode(&c, .T0050));
}

test "own type params on a non-generic struct make a template with one row per instance" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { v: int }
        \\impl P {
        \\ fn conv[U](self, f: fn(int) -> U) -> U { f(self.v) }
        \\ fn with[U](v: int, _u: U) -> P { P { v: v } }
        \\}
        \\fn main() -> int {
        \\ p := P.with(4, true)
        \\ n := p.conv(fn(x) { x * 3 })
        \\ b := p.conv[bool](fn(x) { x > 3 })
        \\ return if b { n + P.with[str](1, "s").v } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    const a = try calleeMethod(&c, "conv", 0);
    const b = try calleeMethod(&c, "conv", 1);
    const tm = findGenericMethod(c.result.templates, a.recv.nominalId(), false, "conv", .program) orelse return error.TestUnexpectedResult;
    try testing.expect(tm.has_self and !tm.recv_is_enum);
    try testing.expect(Type.eql(a.recv, b.recv) and a.recv.kind == .@"struct");
    try testing.expect(a.instance.? != b.instance.?);
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "conv", 0), "P.conv$int"));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "conv", 1), "P.conv$bool"));
    const w = try calleeMethod(&c, "with", 0);
    try testing.expect(w.is_static and Type.eql(w.recv, a.recv));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "with", 0), "P.with$bool"));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "with", 1), "P.with$str"));
}

test "own type params on a non-generic enum make a template" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { one(int), none }
        \\impl E {
        \\ fn fold[U](self, f: fn(int) -> U, d: U) -> U { match self { .one(v) -> f(v), .none -> d } }
        \\ fn make[U](v: int, _u: U) -> E { E.one(v) }
        \\}
        \\fn main() -> int { return E.make(2, true).fold(fn(x) { x }, 0) }
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    const f = try calleeMethod(&c, "fold", 0);
    try testing.expect(f.recv.kind == .@"enum" and !f.is_static);
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "make", 0), "E.make$bool"));
}

test "own type params on a builtin type stay T0013 at the call" {
    const gpa = testing.allocator;
    const src =
        \\impl int {
        \\ fn twice[U](self, u: U) -> U { u }
        \\}
        \\fn main() -> int { return (3).twice(1) }
        \\
    ;
    try expectOnlyDiag(src, .T0013);
    var c = try checkSource(src);
    defer c.deinit(gpa);
    for (c.result.diags) |d| try testing.expect(std.mem.indexOf(u8, d.message, "type_var") == null);
}

test "an associated fn's own type param binds from its args" {
    const gpa = testing.allocator;
    var c = try checkSource(own_box_pre ++
        \\fn main() -> int {
        \\ return Box.with(3, true).get() + Box[int].with(4, "s").get()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    const a = try calleeMethod(&c, "with", 0);
    const b = try calleeMethod(&c, "with", 1);
    try testing.expect(a.is_static and b.is_static);
    try testing.expect(a.instance.? != b.instance.?);
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "with", 0), "Box.with$int$bool"));
    try testing.expect(std.mem.endsWith(u8, try instanceNameOf(&c, "with", 1), "Box.with$int$str"));
}

test "an own-generic method of a bounded impl binds the impl params from the receiver and its own from the lambda" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Pair[K, V] { k: K, v: V }
        \\impl Pair[K has Hashable, V] {
        \\ fn mapv[W](self, f: fn(V) -> W) -> Pair[K, W] { Pair { k: self.k, v: f(self.v) } }
        \\}
        \\fn main() -> int {
        \\ p := Pair { k: "a", v: 2 }
        \\ q := p.mapv(fn(v) { v > 1 })
        \\ return if q.v { q.k.len() } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    const m = try calleeMethod(&c, "mapv", 0);
    const args = c.result.instances[m.instance.?].args;
    try testing.expectEqual(@as(usize, 3), args.len);
    try testing.expect(Type.eql(Type.str, args[0]));
    try testing.expect(Type.eql(Type.int, args[1]));
    try testing.expect(Type.eql(Type.bool, args[2]));
}

test "a lambda of the wrong arity in a generic call reports its count, naming the open param" {
    const pre = own_box_pre ++ "fn apg[T](x: T, f: fn(T, T) -> T) -> T { f(x, x) }\n";
    const own = pre ++ "fn main() -> int {\n b := Box { v: 2 }\n return b.map(fn(x, y) { x + y }).get()\n}\n";
    try expectOnlyDiag(own, .T0039);
    try expectOneDiag(own, .T0039, "lambda takes 2 parameter(s), but the expected type fn(int) -> U takes 1");
    const bare = pre ++ "fn main() -> int { return apg(1, fn(p) { p }) }\n";
    try expectOnlyDiag(bare, .T0039);
    try expectOneDiag(bare, .T0039, "lambda takes 1 parameter(s), but the expected type fn(int, int) -> int takes 2");
}

test "diagnostics spell a generic type with its type arguments" {
    const pre =
        \\struct Box[T] { x: T }
        \\struct Pair[A, B] { a: A, b: B }
        \\fn unbox[T](b: Box[T], d: T) -> T { d }
        \\fn take(p: Pair[str, Box[int]]) -> int { 0 }
        \\fn same[T](a: T, b: T) -> int { 0 }
        \\fn wrap(f: fn(int) -> Box[int]) -> int { 0 }
        \\
    ;
    try expectOneDiag(pre ++ "fn main() -> int { return unbox(Box[int] { x: 1 }) }\n", .T0039, "expected 2 argument(s), got 1; 'unbox' takes (Box[T], T)");
    try expectOneDiag(pre ++ "fn main() -> int {\n p := Pair { a: \"s\", b: Box { x: true } }\n return take(p)\n}\n", .T0040, "argument 1: expected Pair[str, Box[int]], got Pair[str, Box[bool]]; 'take' takes (Pair[str, Box[int]])");
    try expectOneDiag(pre ++ "fn main() -> int { return same(Box { x: 1 }, Box { x: true }) }\n", .T0015, "conflicting types for type parameter 'T': Box[int] vs Box[bool]");
    try expectOneDiag(pre ++ "fn main() -> int { return wrap(5) }\n", .T0040, "argument 1: expected fn(int) -> Box[int], got int; 'wrap' takes (fn(int) -> Box[int])");
    try expectOneDiag(pre ++ "fn main() -> int {\n x := Box { x: true }\n b: Box[int] = x\n return 0\n}\n", .T0041, "cannot bind Box[bool] to 'b' of type Box[int]");
}

test "a generic body failing alike in every instance reports once, as the template spells it" {
    const src =
        \\struct Box[T] { x: T }
        \\fn take(b: Box[int]) -> int { 0 }
        \\fn f[T](v: T) -> int {
        \\ b := Box { x: v }
        \\ return take(b)
        \\}
        \\fn main() -> int { return f(true) + f("s") }
        \\
    ;
    try expectOnlyDiag(src, .T0040);
    try expectOneDiag(src, .T0040, "argument 1: expected Box[int], got Box[T]; 'take' takes (Box[int])");
}

test "an argument that cannot fit its partly bound param is T0040, not T0016" {
    const pre = own_box_pre ++
        \\impl Box[T] { fn then[U](self, f: fn(T) -> Box[U]) -> Box[U] { f(self.v) } }
        \\
    ;
    const annotated = pre ++ "fn main() -> int {\n b := Box { v: 2 }\n return b.map(fn(x: int, y: int) { x + y }).get()\n}\n";
    try expectOnlyDiag(annotated, .T0040);
    try expectOneDiag(annotated, .T0040, "argument 1: expected fn(int) -> U, got fn(int, int) -> int; 'map' takes (fn(int) -> U)");
    const flat = pre ++ "fn main() -> int {\n b := Box { v: 2 }\n return b.then(fn(y) { y + 1 }).get()\n}\n";
    try expectOnlyDiag(flat, .T0040);
    try expectOneDiag(flat, .T0040, "argument 1: expected fn(int) -> Box[U], got fn(int) -> int; 'then' takes (fn(int) -> Box[U])");
}

test "an own-generic arity error shows the signature as the receiver binds it" {
    const pre = own_box_pre ++ "fn main() -> int {\n b := Box { v: 2 }\n";
    try expectOnlyDiag(pre ++ " return b.map().get()\n}\n", .T0039);
    try expectOneDiag(pre ++ " return b.map().get()\n}\n", .T0039, "expected 1 argument(s), got 0; 'map' takes (fn(int) -> U)");
    try expectOneDiag(pre ++ " return b.fold(0)\n}\n", .T0039, "expected 2 argument(s), got 1; 'fold' takes (A, fn(A, int) -> A)");
    // The rejected call's lambda is no second error.
    try expectOnlyDiag(pre ++ " return b.map(fn(x) { x }, 2).get()\n}\n", .T0039);
}

test "a generic call's failed argument is its only error" {
    const src = own_box_pre ++
        \\fn main() -> int {
        \\ b := Box { v: "a" }
        \\ sep := "-"
        \\ return b.map(sep.concat).get().len()
        \\}
        \\
    ;
    try expectOnlyDiag(src, .T0049);
    try expectOnlyDiag("fn id[T](x: T) -> T { x }\nfn main() -> int { return id(1 + true) }\n", .T0044);
}

test "the args of a call rejected for its type params add no T0050" {
    try expectOnlyDiag(
        \\impl int { fn conv[U](self, f: fn(int) -> U) -> U { f(self) } }
        \\fn main() -> int { return (4).conv(fn(x) { x + 1 }) }
        \\
    , .T0013);
    try expectOnlyDiag(own_box_pre ++ "fn main() -> int {\n b := Box { v: 2 }\n return b.map[int, int](fn(x) { x }).get()\n}\n", .T0054);
}

test "a builtin type's method with type params of its own is T0013 at its declaration" {
    const src = "impl int {\n fn twice[U](self, u: U) -> U { u }\n}\nfn main() -> int { return 0 }\n";
    try expectOnlyDiag(src, .T0013);
    try expectOneDiag(src, .T0013, "'twice' declares its own type parameters, which a method of the builtin type 'int' cannot have yet; make it a generic fn");
    var called = try checkSource("impl int {\n fn twice[U](self, u: U) -> U { u }\n}\nfn main() -> int { return (3).twice(1) }\n");
    defer called.deinit(testing.allocator);
    try testing.expectEqual(1, called.result.diags.len);
    try testing.expectEqual(.T0013, called.result.diags[0].code);
}

test "a type param that repeats a name is an error at its declaration" {
    const shadow = "struct Box[T] { v: T }\nimpl Box[T] {\n fn conv[T](self, x: T) -> T { x }\n}\nfn main() -> int { return 0 }\n";
    try expectOnlyDiag(shadow, .T0053);
    try expectOneDiag(shadow, .T0053, "type parameter 'T' is already the impl's; give the method's own a different name");
    const twice = "fn id[T, T](x: T) -> T { x }\nfn main() -> int { return 0 }\n";
    try expectOnlyDiag(twice, .T0053);
    try expectOneDiag(twice, .T0053, "type parameter 'T' is declared twice");
}

test "an impl pattern that is partly concrete or repeats a name is T0013" {
    const want = "an inherent 'impl' whose pattern is partly concrete or repeats a name is not yet supported; name each type parameter once, as in 'impl Pair[A, B]'";
    for ([_][]const u8{ "Pair[T, int]", "Pair[T, T]" }) |pat| {
        const src = try std.fmt.allocPrint(testing.allocator, "struct Pair[A, B] {{ a: A, b: B }}\nimpl {s} {{ fn first(self) -> int {{ 1 }} }}\nfn main() -> int {{ return 0 }}\n", .{pat});
        defer testing.allocator.free(src);
        try expectOnlyDiag(src, .T0013);
        try expectOneDiag(src, .T0013, want);
    }
}

test "an impl rejected for its pattern is its only error: no conformance check, no error at its calls (#66)" {
    const cases = [_]struct { src: []const u8, want: []const u8 }{
        .{ .src =
        \\struct Box[T] { v: T }
        \\impl Box[int] has Into[int] {
        \\    fn into(self) -> int { self.v }
        \\}
        \\fn main() -> int {
        \\    b := Box { v: 7 }
        \\    f := b.into
        \\    return b.into() + (b |> .into()) + f()
        \\}
        \\
        , .want = "a conformance 'impl' on a concrete type instance is not yet supported (use 'impl Box[T]')" },
        .{ .src =
        \\struct Cell[T] { v: T }
        \\impl Cell[int] {
        \\    fn get(self) -> int { self.v }
        \\    fn twice(self) -> int { self.v * 2 }
        \\}
        \\fn main() -> int {
        \\    c := Cell { v: 7 }
        \\    return c.get() + c.twice()
        \\}
        \\
        , .want = "an inherent 'impl' on a concrete type instance is not yet supported (use 'impl Cell[T]')" },
        .{ .src =
        \\struct Pair[K, V] { k: K, v: V }
        \\impl Pair[K, K] has Into[int] {
        \\    fn into(self) -> int { 1 }
        \\}
        \\fn main() -> int {
        \\    p := Pair { k: 1, v: 2 }
        \\    return p.into()
        \\}
        \\
        , .want = "a conformance 'impl' whose pattern is partly concrete or repeats a name is not yet supported; name each type parameter once, as in 'impl Pair[K, V]'" },
        .{ .src =
        \\enum Opt[T] { some(T), none }
        \\impl Opt[int] {
        \\    fn get(self) -> int { 1 }
        \\}
        \\fn main() -> int {
        \\    o := Opt.some(1)
        \\    return o.get()
        \\}
        \\
        , .want = "an inherent 'impl' on a concrete type instance is not yet supported (use 'impl Opt[T]')" },
        // A bound the rejected conformance would have met is not reported unmet.
        .{ .src =
        \\protocol Show { fn render(self) -> int }
        \\struct Cell[T] { v: T }
        \\impl Cell[int] has Show { fn render(self) -> int { self.v } }
        \\fn use_show[T has Show](t: T) -> int { t.render() }
        \\fn main() -> int {
        \\    b := Cell { v: 7 }
        \\    return use_show(b)
        \\}
        \\
        , .want = "a conformance 'impl' on a concrete type instance is not yet supported (use 'impl Cell[T]')" },
        // Nor is a call of its method through the type.
        .{ .src =
        \\struct Box[T] { v: T }
        \\impl Box[int] {
        \\    fn make() -> Box[int] { Box { v: 1 } }
        \\    fn get(self) -> int { self.v }
        \\}
        \\fn main() -> int {
        \\    b := Box[int].make()
        \\    c := Box.make()
        \\    return b.get() + c.v + Box.get(b)
        \\}
        \\
        , .want = "an inherent 'impl' on a concrete type instance is not yet supported (use 'impl Box[T]')" },
        // A conformance rejected for its protocol args is the only error too.
        .{ .src =
        \\struct P { x: int }
        \\impl P has Into[Option[int]] { fn into(self) -> Option[int] { Option.some(self.x) } }
        \\fn main() -> int {
        \\    p := P { x: 1 }
        \\    _o: Option[int] = p.into[Option[int]]()
        \\    return 0
        \\}
        \\
        , .want = "a generic-protocol argument must be a concrete non-composite value type (composite protocol args are not yet supported)" },
    };
    for (cases) |c| {
        try expectOnlyDiag(c.src, .T0013);
        try expectOneDiag(c.src, .T0013, c.want);
    }

    // A method the rejected impl does not provide is still an error.
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[int] { fn get(self) -> int { self.v } }
        \\fn main() -> int {
        \\    b := Box { v: 7 }
        \\    return b.get() + b.nope()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), c.result.diags.len);
    try testing.expectEqual(codes.Code.T0013, c.result.diags[0].code);
    try testing.expectEqual(codes.Code.T0018, c.result.diags[1].code);
}

test "an associated fn with type params of its own is not a value" {
    const src = own_box_pre ++ "fn main() -> int {\n h := Box[int].with\n return 0\n}\n";
    try expectOnlyDiag(src, .T0049);
    try expectOneDiag(src, .T0049, "'with' is a generic function and cannot be used as a value; wrap the call in a lambda");
}

test "a struct or enum name used as a value is T0056, an enum's with a hint, wherever it stands" {
    const pre =
        \\enum Shape { Circle, Square(int) }
        \\enum E[T] { a(T), b }
        \\enum W { s { r: int } }
        \\struct P { x: int }
        \\struct N(int)
        \\fn id[T](x: T) -> T { x }
        \\
    ;
    const shape = "type 'Shape' is not a value; name one of its variants, as in 'Shape.Circle'";
    const cases = [_]struct { body: []const u8, want: []const u8 }{
        .{ .body = "_x := Shape", .want = shape },
        .{ .body = "_x := Shape |> id", .want = shape },
        .{ .body = "_x := id(Shape)", .want = shape },
        .{ .body = "_x := Shape[int]", .want = shape },
        .{ .body = "_b := Shape == Shape", .want = shape },
        .{ .body = "for _x in Shape {}", .want = shape },
        .{ .body = "_x := E", .want = "type 'E' is not a value; name one of its variants, as in 'E.a(..)'" },
        .{ .body = "_x := W", .want = "type 'W' is not a value; name one of its variants, as in 'W.s { .. }'" },
        .{ .body = "_x := Option", .want = "type 'Option' is not a value; name one of its variants, as in 'Option.some(..)'" },
        .{ .body = "_x := P", .want = "type 'P' is not a value" },
        .{ .body = "_x := N", .want = "type 'N' is not a value" },
    };
    for (cases) |case| {
        const src = try std.fmt.allocPrint(testing.allocator, "{s}fn main() -> int {{\n {s}\n return 0\n}}\n", .{ pre, case.body });
        defer testing.allocator.free(src);
        try expectOneDiag(src, .T0056, case.want);
    }
}

test "a call's expected type reaches a literal receiver only as a hint" {
    const gpa = testing.allocator;
    var c = try checkSource(own_box_pre ++
        \\fn names() -> Box[str] {
        \\ return Box { v: 1 }.map(fn(v) { v.to_string() })
        \\}
        \\fn main() -> int {
        \\ n: int = Box { v: 2 }.fold(0, fn(a, x) { a + x })
        \\ x: int8 = Box { v: 1 }.get()
        \\ return names().get().len() + n + if x == 1 { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
}

test "a struct literal's field value is typed against its field" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int8, o: Option[int] }
        \\fn main() -> int {
        \\ p := P { x: 5, o: .none }
        \\ q: P = P { x: 6, o: Option.some(1) }
        \\ return if p.x == 5 { q.o.expect("no value") } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
}

test "a generic construction's part or call's arg takes its type from the parent's expected type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\struct OBox[T] { v: Option[T] }
        \\fn wrap[T](x: T) -> Option[T] { Option.some(x) }
        \\fn nest(v: int) -> Option[Option[int]] { Option.some(Option.some(v)) }
        \\fn main() -> int {
        \\ b: Box[int8] = Box { v: 1 }
        \\ o: OBox[int] = OBox { v: Option.none }
        \\ x: Option[int8] = Option.some(5)
        \\ w: Option[int8] = wrap(5)
        \\ ww: Option[Option[int]] = wrap(Option.some(1))
        \\ return if b.v == 1 && o.v.is_none() && x.is_some() && w.is_some() && ww.is_some() && nest(1).is_some() { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try expectOwnGenericOk(&c);
    // The part is checked against its own type, so a literal too big for it is one T0034.
    try expectOnlyDiag("struct Box[T] { v: T }\nfn main() -> int {\n b: Box[int8] = Box { v: 300 }\n return 0\n}\n", .T0034);
}

test "an own-generic call that selects no instance keeps its template, never another call's row" {
    const gpa = testing.allocator;
    var c = try checkSource(own_box_pre ++
        \\fn main() -> int {
        \\ b := Box { v: "s" }
        \\ k := b.map(fn(s) { s.len() })
        \\ z := b.map(fn(s) { s.nope() })
        \\ _w := b.map()
        \\ return k.get() + z.get()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expect(c.result.diags.len > 0);
    try testing.expect((try calleeView(&c, "map", 0)) == .method);
    const map_fn = c.result.callees[0][(try memberSlot(&c, "map", 1)).int()].templateFn() orelse return error.TestExpectedTemplate;
    try testing.expectEqual(map_fn, c.result.callees[0][(try memberSlot(&c, "map", 2)).int()].templateFn().?);
    try testing.expectEqual(Callee.View.none, try calleeView(&c, "map", 1));
}

test "a capture passed where a non-function is wanted is T0040 with the capture hint" {
    const pre = "fn add(a: int, b: int) -> int { a + b }\nfn neg(a: int) -> int { 0 - a }\nfn apply(f: fn(int) -> int, x: int) -> int { f(x) }\n";
    try expectOnlyDiag(pre ++ "fn main() -> int { return neg(add(1, _)) }\n", .T0040);
    try expectOneDiag(pre ++ "fn main() -> int { return neg(add(1, _)) }\n", .T0040, "argument 1: expected int, got fn(int) -> int; 'neg' takes (int); its '_' makes it a function, so to pass the call's result, write the lambda around the outer call, as in fn(v: int) { f(g(v)) }");
    const gpa = testing.allocator;
    var ok = try checkSource(pre ++ "fn main() -> int { return apply(neg(_), 3) }\n");
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
}

test "a capture passed as a constructor's non-function value gets the capture hint" {
    const gpa = testing.allocator;
    const hint = "; its '_' makes it a function, so to pass the call's result, write the lambda around the outer call, as in fn(v: int) { f(g(v)) }";
    const cases = [_]struct { body: []const u8, want: []const u8 }{
        .{ .body = "_s := Shape.circle(add(1, _))", .want = "variant 'Shape.circle': expected int, got fn(int) -> int" ++ hint },
        .{ .body = "_p := Pt(add(1, _), 2)", .want = "tuple struct 'Pt': expected int, got fn(int) -> int" ++ hint },
        .{ .body = "_q := Pt(true, 2)", .want = "tuple struct 'Pt': expected int, got bool" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa, "fn add(a: int, b: int) -> int {{ a + b }}\nenum Shape {{ circle(int) }}\nstruct Pt(int, int)\nfn main() -> int {{\n {s}\n return 0\n}}\n", .{cs.body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer for (c.result.diags) |d| std.debug.print("{s}: got {s}: {s}\n", .{ cs.body, @tagName(d.code), d.message });
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqualStrings(cs.want, c.result.diags[0].message);
    }
}

test "a '?' or jump in a capture's arguments names the capture, not a lambda" {
    const gpa = testing.allocator;
    const cases = [_]struct { body: []const u8, code: codes.Code, want: []const u8 }{
        .{ .body = "_h := add(get(1)?, _)", .code = .T0032, .want = "'?' in a '_' capture exits the capture, so it needs an expected fn type returning Option[..] or Result[..]; or write the lambda out with its '-> Option[..]'" },
        .{ .body = "loop {\n _g := add(if t { break } else { 1 }, _)\n }", .code = .T0051, .want = "cannot break a loop outside this '_' capture" },
        .{ .body = "loop {\n _g := add(if t { continue } else { 1 }, _)\n break\n }", .code = .T0051, .want = "cannot continue a loop outside this '_' capture" },
        .{ .body = "_h := fn() -> Option[int] { _k := add(get(1)?, _)\n Option.none }", .code = .T0032, .want = "'?' in a '_' capture exits the capture, so it needs an expected fn type returning Option[..] or Result[..]; or write the lambda out with its '-> Option[..]'" },
        .{ .body = "_h := add(1, _)\n _l := fn() { _m := get(1)? }", .code = .T0032, .want = "'?' in a lambda needs its return type; annotate -> Option[..] or Result[..]" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(gpa, "fn add(a: int, b: int) -> int {{ a + b }}\nfn get(k: int) -> Option[int] {{ Option.some(k) }}\nfn f(t: bool) -> Option[int] {{\n {s}\n return Option.none\n}}\n", .{cs.body});
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        errdefer for (c.result.diags) |d| std.debug.print("{s}: got {s}: {s}\n", .{ cs.body, @tagName(d.code), d.message });
        try testing.expect(hasCode(&c, cs.code));
        for (c.result.diags) |d| if (d.code == cs.code) try testing.expectEqualStrings(cs.want, d.message);
    }
}

test "a capture dropped as a statement is W0012, since its call never runs" {
    const gpa = testing.allocator;
    const pre = "fn show(x: int) { _y := x }\nfn add(a: int, b: int) -> int { a + b }\n";
    const msg = "unused '_' capture; the '_' makes the call a function, which is never called; to pass a local named '_', rename it";
    for ([_][]const u8{
        pre ++ "fn main() -> int {\n _ := 5\n show(_)\n return 0\n}\n",
        pre ++ "fn main() -> int {\n add(1, _)\n return 0\n}\n",
        pre ++ "fn main() {\n add(1, _)\n}\n",
    }) |src| {
        try expectOnlyDiag(src, .W0012);
        try expectOneDiag(src, .W0012, msg);
    }
    var ok = try checkSource(pre ++ "fn main() -> int {\n f := add(1, _)\n 2 |> show(_)\n return f(1)\n}\n");
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
}

test "a function value dropped as a statement is W0012, with the call spelled from the source" {
    const pre =
        \\struct C { n: int, f: fn() -> int }
        \\struct T2(fn(int) -> int)
        \\impl C {
        \\ fn bump(mut self) { self.n = self.n + 1 }
        \\ fn add(self, k: int) -> int { self.n + k }
        \\}
        \\fn one() -> int { 1 }
        \\fn inc(x: int) -> int { x + 1 }
        \\
    ;
    const Case = struct { body: []const u8, want: []const u8 };
    const cases = [_]Case{
        .{ .body = "c.bump", .want = "function value is never called; did you mean 'c.bump()'?" },
        .{ .body = "c.add", .want = "function value is never called; did you mean 'c.add(..)'?" },
        .{ .body = "one", .want = "function value is never called; did you mean 'one()'?" },
        .{ .body = "inc", .want = "function value is never called; did you mean 'inc(..)'?" },
        .{ .body = "c.f", .want = "function value is never called; did you mean 'c.f()'?" },
        .{ .body = "t.0", .want = "function value is never called; did you mean 't.0(..)'?" },
        .{ .body = "g := one\n g", .want = "function value is never called; did you mean 'g()'?" },
        .{ .body = "fn local() -> int { 2 }\n local", .want = "function value is never called; did you mean 'local()'?" },
        .{ .body = "c.\n  bump", .want = "function value is never called; add '()' to call it" },
        .{ .body = "fn() { c.bump() }", .want = "lambda is never called, so its body never runs; bind it with ':=' and call it, or write its statements without the lambda" },
        .{ .body = "if c.n > 0 { one }", .want = "function value is never called; did you mean 'one()'?" },
        .{ .body = "_r: fn(int) = fn(_v) { c.bump }", .want = "function value is never called; did you mean 'c.bump()'?" },
    };
    for (cases) |cs| {
        const src = try std.fmt.allocPrint(testing.allocator, "{s}fn main() -> int {{\n c := C {{ n: 0, f: one }}\n t := T2(inc)\n {s}\n return c.n + t.0(0)\n}}\n", .{ pre, cs.body });
        defer testing.allocator.free(src);
        try expectOnlyDiag(src, .W0012);
        try expectOneDiag(src, .W0012, cs.want);
    }
}

test "a function value that is used, or a call that returns one, is not W0012" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct C { n: int }
        \\impl C { fn get(self) -> int { self.n } }
        \\fn one() -> int { 1 }
        \\fn apply(f: fn() -> int) -> int { f() }
        \\fn make() -> fn() -> int { one }
        \\fn tail() -> fn() -> int { one }
        \\fn ret() -> fn() -> int { return one }
        \\fn pick(b: bool) -> fn() -> int { if b { one } else { make() } }
        \\fn main() -> int {
        \\ c := C { n: 0 }
        \\ f := c.get
        \\ g: fn() -> int = make()
        \\ g = one
        \\ make()
        \\ h := { one }
        \\ k := fn() { one }
        \\ _m := fn(_v: int) { c.get }
        \\ return f() + g() + apply(one) + (one |> apply) + h() + k()() + tail()() + ret()() + pick(true)()
        \\}
    );
    defer c.deinit(gpa);
    for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "a method named like a fn-typed field of its struct is W0013 at the method, naming both rules" {
    const gpa = testing.allocator;
    const Case = struct { src: []const u8, want: []const u8 };
    const cases = [_]Case{
        .{
            .src = "struct H { run: fn(int) -> int }\nimpl H {\n fn run(self, x: int) -> int { x + 1000 }\n}\nfn main() -> int {\n h := H { run: fn(x) { x + 1 } }\n f := h.run\n return h.run(1) + f(1)\n}\n",
            .want = "method 'run' on 'H' has the same name as its field 'run' of fn type: 'h.run(..)' calls the method and 'h.run' reads the field; rename one",
        },
        .{
            .src = "struct Box[T] { v: T, f: fn() -> T }\nimpl Box[T] {\n fn f(self) -> T { self.v }\n}\nfn main() -> int {\n b := Box { v: 2, f: fn() { 3 } }\n return b.f()\n}\n",
            .want = "method 'f' on 'Box' has the same name as its field 'f' of fn type: 'b.f(..)' calls the method and 'b.f' reads the field; rename one",
        },
        .{
            .src = "protocol Runner { fn go(self) -> int }\ntype Op = fn() -> int\nstruct H { go: Op }\nimpl H has Runner {\n fn go(self) -> int { 1 }\n}\nfn main() -> int {\n h := H { go: fn() { 2 } }\n return h.go()\n}\n",
            .want = "method 'go' on 'H' has the same name as its field 'go' of fn type: 'h.go(..)' calls the method and 'h.go' reads the field; rename one",
        },
    };
    for (cases) |cs| {
        try expectOnlyDiag(cs.src, .W0013);
        try expectOneDiag(cs.src, .W0013, cs.want);
        var c = try checkSource(cs.src);
        defer c.deinit(gpa);
        const d = c.result.diags[0];
        const sp = d.span();
        const rel = d.relatedSpan().?;
        try testing.expectEqualStrings(cs.src[sp.start..sp.end], cs.src[rel.start..rel.end]);
        try testing.expect(rel.start < sp.start);
    }
}

test "a method beside a field of another type, or an associated fn beside a fn field, is not W0013" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct A { n: int, xs: Box[fn() -> int] }
        \\struct Box[T] { v: T }
        \\impl A {
        \\ fn n(self) -> int { self.n }
        \\ fn xs(self) -> int { self.n }
        \\}
        \\impl Box[T] {
        \\ fn v(self) -> T { self.v }
        \\}
        \\struct H { run: fn(int) -> int }
        \\impl H {
        \\ fn run(x: int) -> int { x }
        \\}
        \\fn one() -> int { 1 }
        \\fn main() -> int {
        \\ a := A { n: 1, xs: Box { v: one } }
        \\ h := H { run: fn(x) { x } }
        \\ f := h.run
        \\ return a.n() + a.xs() + a.xs.v()() + f(1)
        \\}
    );
    defer c.deinit(gpa);
    for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "an enum fn named like a variant is W0015 when associated and W0016 when a method, at the fn" {
    const gpa = testing.allocator;
    const Case = struct { src: []const u8, code: codes.Code, want: []const u8 };
    const cases = [_]Case{
        .{
            .src = "enum E { A(int), B }\nimpl E {\n fn A(n: int) -> E { E.A(n + 1) }\n}\nfn main() -> int {\n e := E.A(1)\n return match e {\n  E.A(v) -> v,\n  E.B -> 99,\n }\n}\n",
            .code = .W0015,
            .want = "associated fn 'A' of 'E' can never be called: 'E.A' always names the variant 'A'; rename the function",
        },
        .{
            .src = "enum O[T] { S(T), N }\nimpl O[T] {\n fn N() -> O[T] { O.N }\n}\nfn main() -> int {\n o: O[int] = O.S(3)\n return match o {\n  O.S(v) -> v,\n  O.N -> 0,\n }\n}\n",
            .code = .W0015,
            .want = "associated fn 'N' of 'O' can never be called: 'O.N' always names the variant 'N'; rename the function",
        },
        .{
            .src = "enum E { A(int), B }\nimpl E {\n fn B(self) -> int { 7 }\n}\nfn main() -> int {\n e := E.B\n return e.B()\n}\n",
            .code = .W0016,
            .want = "method 'B' on 'E' has the name of its variant 'B': '.B(..)' on a value calls the method and 'E.B' is the variant; rename one",
        },
        .{
            .src = "enum O[T] { S(T), N }\nimpl O[T] {\n fn S(self) -> int { 1 }\n}\nfn main() -> int {\n o: O[int] = O.N\n return o.S()\n}\n",
            .code = .W0016,
            .want = "method 'S' on 'O' has the name of its variant 'S': '.S(..)' on a value calls the method and 'O.S' is the variant; rename one",
        },
    };
    for (cases) |cs| {
        try expectOnlyDiag(cs.src, cs.code);
        try expectOneDiag(cs.src, cs.code, cs.want);
        var c = try checkSource(cs.src);
        defer c.deinit(gpa);
        const d = c.result.diags[0];
        const sp = d.span();
        const rel = d.relatedSpan().?;
        try testing.expectEqualStrings(cs.src[sp.start..sp.end], cs.src[rel.start..rel.end]);
        try testing.expect(rel.start < sp.start);
    }
}

test "an enum fn named unlike its variants, or a struct fn, is not W0015 or W0016" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum E { A(int), B }
        \\impl E {
        \\ fn a(n: int) -> E { E.A(n) }
        \\ fn b(self) -> int { 2 }
        \\}
        \\enum O[T] { S(T), N }
        \\impl O[T] {
        \\ fn s(v: T) -> O[T] { O.S(v) }
        \\}
        \\struct A { n: int }
        \\impl A {
        \\ fn A() -> A { A { n: 1 } }
        \\ fn B(self) -> int { self.n }
        \\}
        \\fn main() -> int {
        \\ o: O[int] = O.s(1)
        \\ return E.a(1).b() + A.A().B() + match o { O.S(v) -> v, O.N -> 0 }
        \\}
    );
    defer c.deinit(gpa);
    for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "W0012 is reported once for a generic fn, and not under a poison return type" {
    try expectOnlyDiag("fn gen[T](f: fn(T) -> T, x: T) -> T {\n f\n x\n}\nfn main() -> int { gen(fn(x: int) { x }, 0) + gen(fn(s: str) { s }, \"a\").len() }\n", .W0012);
    try expectOnlyDiag("fn one() -> int { 1 }\nfn f() -> Nope {\n one\n}\nfn main() -> int { 0 }\n", .T0001);
}

test "a capture's '_' at a mut position is T0019; a piped value there is the caller's place" {
    const gpa = testing.allocator;
    const pre = "fn bump(mut x: int) { x = x + 1 }\n";
    const msg = "argument 1 is a capture's '_', a copy of the value it is called with; 'bump' takes it as 'mut', so its change would be lost";
    for ([_][]const u8{
        pre ++ "fn main() -> int {\n _f := bump(_)\n return 0\n}\n",
        pre ++ "fn main() -> int {\n _h: fn(int) = bump(_)\n return 0\n}\n",
    }) |src| {
        try expectOnlyDiag(src, .T0019);
        try expectOneDiag(src, .T0019, msg);
    }
    var ok = try checkSource(pre ++ "fn main() -> int {\n x := 1\n x |> bump(_)\n return x\n}\n");
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
}

test "a capture of a 'mut self' receiver is T0052 only when it escapes, like the lambda it means" {
    const gpa = testing.allocator;
    const pre =
        \\struct C { n: int }
        \\impl C {
        \\ fn add(mut self, k: int) { self.n = self.n + k }
        \\
    ;
    var ok = try checkSource(pre ++
        \\ fn run(mut self) -> int {
        \\  f := self.add(_)
        \\  f(1)
        \\  self.n
        \\ }
        \\}
        \\fn main() -> int {
        \\ c := C { n: 1 }
        \\ return c.run()
        \\}
        \\
    );
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
    try expectOnlyDiag(pre ++
        \\ fn adder(mut self) -> fn(int) { self.add(_) }
        \\}
        \\fn main() -> int {
        \\ c := C { n: 1 }
        \\ f := c.adder()
        \\ f(1)
        \\ return c.n
        \\}
        \\
    , .T0052);
}

test "a capture whose expected fn type takes another count is one T0039 with capture wording" {
    const src = "fn add(a: int, b: int) -> int { a + b }\nfn main() -> int {\n w: fn(int, int) -> int = add(1, _)\n return w(1, 2)\n}\n";
    try expectOnlyDiag(src, .T0039);
    try expectOneDiag(src, .T0039, "a '_' capture takes 1 parameter, but the expected type fn(int, int) -> int takes 2");
}

test "a lambda whose arity differs from the expected fn type is one T0039" {
    const gpa = testing.allocator;
    const srcs = [_][]const u8{
        "fn apply(f: fn(int) -> int) -> int { f(1) }\nfn main() -> int { return apply(fn(a, b) { a }) }\n",
        "fn apply(f: fn(int) -> int) -> int { f(1) }\nfn main() -> int { return apply(fn(a: int, b: int) { a }) }\n",
        "fn main() -> int {\n q: fn(int) -> int = fn() { 1 }\n return q(1)\n}\n",
    };
    for (srcs) |src| {
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0039, c.result.diags[0].code);
        try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "the expected type fn(int) -> int takes 1") != null);
    }
}

test "the return type is inferred from the body and returns" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ a := fn(x: int) { x > 1 }
        \\ b := fn(x: int) {
        \\  if x > 0 { return "pos" }
        \\  "neg"
        \\ }
        \\ d := fn() { return 3 }
        \\ e := fn() { }
        \\ g := fn(x: int) -> int { x }
        \\ e()
        \\ if a(2) { return b(1).len() + d() + g(1) }
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const rets = [_]Type{ Type.bool, Type.str, Type.int, Type.unit, Type.int };
    for (rets, 0..) |r, i| try testing.expect(Type.eql(r, lambdaSig(&c, try nthLambda(&c, i)).ret));
}

test "return and ? are local to the lambda" {
    const gpa = testing.allocator;
    {
        // A lambda's `return` does not return from the owner, which still needs its own.
        var c = try checkSource("fn main() -> int {\n f := fn() { return 1 }\n _x := f()\n}\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0043, c.result.diags[0].code);
        try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "function 'main'") != null);
    }
    {
        // `?` checks against the lambda's return type, not the owner's `int`.
        var c = try checkSource(
            \\fn main() -> int {
            \\ f := fn(o: Option[int]) -> Option[int] {
            \\  v := o?
            \\  .some(v + 1)
            \\ }
            \\ return f(.some(1)).or(0)
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    }
    {
        // An inferred return type cannot take a `?`; a mismatched return and a fall-off are coded.
        var c = try checkSource(
            \\fn main() -> int {
            \\ f := fn(o: Option[int]) { o? }
            \\ g := fn(b: bool) -> int {
            \\  if b { return 1 }
            \\ }
            \\ h := fn(b: bool) {
            \\  if b { return 1 }
            \\  return true
            \\ }
            \\ return 0
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expect(hasCode(&c, .T0032));
        try testing.expect(hasCode(&c, .T0043));
        try testing.expect(hasCode(&c, .T0042));
    }
}

test "break/continue across a lambda is T0051, labeled or not; loops inside work" {
    const gpa = testing.allocator;
    const bad = [_][]const u8{
        "fn main() {\n for i in 0..3 {\n  f := fn() { break }\n }\n}\n",
        "fn main() {\n while true {\n  f := fn() { continue }\n  break\n }\n}\n",
        "fn main() {\n @o loop {\n  f := fn() {\n   for i in 0..2 { break @o }\n  }\n  break\n }\n}\n",
        "fn main() {\n @b {\n  f := fn() { break @b }\n }\n}\n",
    };
    for (bad) |src| {
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try testing.expect(hasCode(&c, .T0051));
    }
    var ok = try checkSource(
        \\fn main() -> int {
        \\ f := fn(n: int) -> int {
        \\  t := 0
        \\  @o for i in 0..n {
        \\   for j in 0..n {
        \\    if j > i { continue }
        \\    if i > 3 { break @o }
        \\    t = t + j
        \\   }
        \\  }
        \\  loop { break t }
        \\ }
        \\ return f(5)
        \\}
        \\
    );
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
    // With no loop anywhere, a jump keeps its plain (uncoded) message.
    var none = try checkSource("fn main() {\n f := fn() { break }\n f()\n}\n");
    defer none.deinit(gpa);
    try testing.expect(!hasCode(&none, .T0051));
    try testing.expect(anyDiagContains(none, "break outside of a loop"));
}

test "env slots take their types from the first use and are laid out once" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct P { x: int, y: int }
        \\struct Big { a: int, b: int, c: int }
        \\fn main() -> int {
        \\ n := 1
        \\ s := "hi"
        \\ u := ()
        \\ p := P { x: 1, y: 2 }
        \\ big := Big { a: 1, b: 2, c: 3 }
        \\ f := fn(k: int) -> int {
        \\  _z := u
        \\  k + n + s.len() + p.x + big.c
        \\ }
        \\ g := fn() -> fn() -> int { fn() { n + p.y } }
        \\ h := fn() { u }
        \\ h()
        \\ return f(1) + g()()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const units = c.result.lambda_units;
    const find = struct {
        fn f(us: []const Closure.LambdaUnit, name: []const u8) !Closure.LambdaUnit {
            for (us) |u| if (std.mem.eql(u8, u.name, name)) return u;
            return error.TestUnitNotFound;
        }
    }.f;
    // Ascending slot: n, s, u, p, big; each at the next multiple of its alignment.
    const f = try find(units, "main$l0");
    try testing.expectEqual(@as(usize, 5), f.env.len);
    const want_kinds = [_]Kind{ .int, .str, .unit, .@"struct", .@"struct" };
    const want_offs = [_]u32{ 0, 8, 24, 24, 40 };
    for (f.env, want_kinds, want_offs, 0..) |es, k, off, i| {
        try testing.expectEqual(k, es.ty.kind);
        try testing.expectEqual(off, es.off);
        try testing.expectEqual(Closure.Hold.copy, heldAt(&c, f, es));
        if (i > 0) try testing.expect(f.env[i - 1].slot < es.slot);
    }
    try testing.expectEqual(@as(u32, 64), f.env_size);
    // The outer lambda of a nested pair carries the inner one's captures through.
    const outer = try find(units, "main$l1");
    const inner = try find(units, "main$l2");
    try testing.expectEqual(@as(usize, 2), outer.env.len);
    try testing.expectEqualSlices(u32, &.{ outer.env[0].slot, outer.env[1].slot }, &.{ inner.env[0].slot, inner.env[1].slot });
    try testing.expectEqual(@as(u32, 24), outer.env_size);
    // Only zero-sized captures: no env.
    const h = try find(units, "main$l3");
    try testing.expectEqual(@as(usize, 1), h.env.len);
    try testing.expectEqual(@as(u32, 0), h.env_size);
}

/// How unit `u`'s owner holds env slot `es` with every box in its cell.
fn heldAt(c: *const Checked, u: Closure.LambdaUnit, es: Closure.EnvSlot) Closure.Hold {
    return Closure.heapStorage(c.result.bodies.items[u.owner]).hold(es.slot);
}

/// The boxed slots of owner-closure record `oc`.
fn boxedOf(c: *const Checked, oc: Closure.OwnerClosure) []const u32 {
    return c.result.bodies.items[oc.owner].boxed;
}

/// The boxed slots of the one owner in `c` that has any, checked against the env of every
/// lambda unit that captures them: a boxed capture is an 8-byte cell word, any other a copy.
fn expectBoxedOnce(c: *const Checked, want_len: usize) ![]const u32 {
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const ocs = c.result.owner_closures;
    if (want_len == 0) {
        try testing.expectEqual(@as(usize, 0), ocs.len);
        for (c.result.lambda_units) |u| for (u.env) |es| try testing.expectEqual(Closure.Hold.copy, heldAt(c, u, es));
        return &.{};
    }
    try testing.expectEqual(@as(usize, 1), ocs.len);
    const boxed = boxedOf(c, ocs[0]);
    try testing.expectEqual(want_len, boxed.len);
    var boxed_env: usize = 0;
    for (c.result.lambda_units) |u| {
        if (!std.meta.eql(u.owner, ocs[0].owner)) continue;
        for (u.env) |es| {
            const is_boxed = Closure.contains(boxed, es.slot);
            try testing.expectEqual(if (is_boxed) Closure.Hold.cell else Closure.Hold.copy, heldAt(c, u, es));
            if (is_boxed) {
                try testing.expectEqual(@as(u32, 0), es.off % 8);
                boxed_env += 1;
            }
        }
    }
    try testing.expect(boxed_env >= want_len);
    return boxed;
}

test "boxed = captured ∩ (assigned ∪ mut self receivers ∪ mut arguments)" {
    const gpa = testing.allocator;
    const pre =
        \\struct P { x: int }
        \\struct Q { p: P }
        \\struct C { n: int }
        \\impl C {
        \\ fn bump(mut self) { self.n = self.n + 1 }
        \\ fn get(self) -> int { self.n }
        \\}
        \\
    ;
    const cases = [_][]const u8{
        // Assigned inside the lambda.
        "fn main() -> int {\n a := 1\n f := fn() { a = 3 }\n f()\n return a\n}\n",
        // Assigned outside it.
        "fn main() -> int {\n a := 1\n f := fn() -> int { a }\n a = 2\n return f()\n}\n",
        // A field store, inside or outside.
        "fn main() -> int {\n p := P { x: 1 }\n f := fn() { p.x = 2 }\n f()\n return p.x\n}\n",
        "fn main() -> int {\n p := P { x: 1 }\n f := fn() -> int { p.x }\n p.x = 4\n return f()\n}\n",
        // A nested field store.
        "fn main() -> int {\n q := Q { p: P { x: 1 } }\n f := fn() -> int { q.p.x }\n q.p.x = 5\n return f()\n}\n",
        // A `mut self` call, outside or inside.
        "fn main() -> int {\n c := C { n: 0 }\n f := fn() -> int { c.get() }\n c.bump()\n return f()\n}\n",
        "fn main() -> int {\n c := C { n: 0 }\n f := fn() { c.bump() }\n f()\n return c.n\n}\n",
        // A param, of the fn and of a lambda.
        "fn g(n: int) -> int {\n f := fn() { n = n + 1 }\n f()\n n\n}\nfn main() -> int { return g(1) }\n",
        "fn main() -> int {\n h := fn(n: int) -> int {\n  f := fn() { n = n + 1 }\n  f()\n  n\n }\n return h(1)\n}\n",
        // A range loop variable.
        "fn main() -> int {\n s := 0\n for i in 0..3 {\n  f := fn() { i = i + 1 }\n  f()\n  s = s + i\n }\n return s\n}\n",
        // A match binding.
        "fn main() -> int {\n o: Option[int] = .some(1)\n return match o {\n  .some(v) -> {\n   f := fn() { v = v + 1 }\n   f()\n   v\n  },\n  .none -> 0,\n }\n}\n",
    };
    for (cases) |body| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, body });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        _ = try expectBoxedOnce(&c, 1);
    }
    // Only the captured-and-assigned slot is boxed: `k` is read-only, `m` is assigned but
    // never captured.
    var mixed = try checkSource(
        \\fn main() -> int {
        \\ k := 5
        \\ m := 0
        \\ a := 1
        \\ f := fn() -> int {
        \\  a = a + k
        \\  a
        \\ }
        \\ m = f()
        \\ return m + a
        \\}
        \\
    );
    defer mixed.deinit(gpa);
    const bx = try expectBoxedOnce(&mixed, 1);
    const f_unit = mixed.result.lambda_units[0];
    try testing.expectEqual(@as(usize, 2), f_unit.env.len);
    try testing.expectEqual(Closure.Hold.copy, heldAt(&mixed, f_unit, f_unit.env[0])); // k
    try testing.expectEqual(Closure.Hold.cell, heldAt(&mixed, f_unit, f_unit.env[1])); // a
    try testing.expectEqual(f_unit.env[1].slot, bx[0]);
    // A boxed capture holds an 8-byte pointer, even for a wider value.
    var wide = try checkSource(
        \\struct Big { a: int, b: int, c: int }
        \\fn main() -> int {
        \\ b := Big { a: 1, b: 2, c: 3 }
        \\ f := fn() { b.c = 9 }
        \\ f()
        \\ return b.c
        \\}
        \\
    );
    defer wide.deinit(gpa);
    _ = try expectBoxedOnce(&wide, 1);
    try testing.expectEqual(@as(u32, 8), wide.result.lambda_units[0].env_size);
    // Read-only captures, and assigned locals no lambda names, are copies.
    var ro = try checkSource(pre ++
        \\fn main() -> int {
        \\ base := 10
        \\ other := 0
        \\ other = 1
        \\ c := C { n: 0 }
        \\ c.bump()
        \\ f := fn(x: int) -> int { x + base }
        \\ return f(other) + c.n
        \\}
        \\
    );
    defer ro.deinit(gpa);
    _ = try expectBoxedOnce(&ro, 0);
    // A local passed as a `mut` argument is written through, so a captured one is boxed,
    // whether the call is outside the lambda or inside it; without the call it is a copy.
    const bump = "fn bump(mut n: int) { n = n + 1 }\n";
    const mut_cases = [_]struct { body: []const u8, boxed: usize }{
        .{ .body = "fn main() -> int {\n c := 0\n f := fn() -> int { c }\n bump(c)\n return f()\n}\n", .boxed = 1 },
        .{ .body = "fn main() -> int {\n c := 0\n f := fn() { bump(c) }\n f()\n return c\n}\n", .boxed = 1 },
        .{ .body = "fn main() -> int {\n c := 0\n f := fn() -> int { c }\n d := c\n bump(d)\n return f() + d\n}\n", .boxed = 0 },
    };
    for (mut_cases) |mc| {
        const src = try std.mem.concat(gpa, u8, &.{ bump, mc.body });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        _ = try expectBoxedOnce(&c, mc.boxed);
    }
}

test "a program with no captures has no owner closure records" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ n := 1
        \\ n = n + 1
        \\ f := fn(x: int) -> int { x * 2 }
        \\ return f(n)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 0), c.result.owner_closures.len);
    try testing.expectEqual(@as(usize, 1), c.result.lambda_units.len);
    try testing.expectEqual(@as(u32, 0), c.result.lambda_units[0].env_size);
}

test "an instance recheck writes its own boxed list" {
    const gpa = testing.allocator;
    // `x.touch()` is a `mut self` call for `A` and a plain one for `B`, so only the `A`
    // instance boxes `x`.
    var c = try checkSource(
        \\struct A { n: int }
        \\struct B { n: int }
        \\impl A { fn touch(mut self) { self.n = self.n + 1 } }
        \\impl B { fn touch(self) { } }
        \\fn poke[T](x: T) -> int {
        \\ f := fn() { x.touch() }
        \\ f()
        \\ 0
        \\}
        \\fn main() -> int { return poke(A { n: 0 }) + poke(B { n: 0 }) }
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 2), c.result.instances.len);
    const ocs = c.result.owner_closures;
    try testing.expectEqual(@as(usize, 1), ocs.len);
    const k = c.result.bodies.items[ocs[0].owner].of.instance;
    const inst = c.result.instances[k];
    try testing.expectEqualStrings("A", c.result.layouts[inst.args[0].struct_id].name);
    try testing.expectEqualSlices(u32, &.{0}, inst.boxed);
    try testing.expectEqualSlices(u32, &.{0}, boxedOf(&c, ocs[0]));
    try testing.expectEqual(@as(usize, 0), c.result.instances[1 - k].boxed.len);
    for (c.result.lambda_units) |u| {
        try testing.expectEqual(@as(usize, 1), u.env.len);
        const want: Closure.Hold = if (c.result.bodies.items[u.owner].of.instance == k) .cell else .copy;
        try testing.expectEqual(want, heldAt(&c, u, u.env[0]));
    }
}

test "a closure that escapes cannot capture a mut self receiver (T0052 at its first use)" {
    const gpa = testing.allocator;
    const src =
        \\struct C { n: int }
        \\impl C {
        \\ fn get(mut self) -> int {
        \\  f := fn() -> int { self.n }
        \\  f()
        \\ }
        \\ fn leak(mut self) -> fn() -> int {
        \\  g := fn() -> int { self.n + 1 }
        \\  g
        \\ }
        \\ fn plain(self) -> fn() -> int {
        \\  fn() -> int { self.n }
        \\ }
        \\}
        \\fn main() -> int {
        \\ c := C { n: 1 }
        \\ h := c.plain()
        \\ return c.get() + h()
        \\}
        \\
    ;
    var c = try checkSource(src);
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    const d = c.result.diags[0];
    try testing.expectEqual(codes.Code.T0052, d.code);
    try testing.expectEqualStrings("a closure that escapes cannot capture the 'mut self' receiver 'self'; copy it to a local first", d.message);
    try testing.expectEqual(@as(u32, @intCast(std.mem.indexOf(u8, src, "self.n + 1").?)), d.byte_offset);
}

test "a closure or a bound method that escapes cannot capture a mut param" {
    const gpa = testing.allocator;
    const pre =
        \\struct P { x: int }
        \\impl P { fn get(self) -> int { self.x } }
        \\struct H { f: fn() -> int }
        \\fn keep(f: fn() -> int) -> H { H { f: f } }
        \\fn rec(f: fn() -> int, n: int) -> int { if n == 0 { f() } else { rec(f, n - 1) } }
        \\fn run(f: fn() -> int) -> int { f() }
        \\fn wrap(f: fn() -> int) -> H { keep(fn() -> int { f() }) }
        \\
    ;
    const main_src = "fn main() -> int {\n a := 1\n return 0\n}\n";
    const escaping = [_][]const u8{
        "fn f(mut n: int) -> fn() -> int {\n fn() -> int { n }\n}\n",
        "fn f(mut n: int) -> fn() -> int {\n fn g() -> int { n }\n g\n}\n",
        "fn f(mut n: int) -> H {\n keep(fn() -> int { n })\n}\n",
        "fn f(mut n: int) -> H {\n H { f: fn() -> int { n } }\n}\n",
        "fn f(mut n: int) -> bool {\n g := fn() -> int { n }\n g == g\n}\n",
        "fn f(mut n: int) -> int {\n rec(fn() -> int { n }, 2)\n}\n",
        "fn f(mut n: int, h: fn(fn() -> int) -> int) -> int {\n h(fn() -> int { n })\n}\n",
        "fn f(mut n: int) -> fn() -> int {\n fn() -> int {\n  g := fn() -> int { n }\n  g()\n }\n}\n",
        "fn f(mut n: int) -> int {\n g := fn() -> int { 0 }\n g = fn() -> int { n }\n g()\n}\n",
        // The last three pin the `:=`-init and the capture rules of `Escape.factsOf`, which
        // no runtime differential sees while the escaped closure's maker is still live.
        "fn f(mut n: int) -> H {\n g := fn() -> int { n }\n h := g\n keep(h)\n}\n",
        "fn f(mut n: int) -> H {\n g := fn() -> int { n }\n keep(fn() -> int { g() })\n}\n",
        "fn f(mut n: int) -> H {\n wrap(fn() -> int { n })\n}\n",
    };
    for (escaping) |body| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, body, main_src });
        defer gpa.free(src);
        try expectOneDiag(src, .T0052, "a closure that escapes cannot capture the 'mut' parameter 'n'; copy it to a local first");
    }
    const legal = [_][]const u8{
        "fn f(mut n: int) -> int {\n g := fn() -> int { n }\n g()\n}\n",
        "fn f(mut n: int) -> int {\n fn g() -> int { n }\n g()\n}\n",
        "fn f(mut n: int) -> int {\n run(fn() -> int { n = n + 1\n n })\n}\n",
        "fn f(mut n: int) -> int {\n (fn() -> int { n })()\n}\n",
        "fn f(mut n: int) -> int {\n run(fn() -> int { run(fn() -> int { n }) })\n}\n",
    };
    for (legal) |body| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, body, main_src });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        for (c.result.diags) |d| if (d.severity == .err) {
            std.debug.print("source:\n{s}\ngot {s}: {s}\n", .{ src, @tagName(d.code), d.message });
            return error.TestUnexpectedDiagnostic;
        };
    }
    const bound_called = try std.mem.concat(gpa, u8, &.{ pre, "fn f(mut p: P) -> int {\n g := p.get\n g()\n}\n", main_src });
    defer gpa.free(bound_called);
    var bc = try checkSource(bound_called);
    defer bc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), bc.result.diags.len);
    for ([_][]const u8{
        "fn f(mut p: P) -> fn() -> int { p.get }\n",
        "fn f(mut p: P) -> H {\n g := p.get\n h := g\n keep(h)\n}\n",
    }) |body| {
        const bound = try std.mem.concat(gpa, u8, &.{ pre, body, main_src });
        defer gpa.free(bound);
        try expectOneDiag(bound, .T0052, "a closure that escapes cannot capture the 'mut' parameter 'p'; copy it to a local first");
    }
}

test "a generic fn no caller instantiates still reports a closure that escapes with a mut param" {
    const gpa = testing.allocator;
    const pre =
        \\pub protocol Each { fn each(self, f: fn()) }
        \\pub struct H { f: fn() -> int }
        \\pub fn keep(f: fn() -> int) -> H { H { f: f } }
        \\pub fn run(f: fn() -> int) -> int { f() }
        \\pub fn grun[U](f: fn() -> U) -> U { f() }
        \\
    ;
    const main_src = "fn main() -> int { return 0 }\n";
    // What escapes in the template's own text escapes in every instance.
    const escaping = [_][]const u8{
        "pub fn f[T](mut n: T) -> fn() -> T {\n fn() -> T { n }\n}\n",
        "pub fn f[T](mut n: int, _t: T) -> H {\n H { f: fn() -> int { n } }\n}\n",
        "pub fn f[T](mut n: int, _t: T) -> H {\n keep(fn() -> int { n })\n}\n",
        "pub fn f[T](mut n: int, _t: T) -> fn() -> int {\n fn g() -> int { n }\n g\n}\n",
    };
    for (escaping) |body| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, body, main_src });
        defer gpa.free(src);
        try expectOnlyDiag(src, .T0052);
    }
    // A call the template cannot follow (a generic fn, a protocol method) is left to the
    // instances, so it reports nothing here.
    const legal = [_][]const u8{
        "pub fn f[T](mut n: int, _t: T) -> int {\n run(fn() -> int { n })\n}\n",
        "pub fn f[T](mut n: int, _t: T) -> int {\n grun(fn() -> int { n })\n}\n",
        "pub fn f[T has Each](mut n: int, t: T) {\n t.each(fn() { n = n + 1 })\n}\n",
        "pub fn f[T](mut n: int, _t: T) -> int {\n g := fn() -> int { n }\n g() + run(g)\n}\n",
    };
    for (legal) |body| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, body, main_src });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        for (c.result.diags) |d| if (d.severity == .err) {
            std.debug.print("source:\n{s}\ngot {s}: {s}\n", .{ src, @tagName(d.code), d.message });
            return error.TestUnexpectedDiagnostic;
        };
    }
    // An instantiated template reports the capture once, from its text and its instance.
    try expectOnlyDiag(pre ++ "fn f[T](mut n: T) -> fn() -> T {\n fn() -> T { n }\n}\nfn main() -> int {\n a := 1\n return f(a)()\n}\n", .T0052);
}

test "a captured mut param is a ref env slot, and a non-escaping capture keeps no box" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn run(f: fn()) { f() }
        \\fn f(mut n: int) {
        \\ t := 0
        \\ run(fn() {
        \\  n = n + 1
        \\  t = t + n
        \\ })
        \\}
        \\fn main() -> int {
        \\ a := 1
        \\ f(a)
        \\ return a
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.lambda_units.len);
    const u = c.result.lambda_units[0];
    try testing.expectEqual(@as(usize, 2), u.env.len);
    try testing.expectEqual(Closure.Hold.param_addr, heldAt(&c, u, u.env[0]));
    try testing.expectEqual(Closure.Hold.cell, heldAt(&c, u, u.env[1]));
    try testing.expectEqual(@as(u32, 16), u.env_size);
    const sid = u.frame_env.?;
    const l = c.result.layouts[sid];
    try testing.expectEqualStrings("env$main.f$l0", l.name);
    try testing.expectEqualSlices(Type, &.{ Type.int, Type.int }, l.field_types);
    try testing.expectEqual(@as(usize, 1), c.result.owner_closures.len);
    const oc = c.result.owner_closures[0];
    try testing.expectEqualSlices(u32, &.{1}, boxedOf(&c, oc));
    try testing.expectEqualSlices(u32, &.{1}, oc.unboxable);
    try testing.expectEqual(Closure.Hold.frame_addr, Closure.ownerStorage(c.result.bodies, c.result.owner_closures, oc.owner, true).hold(1));
}

test "a box stays when an escaping closure, a bound value or an outer escaping lambda holds it" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct C { n: int }
        \\struct H { f: fn() }
        \\impl C { fn bump(mut self) { self.n = self.n + 1 } }
        \\fn run(f: fn()) { f() }
        \\fn f(h: H) -> int {
        \\ a := 0
        \\ b := 0
        \\ c := C { n: 0 }
        \\ d := 0
        \\ run(fn() { a = a + 1 })
        \\ h.f = fn() { b = b + 1 }
        \\ run(fn() { b = b + 2 })
        \\ g := c.bump
        \\ g()
        \\ h.f = fn() { run(fn() { d = d + 1 }) }
        \\ a + b + c.n + d
        \\}
        \\fn main() -> int {
        \\ return f(H { f: fn() {} })
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const oc = c.result.owner_closures[0];
    try testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, boxedOf(&c, oc));
    try testing.expectEqualSlices(u32, &.{1}, oc.unboxable);
    const st = Closure.ownerStorage(c.result.bodies, c.result.owner_closures, oc.owner, true);
    for ([_]u32{ 2, 3, 4 }) |sl| try testing.expectEqual(Closure.Hold.cell, st.hold(sl));
    var where: std.ArrayList(u8) = .empty;
    defer where.deinit(gpa);
    for (c.result.lambda_units) |u| if (std.meta.eql(u.owner, oc.owner)) try where.print(gpa, "{s}:{c} ", .{ u.name, @as(u8, if (u.frame_env != null) 'f' else 'h') });
    try testing.expectEqualStrings("main.f$l0:f main.f$l1:h main.f$l2:f main.f$l4:f main.f$l3:h ", where.items);
}

test "linked local fns keep a frame env only if none of them escapes; their leader decides" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn g1(k: int) -> int {
        \\ x := k
        \\ fn a() -> int { if x > 0 { b() } else { 0 } }
        \\ fn b() -> int { x - 1 }
        \\ a()
        \\}
        \\fn g2(k: int) -> fn() -> int {
        \\ x := k
        \\ fn a() -> int { b() }
        \\ fn b() -> int { x }
        \\ _r := a()
        \\ b
        \\}
        \\fn g3(k: int) -> fn() -> int {
        \\ x := k
        \\ fn a() -> int { x }
        \\ fn b() -> int { 7 }
        \\ _r := a()
        \\ b
        \\}
        \\fn main() -> int {
        \\ return g1(1) + g2(2)() + g3(3)()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var where: std.ArrayList(u8) = .empty;
    defer where.deinit(gpa);
    for (c.result.lambda_units) |u| try where.print(gpa, "{s}:{c}{d} ", .{ u.name, @as(u8, if (u.frame_env != null) 'f' else 'h'), u.env_size });
    // g2's group escapes only through `b`, so its leader `a` makes a heap env. In g3, `b`
    // escapes but names no sibling, so it carries no env and `a` keeps a frame one.
    try testing.expectEqualStrings("main.g1$l0:f8 main.g1$l1:h8 main.g2$l0:h8 main.g2$l1:h8 main.g3$l0:f8 main.g3$l1:h0 ", where.items);
}

test "a fn or method with mut params is not a value (T0049)" {
    const pre =
        \\struct P { x: int }
        \\impl P {
        \\ fn add(self, mut d: int) { d = self.x }
        \\ fn grow(mut self) { self.x = self.x + 1 }
        \\}
        \\fn bump(mut n: int) { n = n + 1 }
        \\fn apply(f: fn(int), v: int) { f(v) }
        \\
    ;
    const cases = [_]struct { body: []const u8, msg: []const u8 }{
        .{ .body = "fn main() -> int {\n _g := bump\n return 0\n}\n", .msg = "'bump' is a function with 'mut' parameters and cannot be used as a value" },
        .{ .body = "fn main() -> int {\n apply(bump, 1)\n return 0\n}\n", .msg = "'bump' is a function with 'mut' parameters and cannot be used as a value" },
        .{ .body = "fn main() -> int {\n p := P { x: 1 }\n _h := p.add\n p.grow()\n return p.x\n}\n", .msg = "'add' is a method with 'mut' parameters and cannot be used as a value; wrap the call in a lambda" },
    };
    for (cases) |k| {
        const src = try std.mem.concat(testing.allocator, u8, &.{ pre, k.body });
        defer testing.allocator.free(src);
        try expectOnlyDiag(src, .T0049);
        try expectOneDiag(src, .T0049, k.msg);
    }
    // A `mut self` method with no other `mut` param still binds, and a lambda can forward
    // its own by-value param to a `mut` param.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(pre ++
        \\fn main() -> int {
        \\ p := P { x: 1 }
        \\ g := p.grow
        \\ g()
        \\ apply(fn(v: int) { bump(v) }, 2)
        \\ p.add(p.x)
        \\ return p.x
        \\}
        \\
    ));
}

test "a plain self method call on a captured local is not a mutation" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Bag { n: int }
        \\impl Bag { fn push(self, v: int) -> int { self.n + v } }
        \\fn main() -> int {
        \\ b := Bag { n: 1 }
        \\ f := fn(v: int) -> int { b.push(v) }
        \\ return f(2) + b.push(3)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "lambda units: one per boundary per lowerable owner, named by rank" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn wrap[T](x: T) -> T {
        \\ id := fn(y: T) -> T { y }
        \\ id(x)
        \\}
        \\fn main() -> int {
        \\ a := fn(x: int) -> int { x }
        \\ b := fn() -> fn() -> int { fn() { 2 } }
        \\ if wrap(true) { return a(1) + b()() + wrap(3) }
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const units = c.result.lambda_units;
    // main's three lambdas (outer-first ranks, node order), then one per wrap instance.
    try testing.expectEqual(@as(usize, 5), units.len);
    for (units[1..], units[0 .. units.len - 1]) |cur, prev| {
        try testing.expect(prev.owner < cur.owner or (prev.owner == cur.owner and prev.node.int() < cur.node.int()));
    }
    try testing.expectEqualStrings("main$l0", units[0].name);
    try testing.expectEqualStrings("main$l2", units[1].name); // the inner `fn() { 2 }` precedes its parent
    try testing.expectEqualStrings("main$l1", units[2].name);
    const bodies = c.result.bodies.items;
    try testing.expect(bodies[units[0].owner].of == .base and bodies[units[1].owner].of == .base);
    try testing.expect(bodies[units[3].owner].of == .instance and bodies[units[4].owner].of == .instance);
    for (units[3..]) |u| {
        const inst = c.result.instances[bodies[u.owner].of.instance];
        try testing.expect(std.mem.endsWith(u8, u.name, "$l0"));
        try testing.expect(std.mem.startsWith(u8, u.name, inst.name.?));
        try testing.expectEqual(@as(usize, 1), u.sig.params.len);
        try testing.expect(Type.eql(inst.args[0], u.sig.params[0]));
        try testing.expect(Type.eql(inst.args[0], u.sig.ret));
    }
    // A param binds the resolver slot the owner body gave it.
    const a_lam = try nthLambda(&c, 1);
    const a_param = Ast.protoAt(c.tree, c.tree.nodes[a_lam.int()].lhs.int()).params[0];
    try testing.expectEqual(c.resolve.resolutions[0][a_param.int()].local, units[0].param_slots[0]);
}

/// The `nth` (node order) `.local_fn` node.
fn nthLocalFn(c: *const Checked, nth: usize) !Ast.Index {
    var seen: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .local_fn) continue;
        if (seen == nth) return Ast.Index.from(@intCast(i));
        seen += 1;
    }
    return error.TestLocalFnNotFound;
}

test "a local fn is typed before its body, so it recurses" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ base := 10
        \\ fn fact(n: int) -> int { if n <= 1 { base } else { n * fact(n - 1) } }
        \\ fn noop() { }
        \\ noop()
        \\ return fact(3)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const fact = try nthLocalFn(&c, 0);
    const fact_sig = lambdaSig(&c, fact);
    try testing.expectEqual(@as(usize, 1), fact_sig.params.len);
    try testing.expect(Type.eql(Type.int, fact_sig.params[0]));
    try testing.expect(Type.eql(Type.int, fact_sig.ret));
    // A missing `-> R` is unit, not inferred.
    try testing.expect(Type.eql(Type.unit, lambdaSig(&c, try nthLocalFn(&c, 1)).ret));
    // Each is a lambda unit that knows its own name slot; `fact` captures only `base`.
    const units = c.result.lambda_units;
    try testing.expectEqual(@as(usize, 2), units.len);
    try testing.expectEqual(fact, units[0].node);
    try testing.expectEqualStrings("main$l0", units[0].name);
    try testing.expectEqual(@as(?u32, c.resolve.resolutions[0][fact.int()].local), units[0].self_slot);
    try testing.expectEqual(@as(usize, 1), units[0].env.len);
    try testing.expectEqual(@as(u32, 0), units[0].env[0].slot);
    // Adjacent, so one group, but neither names the other: `noop` keeps its own empty env.
    try testing.expectEqual(@as(u32, 0), units[1].env_size);
    try testing.expect(units[1].self_slot != null);
    try testing.expectEqual(units[0].group.lo, units[1].group.lo);
    try testing.expect(units[0].group.leader != units[1].group.leader);
    // The recursive call inside the body is a call of the fn-typed local.
    var saw_value = false;
    for (c.tree.nodes) |n| {
        if (n.tag != .call) continue;
        const callee = c.tree.nodes[n.lhs.int()];
        if (callee.tag != .identifier or !std.mem.eql(u8, c.tokens[callee.main_token].text(c.source), "fact")) continue;
        try testing.expect(c.result.callees[0][Callee.callSlot(c.tree, n).?.int()].view() == .value);
        saw_value = true;
    }
    try testing.expect(saw_value);
}

test "a group's fns are typed before any body, so the first calls the second" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ fn even(n: int) -> bool { if n == 0 { true } else { odd(n - 1) } }
        \\ fn odd(n: int) -> bool { if n == 0 { false } else { even(n - 1) } }
        \\ return if even(4) { 1 } else { 0 }
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const odd = try nthLocalFn(&c, 1);
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .call) continue;
        const callee = c.tree.nodes[n.lhs.int()];
        if (callee.tag != .identifier or !std.mem.eql(u8, c.tokens[callee.main_token].text(c.source), "odd")) continue;
        try testing.expect(Type.eql(Type.bool, c.result.node_types[0][i]));
        try testing.expect(Type.eql(c.result.node_types[0][odd.int()], c.result.node_types[0][n.lhs.int()]));
    }
}

test "a captured local fn is copied, never boxed" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\fn main() -> int {
        \\ count := 0
        \\ fn bump() { count = count + 1 }
        \\ g := fn() { bump() }
        \\ g()
        \\ return count
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    try testing.expectEqual(@as(usize, 1), c.result.owner_closures.len);
    try testing.expectEqualSlices(u32, &.{0}, boxedOf(&c, c.result.owner_closures[0]));
    const g_unit = c.result.lambda_units[1];
    try testing.expectEqual(@as(usize, 1), g_unit.env.len);
    try testing.expectEqual(@as(u32, 1), g_unit.env[0].slot);
    try testing.expectEqual(Closure.Hold.copy, heldAt(&c, g_unit, g_unit.env[0]));
    try testing.expectEqual(@as(u32, 16), g_unit.env_size);
}

test "missing return in a local fn is T0043" {
    const gpa = testing.allocator;
    {
        var c = try checkSource("fn main() -> int {\n fn f(n: int) -> int {\n  if n > 0 { return 1 }\n }\n return f(1)\n}\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0043, c.result.diags[0].code);
        try testing.expectEqualStrings("function 'f' must return int but may fall off the end", c.result.diags[0].message);
    }
    {
        // A local fn's `return` does not return from the owner, which still needs its own.
        var c = try checkSource("fn main() -> int {\n fn f() -> int { return 1 }\n _x := f()\n}\n");
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expect(std.mem.indexOf(u8, c.result.diags[0].message, "function 'main'") != null);
    }
    {
        // `return` and `?` check against the local fn's own return type.
        var c = try checkSource(
            \\fn main() -> int {
            \\ fn next(o: Option[int]) -> Option[int] {
            \\  v := o?
            \\  return .some(v + 1)
            \\ }
            \\ fn bad() -> int { return true }
            \\ return next(.some(1)).or(0) + bad()
            \\}
            \\
        );
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 1), c.result.diags.len);
        try testing.expectEqual(codes.Code.T0042, c.result.diags[0].code);
    }
    {
        // A jump out of a local fn is T0051, named as one.
        var c = try checkSource("fn main() {\n while true {\n  fn f() { break }\n  f()\n }\n}\n");
        defer c.deinit(gpa);
        try testing.expect(hasCode(&c, .T0051));
        try testing.expect(anyDiagContains(c, "cannot break a loop outside this local function"));
    }
}

/// The one diagnostic of `src`, with its code, message and the source text it spans.
fn expectSoleDiagAt(src: []const u8, code: codes.Code, message: []const u8, at: []const u8) !void {
    const gpa = testing.allocator;
    var c = try checkSource(src);
    defer c.deinit(gpa);
    errdefer for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    const d = c.result.diags[0];
    try testing.expectEqual(code, d.code);
    try testing.expectEqualStrings(message, d.message);
    const sp = d.span();
    try testing.expectEqualStrings(at, src[sp.start..sp.end]);
}

test "a negative literal into an unsigned width is one T0034 at the whole literal" {
    try expectSoleDiagAt("fn main() -> int {\n v: uint8 = -1\n return if v == 255 { 7 } else { 0 }\n}\n", .T0034, "literal out of range for type 'uint8'", "-1");
    try expectSoleDiagAt("fn id(x: uint8) -> uint8 { x }\nfn main() -> int {\n v := id(-1)\n return if v == 255 { 7 } else { 0 }\n}\n", .T0034, "literal out of range for type 'uint8'", "-1");
    try expectSoleDiagAt("fn main() -> int {\n _v: uint = -1\n return 0\n}\n", .T0034, "literal out of range for type 'uint'", "-1");
    try expectSoleDiagAt("fn main() -> int {\n _v: uint8 = -300\n return 0\n}\n", .T0034, "literal out of range for type 'uint8'", "-300");
    try expectChecksClean("fn main() -> int {\n _a: uint8 = -0\n _b: int8 = -128\n _c: uint = -0x0\n return 0\n}\n");
}

test "a constant integer operand takes a narrow sibling's width and is range-checked there once" {
    try expectChecksClean(
        \\fn main() -> int {
        \\ b: int8 = 5
        \\ b += -1
        \\ b = b + -1
        \\ b = -1 + b
        \\ b = b * -(1)
        \\ b += 1 + 2
        \\ b = b + (1 + 2)
        \\ u: uint8 = 3
        \\ u = u & ~1
        \\ _c: int8 = 1 + 2 + 3
        \\ return if b == 4 && u == 2 { 0 } else { 1 }
        \\}
        \\
    );
    try expectSoleDiagAt("fn main() -> int {\n b: int8 = 5\n b += -200\n return if b == 0 { 0 } else { 1 }\n}\n", .T0034, "literal out of range for type 'int8'", "-200");
    try expectSoleDiagAt("fn main() -> int {\n b: int8 = 5\n b = b + (100 + 100)\n return if b == 0 { 0 } else { 1 }\n}\n", .T0034, "literal out of range for type 'int8'", "100 + 100");
    try expectSoleDiagAt("fn main() -> int {\n _c: int8 = -(100 + 100)\n return 0\n}\n", .T0034, "literal out of range for type 'int8'", "100 + 100");
    try expectSoleDiagAt("fn main() -> int {\n _c: int8 = -(-128)\n return 0\n}\n", .T0034, "literal out of range for type 'int8'", "-(-128)");
    try expectSoleDiagAt("fn main() -> int {\n _d: uint8 = 1 - 2\n return 0\n}\n", .T0034, "literal out of range for type 'uint8'", "1 - 2");
    try expectSoleDiagAt("fn main() -> int {\n b: int8 = 5\n x: int = 1\n b = b + x\n return if b == 0 { 0 } else { 1 }\n}\n", .T0044, "operands of '+' must have the same type", "b + x");
}

test "a body ending in a value of the wrong type is T0042 at that value, not a missing return" {
    // A lambda held to an expected type it does not declare names the type it really has.
    try expectSoleDiagAt(
        \\fn keep(p: fn(int) -> bool) -> bool { p(1) }
        \\fn main() -> int {
        \\ if keep(fn(x) { x % 2 }) { return 1 }
        \\ return 0
        \\}
        \\
    , .T0042, "lambda has type fn(int) -> int, but fn(int) -> bool is expected: its value is int, not bool", "x % 2");
    try expectSoleDiagAt(
        \\fn main() -> int {
        \\ f: fn(int, int) -> bool = fn(a, b) { a - b }
        \\ return if f(1, 2) { 1 } else { 0 }
        \\}
        \\
    , .T0042, "lambda has type fn(int, int) -> int, but fn(int, int) -> bool is expected: its value is int, not bool", "a - b");
    // Its `return` names the expected type too: the lambda declared nothing.
    try expectSoleDiagAt(
        \\fn keep(p: fn(int) -> bool) -> bool { p(1) }
        \\fn main() -> int {
        \\ if keep(fn(x) { return x }) { return 1 }
        \\ return 0
        \\}
        \\
    , .T0042, "return type int does not match expected bool (the lambda is expected to be fn(int) -> bool)", "x");
    // A param the expected type does not have: the lambda keeps its own return type, so
    // the call's mismatch spells it.
    try expectSoleDiagAt(
        \\fn keep(p: fn(int) -> str) -> str { p(1) }
        \\fn main() -> int {
        \\ return keep(fn(_x: bool) { 1 }).len()
        \\}
        \\
    , .T0040, "argument 1: expected fn(int) -> str, got fn(bool) -> int; 'keep' takes (fn(int) -> str)", "fn(_x: bool) { 1 }");
    // A declared return type: the fn, the local fn and the annotated lambda alike.
    try expectSoleDiagAt("fn f() -> bool { 1 }\nfn main() -> int { return if f() { 1 } else { 0 } }\n", .T0042, "value int does not match declared return type bool", "1");
    try expectSoleDiagAt("fn main() -> int {\n fn g() -> bool { 2 }\n return if g() { 1 } else { 0 }\n}\n", .T0042, "value int does not match declared return type bool", "2");
    try expectSoleDiagAt("fn main() -> int {\n h := fn(x: int) -> bool { x }\n return if h(1) { 1 } else { 0 }\n}\n", .T0042, "value int does not match declared return type bool", "x");
    // A body that ends with no value still may fall off the end.
    try expectSoleDiagAt("fn f(b: bool) -> int {\n if b { return 1 }\n}\nfn main() -> int { return f(true) }\n", .T0043, "function 'f' must return int but may fall off the end", "f");
}

test "the lambda args of a rejected call add no T0050" {
    // An unknown method, a wrong arg count, and a `_` capture in each.
    try expectOnlyDiag("fn main() -> int {\n _a := 5.frob(fn(v) { v })\n return 0\n}\n", .T0018);
    try expectOnlyDiag("fn add(a: int, b: int) -> int { a + b }\nfn main() -> int {\n _a := 5.frob(add(1, _))\n return 0\n}\n", .T0018);
    try expectOnlyDiag("fn add(a: int, b: int) -> int { a + b }\nfn main() -> int {\n _a := add(fn(v) { v })\n return 0\n}\n", .T0039);
    try expectOnlyDiag(own_box_pre ++ "fn main() -> int {\n b := Box { v: 2 }\n _c := b.mapp(fn(v) { v + 1 })\n return 0\n}\n", .T0018);
}

test "a method arg that disagrees with the receiver's type is T0040, not a type-param conflict" {
    try expectSoleDiagAt(own_box_pre ++
        \\fn main() -> int {
        \\ b := Box { v: 2 }
        \\ return b.map(fn(_x: str) -> int { 1 }).get()
        \\}
        \\
    , .T0040, "argument 1: expected fn(int) -> U, got fn(str) -> int; 'map' takes (fn(int) -> U)", "fn(_x: str) -> int { 1 }");
}

test "type args on a function value or a plain fn are T0054; on a generic fn value T0049" {
    try expectSoleDiagAt("fn main() -> int {\n g := fn(v: int) { v }\n return g[int](1)\n}\n", .T0054, "'g' is not generic; drop the type arguments", "[int]");
    try expectSoleDiagAt("fn add(a: int, b: int) -> int { a + b }\nfn main() -> int {\n _f := add[int]\n return 0\n}\n", .T0054, "'add' is not generic; drop the type arguments", "[int]");
    try expectSoleDiagAt("fn id[T](x: T) -> T { x }\nfn main() -> int {\n _f := id[int]\n return 0\n}\n", .T0049, "'id' is a generic function and cannot be used as a value", "id");
}

test "a call of a module or an enum is T0055, not a silent poison" {
    try expectSoleDiagAt("enum Color { red, green }\nfn main() -> int {\n _c := Color(1)\n return 0\n}\n", .T0055, "enum 'Color' is not a function; construct one of its variants, as in 'Color.v(..)'", "Color");
}

test "main with parameters is T0073 at check time" {
    try expectSoleDiagAt("fn main(a: int, b: int) -> int { a + b }\n", .T0073, "main takes no parameters", "a: int, b: int");
}

test "a call of a function value names it in the signature suffix" {
    try expectSoleDiagAt("fn add(a: int, b: int) -> int { a + b }\nfn main() -> int {\n f := add\n return f(1)\n}\n", .T0039, "expected 2 argument(s), got 1; 'f' takes (int, int)", "(1)");
    // An expression callee would quote badly, so it keeps the bare count.
    try expectSoleDiagAt("fn add(a: int, b: int) -> int { a + b }\nfn mk() -> fn(int, int) -> int { add }\nfn main() -> int {\n return mk()(1)\n}\n", .T0039, "expected 2 argument(s), got 1", "(1)");
}

test "a pattern qualified by the wrong enum is one error, not also a missing variant" {
    try expectSoleDiagAt(
        \\enum Color { red, green }
        \\enum Size { small, big }
        \\fn f(c: Color) -> int { match c { Size.small -> 1, _ -> 0 } }
        \\fn main() -> int { return f(Color.red) }
        \\
    , .T0067, "pattern enum 'Size' does not match scrutinee 'Color'", "Size");
}

/// The `nth` (node order) value-position `field_access` spelled `member`: one no call
/// has as its member.
fn valueMember(c: *const Checked, member: []const u8, nth: usize) !Ast.Index {
    var seen: usize = 0;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag != .field_access) continue;
        if (!std.mem.eql(u8, c.tokens[n.main_token].text(c.source), member)) continue;
        const idx = Ast.Index.from(@intCast(i));
        if (isCallMember(c.tree, idx)) continue;
        if (seen == nth) return idx;
        seen += 1;
    }
    return error.TestMemberNotFound;
}

pub fn expectOneDiag(src: []const u8, code: codes.Code, message: []const u8) !void {
    const gpa = testing.allocator;
    var c = try checkSource(src);
    defer c.deinit(gpa);
    for (c.result.diags) |d| if (d.code == code and std.mem.eql(u8, d.message, message)) return;
    for (c.result.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
    return error.TestExpectedDiagnostic;
}

test "in a bounded template a generic call's diagnostics name the template's own vars apart from the call's" {
    // `ap`'s `B` binds to the template's `K` and `A` stays open; both are type var 0 unless
    // the open one sits past the template's vars.
    const pre = "protocol Add { fn add(self, o: Self) -> Self }\nfn ap[A, B](f: fn(B) -> A, b: B) -> A { f(b) }\n";
    try expectOneDiag(pre ++ "fn _go[K has Add, V](k: K, _v: V) -> K { ap(fn(_o, _p) { k }, k) }\n", .T0039, "lambda takes 2 parameter(s), but the expected type fn(K) -> A takes 1");
    try expectOneDiag(pre ++ "fn _go[K has Add, V](k: K, _v: V) -> K { ap(5, k) }\n", .T0040, "argument 1: expected fn(K) -> A, got int; 'ap' takes (fn(K) -> A, K)");
    try expectOneDiag(pre ++ "fn _go[K has Add, V](k: K, _v: V) -> K { ap(fn(_o: str) { k }, k) }\n", .T0015, "conflicting types for type parameter 'B': str vs K");
}

test "acc.plus as a value types fn(int) -> int and records bound_method" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, x: int) -> int { self.n + x } }
        \\fn main() -> int {
        \\ acc := Acc { n: 1 }
        \\ f := acc.plus
        \\ return f(2)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const fa = try valueMember(&c, "plus", 0);
    const ty = c.result.node_types[0][fa.int()];
    try testing.expect(LayoutEngine.isFnStruct(ty, c.result.layouts));
    const fs = c.result.layouts[ty.struct_id].fn_sig.?;
    try testing.expectEqual(@as(usize, 1), fs.params.len);
    try testing.expectEqual(Kind.int, fs.params[0].kind);
    try testing.expectEqual(Kind.int, fs.ret.kind);
    const mi = c.result.callees[0][fa.int()].view().bound_method;
    try testing.expectEqualStrings("plus", c.result.methods[mi].name);
    // The receiver is never assigned, so the value copies it and nothing is boxed.
    try testing.expectEqual(@as(usize, 0), c.result.owner_closures.len);
    try testing.expectEqual(@as(usize, 1), c.result.adapters.len);
    const a = c.result.adapters[0];
    try testing.expectEqual(Closure.AdapterKind.bound_copy, a.kind);
    try testing.expectEqual(mi, a.key);
    try testing.expectEqualStrings("main.Acc.plus$bound", a.name);
    try testing.expectEqualStrings("main.Acc.plus", a.target.name);
    try testing.expectEqual(@as(usize, 1), a.sig.params.len);
    try testing.expectEqualStrings("Acc", c.result.layouts[a.recv.struct_id].name);
    try testing.expect(!a.mut_self);
}

test "K.make as a value is the null-env thunk of the associated fn" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct K { v: int }
        \\impl K { fn make(v: int) -> K { K { v: v } } }
        \\fn main() -> int {
        \\ f := K.make
        \\ return f(2).v
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const fa = try valueMember(&c, "make", 0);
    const mi = c.result.callees[0][fa.int()].view().bound_method;
    try testing.expect(c.result.methods[mi].is_static);
    try testing.expectEqual(@as(usize, 1), c.result.adapters.len);
    const a = c.result.adapters[0];
    try testing.expectEqual(Closure.AdapterKind.assoc_value, a.kind);
    try testing.expectEqualStrings("main.K.make$fnval", a.name);
    try testing.expectEqualStrings("main.K.make", a.target.name);
    try testing.expectEqual(@as(usize, 1), a.sig.params.len);
}

test "S[int].mk and a bare S.mk with an expected fn type seal to one instance" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct S[T] { v: T }
        \\impl S[T] { fn mk(v: T) -> S[T] { S { v: v } } }
        \\fn main() -> int {
        \\ f := S[int].mk
        \\ g: fn(int) -> S[int] = S.mk
        \\ return f(2).v + g(3).v
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const f = c.result.callees[0][(try valueMember(&c, "mk", 0)).int()].view().bound_method;
    const g = c.result.callees[0][(try valueMember(&c, "mk", 1)).int()].view().bound_method;
    try testing.expectEqual(f, g);
    try testing.expect(c.result.methods[f].is_static and c.result.methods[f].instance != null);
    try testing.expectEqual(@as(usize, 1), c.result.adapters.len);
    try testing.expectEqual(Closure.AdapterKind.assoc_value, c.result.adapters[0].kind);
    for (c.result.callees[0]) |x| try testing.expect(!x.isPending());
}

test "a bare generic S.mk as a value with no expected fn type is T0016" {
    try expectOneDiag(
        \\struct S[T] { v: T }
        \\impl S[T] { fn mk(v: T) -> S[T] { S { v: v } } }
        \\fn main() -> int {
        \\ _f := S.mk
        \\ return 0
        \\}
        \\
    , .T0016, "cannot infer type parameter 'T' for 'S.mk'; add explicit type arguments, e.g. S[int].mk");
}

test "an associated fn with a mut param is not a value" {
    try expectOneDiag(
        \\struct K { v: int }
        \\impl K { fn add(mut t: int, x: int) { t += x } }
        \\fn main() -> int {
        \\ _f := K.add
        \\ return 0
        \\}
        \\
    , .T0049, "'add' is a function with 'mut' parameters and cannot be used as a value; wrap the call in a lambda");
}

test "an associated fn of a generic enum is called on its type; a variant keeps its name" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\enum O[T] { a(T), b(int) }
        \\impl O[T] {
        \\ fn mk(v: T) -> O[T] { O.a(v) }
        \\ fn b(v: int) -> O[T] { O.b(v + 1) }
        \\}
        \\fn main() -> int {
        \\ _x := O[int].mk(1)
        \\ _y := O.mk(true)
        \\ _z: O[int] = O.b(2)
        \\ return 0
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.W0015, c.result.diags[0].code);
    var calls: usize = 0;
    for (c.result.callees[0]) |x| switch (x.view()) {
        .method => |mi| {
            if (std.mem.eql(u8, c.result.methods[mi].name, "mk")) calls += 1;
            try testing.expect(!std.mem.eql(u8, c.result.methods[mi].name, "b"));
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 2), calls);
}

test "b.get on Box[int] seals to the instance method" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn get(self) -> T { self.v } }
        \\fn main() -> int {
        \\ b := Box[int] { v: 5 }
        \\ g := b.get
        \\ return g()
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const fa = try valueMember(&c, "get", 0);
    const mi = c.result.callees[0][fa.int()].view().bound_method;
    const ii = c.result.methods[mi].instance.?;
    // No call names `get`, so only the bound value discovered the instance.
    try testing.expectEqual(@as(usize, 1), c.result.instances.len);
    const fs = c.result.layouts[c.result.node_types[0][fa.int()].struct_id].fn_sig.?;
    try testing.expectEqual(@as(usize, 0), fs.params.len);
    try testing.expectEqual(Kind.int, fs.ret.kind);
    try testing.expectEqual(@as(usize, 1), c.result.adapters.len);
    try testing.expectEqualStrings(c.result.instances[ii].name.?, c.result.adapters[0].target.name);
    for (c.result.callees[0]) |x| try testing.expect(!x.isPending());
}

test "a mut self bound method boxes its local receiver" {
    const gpa = testing.allocator;
    const pre =
        \\struct C { n: int }
        \\impl C {
        \\ fn bump(mut self) { self.n = self.n + 1 }
        \\ fn get(self) -> int { self.n }
        \\}
        \\
    ;
    const boxes = [_][]const u8{
        // A `mut self` method writes the receiver through the value.
        "fn main() -> int {\n c := C { n: 0 }\n inc := c.bump\n inc()\n return c.n\n}\n",
        // A plain method on a receiver assigned later, or mutated by a call.
        "fn main() -> int {\n c := C { n: 0 }\n g := c.get\n c = C { n: 2 }\n return g()\n}\n",
        "fn main() -> int {\n c := C { n: 0 }\n g := c.get\n c.bump()\n return g()\n}\n",
        // The same with a lambda in the fn.
        "fn main() -> int {\n c := C { n: 0 }\n g := c.get\n h := fn() -> int { 1 }\n c = C { n: 2 }\n return g() + h()\n}\n",
        // A param receiver.
        "fn k(c: C) -> int {\n inc := c.bump\n inc()\n c.n\n}\nfn main() -> int { return k(C { n: 0 }) }\n",
    };
    for (boxes) |body| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, body });
        defer gpa.free(src);
        var c = try checkSource(src);
        defer c.deinit(gpa);
        try testing.expectEqual(@as(usize, 0), c.result.diags.len);
        const fa = c.tree.nodes[(try valueMember(&c, if (std.mem.indexOf(u8, body, "c.bump\n") != null) "bump" else "get", 0)).int()];
        const slot = c.resolve.resolutions[0][fa.lhs.int()].local;
        try testing.expectEqual(@as(usize, 1), c.result.owner_closures.len);
        try testing.expectEqualSlices(u32, &.{slot}, boxedOf(&c, c.result.owner_closures[0]));
        try testing.expectEqual(@as(usize, 1), c.result.adapters.len);
        try testing.expectEqual(Closure.AdapterKind.bound_box, c.result.adapters[0].kind);
        try testing.expect(std.mem.endsWith(u8, c.result.adapters[0].name, "$boundbox"));
    }
    var bumped = try checkSource(pre ++ "fn main() -> int {\n c := C { n: 0 }\n inc := c.bump\n inc()\n return c.n\n}\n");
    defer bumped.deinit(gpa);
    try testing.expect(bumped.result.adapters[0].mut_self);
    // A receiver no one writes stays a copy.
    var ro = try checkSource(pre ++ "fn main() -> int {\n c := C { n: 3 }\n g := c.get\n return g() + c.n\n}\n");
    defer ro.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ro.result.owner_closures.len);
    try testing.expectEqual(Closure.AdapterKind.bound_copy, ro.result.adapters[0].kind);
}

test "mut self bound on a temporary is T0019, on a field T0072" {
    const pre =
        \\struct C { n: int }
        \\struct H { c: C }
        \\impl C {
        \\ fn bump(mut self) { self.n = self.n + 1 }
        \\ fn get(self) -> int { self.n }
        \\}
        \\fn mk() -> C { C { n: 0 } }
        \\
    ;
    try expectOneDiag(pre ++ "fn main() -> int {\n inc := mk().bump\n inc()\n return 0\n}\n", .T0019, "cannot bind mutating method 'bump' to a temporary; the receiver must be a local variable");
    try expectOneDiag(pre ++ "fn main() -> int {\n h := H { c: C { n: 0 } }\n inc := h.c.bump\n inc()\n return h.c.n\n}\n", .T0072, "binding the 'mut self' method 'bump' needs a local receiver, not a field; bind it on a local variable");
    // A bound method holds a `mut self` receiver by address, so it must not escape.
    const gpa = testing.allocator;
    var called = try checkSource(pre ++ "impl H {\n fn peek(mut self) -> int {\n  g := self.total\n  g()\n }\n fn total(self) -> int { self.c.n }\n}\nfn main() -> int {\n h := H { c: C { n: 0 } }\n return h.peek()\n}\n");
    defer called.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), called.result.diags.len);
    try expectOneDiag(pre ++ "impl H {\n fn peek(mut self) -> fn() -> int { self.total }\n fn total(self) -> int { self.c.n }\n}\nfn main() -> int {\n h := H { c: C { n: 0 } }\n g := h.peek()\n return g()\n}\n", .T0052, "a closure that escapes cannot capture the 'mut self' receiver 'self'; copy it to a local first");
    // A plain method on a field or a temporary copies the receiver once: no error.
    var ok = try checkSource(pre ++ "fn main() -> int {\n h := H { c: C { n: 1 } }\n g := h.c.get\n t := mk().get\n return g() + t()\n}\n");
    defer ok.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), ok.result.diags.len);
    try testing.expectEqual(@as(usize, 0), ok.result.owner_closures.len);
}

test "a builtin method as a value is T0049" {
    const msg = "'len' is a built-in method and cannot be used as a value; wrap the call in a lambda";
    try expectOneDiag("fn main() -> int {\n s := \"hi\"\n f := s.len\n return f()\n}\n", .T0049, msg);
    try expectOneDiag("fn main() -> int {\n n := 1\n f := n.eq\n return if f(1) { 1 } else { 0 }\n}\n", .T0049, "'eq' is a built-in method and cannot be used as a value; wrap the call in a lambda");
    try expectOneDiag("fn main() -> int {\n o: Option[int] = .some(1)\n f := o.expect\n return f()\n}\n", .T0049, "'expect' is a built-in method and cannot be used as a value; wrap the call in a lambda");
    try expectOneDiag("struct P { x: int }\nfn main() -> int {\n p := P { x: 1 }\n f := p.to_string\n return 0\n}\n", .T0049, "'to_string' is a built-in method and cannot be used as a value; wrap the call in a lambda");
    try expectOneDiag(
        \\protocol Shape { fn area(self) -> int }
        \\struct Sq { s: int }
        \\impl Sq has Shape { fn area(self) -> int { self.s * self.s } }
        \\fn first[T has Shape](x: T) -> int {
        \\ f := x.area
        \\ f()
        \\}
        \\fn main() -> int { return first(Sq { s: 2 }) }
        \\
    , .T0049, "'area' is a protocol method of a type parameter and cannot be used as a value; wrap the call in a lambda");
}

test "a static method as a value is T0018" {
    const msg = "'make' is an associated function, not a method; it has no receiver to bind";
    try expectOneDiag(
        \\struct Acc { n: int }
        \\impl Acc { fn make() -> Acc { Acc { n: 0 } } }
        \\fn main() -> int {
        \\ a := Acc { n: 1 }
        \\ f := a.make
        \\ return 0
        \\}
        \\
    , .T0018, msg);
    try expectOneDiag(
        \\struct Box[T] { v: T }
        \\impl Box[T] { fn make(v: T) -> Box[T] { Box[T] { v: v } } }
        \\fn main() -> int {
        \\ b := Box[int] { v: 1 }
        \\ f := b.make
        \\ return 0
        \\}
        \\
    , .T0018, msg);
}

test "a field wins over a method in value position" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct S { run: fn(int) -> int }
        \\impl S { fn run(self, x: int) -> int { x + 100 } }
        \\fn dbl(x: int) -> int { x * 2 }
        \\fn main() -> int {
        \\ s := S { run: dbl }
        \\ f := s.run
        \\ return f(1) + s.run(1)
        \\}
        \\
    );
    defer c.deinit(gpa);
    // The documented rule, which W0013 points out at the method.
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqual(codes.Code.W0013, c.result.diags[0].code);
    const fa = try valueMember(&c, "run", 0);
    try testing.expectEqual(Callee.View.none, c.result.callees[0][fa.int()].view());
    for (c.result.adapters) |a| try testing.expectEqual(Closure.AdapterKind.fn_value, a.kind);
    // An assignment target names a field, never a bound method.
    try expectOneDiag(
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, x: int) -> int { self.n + x } }
        \\fn one(x: int) -> int { x }
        \\fn main() -> int {
        \\ a := Acc { n: 1 }
        \\ a.plus = one
        \\ return a.n
        \\}
        \\
    , .T0046, "no field 'plus' in struct 'Acc'");
    // Only the target's own member: the rest of the target, a call argument or a lambda
    // body in it, still binds. A field of a call result is no place (T0075), but the
    // target is still checked.
    var inner = try checkSource(
        \\struct Acc { n: int }
        \\struct P { x: int }
        \\impl Acc { fn plus(self, x: int) -> int { self.n + x } }
        \\impl P { fn me(self, f: fn(int) -> int) -> P { P { x: f(self.x) } } }
        \\fn main() -> int {
        \\ acc := Acc { n: 1 }
        \\ p := P { x: 1 }
        \\ p.me(acc.plus).x = 7
        \\ p.me(fn(v: int) -> int {
        \\  g := acc.plus
        \\  g(v)
        \\ }).x = 8
        \\ return p.x
        \\}
        \\
    );
    defer inner.deinit(gpa);
    try testing.expectEqual(@as(usize, 2), inner.result.diags.len);
    for (inner.result.diags) |d| try testing.expectEqual(codes.Code.T0075, d.code);
    for (0..2) |k| {
        const m = try valueMember(&inner, "plus", k);
        try testing.expect(inner.result.callees[0][m.int()].view() == .bound_method);
    }
}

test "past the owner's frame-env budget a non-escaping closure keeps a heap env" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Big { a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int, j: int, k: int, l: int, m: int, n: int, o: int, p: int, q: int, r: int, s: int, t: int }
        \\fn run(f: fn() -> int) -> int { f() }
        \\fn main() -> int {
        \\ x := Big { a: 1, b: 0, c: 0, d: 0, e: 0, f: 0, g: 0, h: 0, i: 0, j: 0, k: 0, l: 0, m: 0, n: 0, o: 0, p: 0, q: 0, r: 0, s: 0, t: 0 }
        \\ y := 2
        \\ return run(fn() -> int { x.a }) + run(fn() -> int { x.a + 1 }) + run(fn() -> int { y })
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var got: [3]bool = undefined;
    for (c.result.lambda_units, &got) |u, *g| g.* = u.frame_env != null;
    // 160 + 160 > 256: the second goes to the heap; the 8-byte third still fits.
    try testing.expectEqualSlices(bool, &.{ true, false, true }, &got);
}

test "a captured-and-mutated local over 16 bytes stays boxed" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Acc { a: int, b: int, n: int }
        \\fn run(f: fn()) { f() }
        \\fn main() -> int {
        \\ acc := Acc { a: 0, b: 0, n: 0 }
        \\ t := 0
        \\ run(fn() {
        \\  acc.n = acc.n + 1
        \\  t = t + 2
        \\ })
        \\ return acc.n + t
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    const oc = c.result.owner_closures[0];
    try testing.expectEqualSlices(u32, &.{ 0, 1 }, boxedOf(&c, oc));
    try testing.expectEqualSlices(u32, &.{1}, oc.unboxable);
    try testing.expectEqual(Closure.Hold.cell, Closure.ownerStorage(c.result.bodies, c.result.owner_closures, oc.owner, true).hold(0));
    try testing.expect(c.result.lambda_units[0].frame_env != null);
}

test "a bound method on a mut param gets the $boundref adapter; on a boxed local still $boundbox" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct C { n: int }
        \\impl C {
        \\ fn bump(mut self) { self.n = self.n + 1 }
        \\ fn get(self) -> int { self.n }
        \\}
        \\fn by_ref(mut p: C) -> int {
        \\ g := p.get
        \\ p.n = 3
        \\ g()
        \\}
        \\fn main() -> int {
        \\ c := C { n: 0 }
        \\ inc := c.bump
        \\ inc()
        \\ return by_ref(c)
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    var kinds: std.ArrayList(u8) = .empty;
    defer kinds.deinit(gpa);
    for (c.result.adapters) |a| try kinds.print(gpa, "{s}:{s} ", .{ @tagName(a.kind), a.name });
    try testing.expectEqualStrings("bound_box:main.C.bump$boundbox bound_ref:main.C.get$boundref ", kinds.items);
}
