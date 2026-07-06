//! The front-of-pipeline user type/protocol registration phase, extracted from the
//! `Typecheck` mega-struct as free functions over `*Typecheck` (Zig has no struct-field
//! privacy, so these read the checker's fields directly). `registerStructs`/`registerEnums`/
//! `registerProtocols` populate the global id spaces in module-then-decl append order;
//! `decodeTemplateFields`/`decodeTemplateVariants` decode each generic template's fields/
//! variants as `type_var` PATTERNS (the input to `substReify`). Driven from `runGraph`'s
//! Phase 0.

const Ast = @import("../ast/Ast.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Typecheck = @import("../types.zig");
const Type = Typecheck.Type;
const VariantForm = Typecheck.VariantForm;
const VariantSym = LayoutEngine.VariantSym;
const type_names = Typecheck.type_names;

/// Register the struct decls among `decl_nodes` (of the currently-active tree).
/// `mod` is the owning module id (0 single-file). Global ids are assigned in
/// append order; per-module duplicate/shadow diagnostics mirror the single-file
/// rules. The bare name → global id binding goes into the active struct map
/// (`activeStructMap`, the current module's table in the ctx).
pub fn registerStructs(t: *Typecheck, decl_nodes: []const Ast.Index, mod: u32) !void {
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
        // `decl.rhs`. Collect the ordered param NAMES so template field-type refs
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

/// Phase 0a: decode each generic struct TEMPLATE's field types as PATTERNS into
/// its `field_names`/`field_types`, with the template's generic params in scope so a
/// `T`-spelled ref decodes to `type_var(ord)` and a `Box[T]` field to
/// `App(Box, [type_var 0])`. These patterns are the input to `substReify`, which
/// grounds them per reified instance. The template itself is never a value type, so
/// it is marked `.done` (size 0) and Phase 0b skips laying it out.
pub fn decodeTemplateFields(t: *Typecheck) !void {
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

/// Phase 0a: decode each generic ENUM TEMPLATE's variants into `VariantSym`s whose
/// payload `field_types` are PATTERNS (`left(L)` -> `type_var(0)`; `w(Box[T])` ->
/// `App(Box,[type_var 0])`), with the template's generic params in scope so an
/// `L`-spelled ref decodes to `type_var(ord)`. These patterns are the input to
/// `substReify`, which grounds them per reified instance in `reifyAppToEnum`. The
/// template is never a value type, so it is marked `.done` (size 0) and Phase 0b skips
/// laying it out. Mirrors `decodeTemplateFields` + the variant decode of `layoutEnum`.
pub fn decodeTemplateVariants(t: *Typecheck) !void {
    for (0..t.enums.items.len) |id| {
        if (!t.enums.items[id].is_generic) continue;
        // The native prelude generic enums (Option/Result) are AST-less and already
        // variant-decoded in `registerPrelude`; skip the AST deref (else OOB on Ast.none).
        if (t.enums.items[id].decl_node == Ast.none) continue;
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
pub fn registerEnums(t: *Typecheck, decl_nodes: []const Ast.Index, mod: u32) !void {
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
        // `decl.rhs` (mirror registerStructs). Collect the ordered param NAMES so
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

/// Phase 0c: register the `protocol` decls among `decl_nodes` (of the active
/// tree) into ONE global id space. Global ids are assigned in append order
/// (module-then-decl); the bare name → global id binding goes into the active protocol
/// map (this module's table). Each protocol's required method NAMES are collected
/// (declaration order) for the coherence completeness check. A same-module duplicate
/// `protocol P` is first-wins (silent); cross-module same-named protocols get DISTINCT
/// ids (nominal distinctness, like structs/enums).
pub fn registerProtocols(t: *Typecheck, decl_nodes: []const Ast.Index, mod: u32) !void {
    for (decl_nodes) |decl_idx| {
        const decl = t.tree.nodes[decl_idx.int()];
        if (decl.tag != .protocol_decl) continue;
        const name = t.nameText(decl.main_token);
        if (t.activeProtocolMap().get(name) != null) continue; // first-wins duplicate
        // Required method names (declaration order); the sigs live as a Range in `lhs`.
        // Once appended, the `ProtocolSym` owns `names` (freed at teardown), so no
        // `errdefer` here — mirroring `registerStructs`/`registerEnums` (an OOM on the
        // append below aborts the compile anyway).
        const sig_nodes = Ast.rangeSlice(t.tree, decl.lhs.int());
        const names = try t.gpa.alloc([]const u8, sig_nodes.len);
        for (sig_nodes, 0..) |snode, i| names[i] = t.nameText(t.tree.nodes[snode.int()].main_token);
        // Generic type-params: `protocol Into[U]` -> `["U"]`. Stored un-offset on
        // the `ProtocolSym` (for arity checks); the OUTER array is owned by `t.protocols`.
        const gp_nodes = Ast.protocolGenericParams(t.tree, decl_idx);
        const generic_params = try t.gpa.alloc([]const u8, gp_nodes.len);
        for (gp_nodes, 0..) |gp, i| generic_params[i] = t.nameText(t.tree.nodes[gp.int()].main_token);
        // Decode each method SIGNATURE, Self- and protocol-param-aware. Force
        // the synthetic `self` slot to `Type.typeVar(0)` (decoding it would resolve the
        // protocol-name type-ref and spuriously T0001); a `Self`-typed later param resolves
        // to `type_var(0)` via the `cur_self_type` hook. A protocol generic param resolves
        // to `type_var(1..)` via a Self-OFFSET `cur_generic_params` (index 0 is a `""`
        // placeholder that never matches a source ident, reserving ordinal 0 for `Self`),
        // so `-> U` decodes to `type_var(1)` and never aliases the receiver. Structs/enums
        // are already registered (Phase 0), so a struct/enum param type resolves.
        const method_params = try t.gpa.alloc([]const Type, sig_nodes.len);
        const method_rets = try t.gpa.alloc(Type, sig_nodes.len);
        const prev_self = t.cur_self_type;
        t.cur_self_type = Type.typeVar(0);
        defer t.cur_self_type = prev_self;
        const offset_params = try t.gpa.alloc([]const u8, generic_params.len + 1);
        defer t.gpa.free(offset_params);
        offset_params[0] = ""; // ordinal 0 = Self, never matched by a source identifier
        for (generic_params, 0..) |gp, i| offset_params[i + 1] = gp;
        const prev_gp = t.cur_generic_params;
        t.cur_generic_params = offset_params;
        defer t.cur_generic_params = prev_gp;
        for (sig_nodes, 0..) |snode, i| {
            const mp = Ast.protoAt(t.tree, t.tree.nodes[snode.int()].lhs.int());
            const pars = try t.gpa.alloc(Type, mp.params.len);
            for (mp.params, 0..) |param_idx, j| {
                if (j == 0) {
                    pars[j] = Type.typeVar(0); // synthetic `self`
                } else {
                    pars[j] = t.typeFromNode(t.tree.nodes[param_idx.int()].lhs);
                }
            }
            method_params[i] = pars;
            method_rets[i] = if (mp.ret_type == Ast.none) Type.unit else t.typeFromNode(mp.ret_type);
        }
        const id: u32 = @intCast(t.protocols.items.len);
        try t.protocols.append(t.gpa, .{
            .name = name,
            .mod = mod,
            .pub_export = t.tree.isPub(decl_idx),
            .decl_node = decl_idx,
            .generic_params = generic_params,
            .methods = names,
            .method_params = method_params,
            .method_rets = method_rets,
        });
        try t.activeProtocolMap().put(t.gpa, name, id);
    }
}
