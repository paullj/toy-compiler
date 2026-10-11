//! What a token IS, rendered for a hover: a callable's signature (with parameter names), a
//! binding as `name: Type`, a type's declaration as written, a field, or an enum variant.
//! Declarations render from their source text (generic parameters and all); uses render
//! from the checked, concrete types.

const std = @import("std");
const toyc = @import("toy_compiler");
const render = @import("render.zig");
const Analysis = @import("Analysis.zig");

const Ast = toyc.Ast;
const Graph = toyc.Graph;
const Type = toyc.Typecheck.Type;
const Layout = toyc.Typecheck.Layout;
const EnumLayout = toyc.Typecheck.EnumLayout;

const bareName = render.bareName;
const isRenderable = render.isRenderable;
const renderNominal = render.renderNominal;
const renderType = render.renderType;
const unmangled = render.unmangled;

/// What the token under the cursor IS, in the entry module: a callable's signature
/// (with parameter names), a binding as `name: Type`, a type's definition, a field, or a
/// variant — falling back to the bare type of the expression the token heads.
pub const Describer = struct {
    a: std.mem.Allocator,
    an: *const Analysis,
    buf: std.ArrayList(u8) = .empty,

    fn module(d: *const Describer) *const Graph.Module {
        return d.an.module();
    }

    fn tc(d: *const Describer) *const toyc.TypecheckGraph.GraphResult {
        return d.an.tc();
    }

    fn text(d: *const Describer, tok: u32) []const u8 {
        const t = d.module().tokens[tok];
        return d.module().source[t.start..t.end];
    }

    pub fn describe(d: *Describer, tok: u32) !bool {
        const m = d.module();
        const resolutions = d.an.resolutions();
        const node_types = d.an.nodeTypes();

        // func-first: a token shared by an identifier and its enclosing node (a callee and
        // its `call`, a `mod.fn` access and its receiver) should show the callable.
        var ty: ?Type = null;
        var local = false;
        var decl: ?u32 = null;
        for (d.an.nodesAt(tok)) |i| {
            const n = m.nodes[i];
            if (resolutions[i] == .func) {
                // A call of a generic fn shows the instance it runs.
                if (d.an.freeCallee(d.an.callSlotOf(i))) |fc| if (fc == .instance) if (try d.instance(fc.instance)) return true;
                if (try d.function(resolutions[i].func)) return true;
            }
            // Before the variant fallback: a method never shows as a same-named variant.
            if (n.tag == .field_access and resolutions[i] != .func) if (try d.callee(i)) return true;
            if (ty == null and isRenderable(node_types[i])) ty = node_types[i];
            switch (n.tag) {
                .identifier => if (resolutions[i] == .local) {
                    local = true;
                },
                .var_decl, .pattern_binding, .for_stmt, .for_in_stmt, .for_in2_stmt, .local_fn => local = true,
                .fn_decl, .param, .field_init, .struct_decl, .enum_decl, .protocol_decl, .enum_variant_unit, .enum_variant_tuple, .enum_variant_struct, .enum_init_unit, .enum_init_tuple, .enum_init_struct, .pattern_variant => {
                    if (decl == null) decl = i;
                },
                else => {},
            }
        }

        if (decl) |i| if (try d.declaration(i, tok, ty)) return true;
        if (ty) |t| {
            if (local) try d.buf.print(d.a, "{s}: ", .{d.text(tok)});
            try renderType(d.a, &d.buf, t, d.tc().layouts, d.tc().enum_layouts);
            return true;
        }
        if (try d.qualifiedVariant(tok)) return true;
        // A type name is left unresolved by the resolver; match it against the type tables.
        return d.typeDefinition(d.text(tok));
    }

    /// `Enum.variant` spelled through the type name (`Option.some(x)`) is a field access,
    /// not a variant node; the call it heads carries the concrete enum type.
    fn qualifiedVariant(d: *Describer, tok: u32) !bool {
        const m = d.module();
        const node_types = d.an.nodeTypes();
        for (d.an.nodesAt(tok)) |i| {
            if (m.nodes[i].tag != .field_access) continue;
            if (isRenderable(node_types[i])) return d.variant(i, node_types[i]);
            for (m.nodes, 0..) |call, c| {
                if (call.tag == .call and call.lhs.int() == i) return d.variant(i, node_types[c]);
            }
        }
        return false;
    }

    fn declaration(d: *Describer, i: u32, tok: u32, ty: ?Type) !bool {
        const m = d.module();
        const n = m.nodes[i];
        switch (n.tag) {
            .fn_decl => {
                if (d.an.gidOf(i)) |f| return d.function(f);
                // A protocol's method signature never enters the fn table: show it as written.
                try d.appendNodeText(m, i);
                return true;
            },
            .param => {
                const t = ty orelse d.paramType(i);
                if (t != null and isRenderable(t.?)) {
                    try d.buf.print(d.a, "{s}: ", .{d.text(tok)});
                    try renderType(d.a, &d.buf, t.?, d.tc().layouts, d.tc().enum_layouts);
                    return true;
                }
                // A generic template's field (`left: A`) has no concrete type: show it as written.
                try d.appendNodeText(m, i);
                return true;
            },
            .field_init => {
                const t = d.an.nodeTypes()[n.lhs.int()];
                if (!isRenderable(t)) return false;
                try d.buf.print(d.a, "{s}: ", .{d.text(tok)});
                try renderType(d.a, &d.buf, t, d.tc().layouts, d.tc().enum_layouts);
                return true;
            },
            .struct_decl, .enum_decl, .protocol_decl => return d.typeDefinition(d.text(tok)),
            else => return d.variant(i, ty),
        }
    }

    /// A struct field's or a fn parameter's declared type, found through its owner.
    fn paramType(d: *const Describer, i: u32) ?Type {
        const m = d.module();
        const tree = m.tree();
        for (m.nodes, 0..) |owner, o| switch (owner.tag) {
            .fn_decl => {
                const proto = Ast.protoAt(tree, owner.lhs.int());
                for (proto.params, 0..) |p, k| if (p.int() == i) {
                    const f = d.an.gidOf(@intCast(o)) orelse return null;
                    const sig = d.tc().sigs[f];
                    return if (k < sig.params.len) sig.params[k] else null;
                };
            },
            .struct_decl => for (Ast.rangeSlice(tree, owner.lhs.int())) |p| if (p.int() == i) {
                const l = d.structNamed(d.text(owner.main_token)) orelse return null;
                return fieldType(l.field_names, l.field_types, d.text(m.nodes[i].main_token));
            },
            else => {},
        };
        return null;
    }

    /// Fn `f` as `fn name(p: P, ..) -> R`. A generic fn's checked types are opaque, so it
    /// shows as its declaration spells it.
    fn function(d: *Describer, f: u32) !bool {
        if (f >= d.tc().sigs.len) return false;
        const sig = d.tc().sigs[f];
        const name = unmangled(bareName(sig.name));
        if (sig.isGenericTemplate()) if (d.fnDecl(f)) |decl| {
            try render.appendFnHeader(d.a, &d.buf, decl.module, decl.node, name);
            return true;
        };
        return d.signature(name, try d.paramNames(f), sig.params, sig.mut_mask, sig.ret);
    }

    /// The instance a call of a generic fn runs, with its concrete types.
    fn instance(d: *Describer, ii: u32) !bool {
        const t = d.tc();
        if (ii >= t.instances.len) return false;
        const inst = t.instances[ii];
        if (inst.template_gid >= t.sigs.len) return false;
        const sig = t.sigs[inst.template_gid];
        return d.signature(unmangled(bareName(sig.name)), try d.paramNames(inst.template_gid), inst.params, sig.mut_mask, inst.ret);
    }

    fn signature(d: *Describer, name: []const u8, names: []const []const u8, params: []const Type, mut_mask: u64, ret: Type) !bool {
        try d.buf.print(d.a, "fn {s}(", .{name});
        for (params, 0..) |p, k| {
            if (k != 0) try d.buf.appendSlice(d.a, ", ");
            if (toyc.Modes.isMut(mut_mask, k)) try d.buf.appendSlice(d.a, "mut ");
            if (k < names.len) try d.buf.print(d.a, "{s}: ", .{names[k]});
            try renderType(d.a, &d.buf, p, d.tc().layouts, d.tc().enum_layouts);
        }
        try d.buf.appendSlice(d.a, ") -> ");
        try renderType(d.a, &d.buf, ret, d.tc().layouts, d.tc().enum_layouts);
        return true;
    }

    /// Fn `f`'s declaration, when it has source.
    fn fnDecl(d: *const Describer, f: u32) ?struct { module: *const Graph.Module, node: u32 } {
        const fns = d.an.res().fns;
        if (f >= fns.len or fns[f].decl_node == Ast.none) return null;
        return .{ .module = &d.an.graph.modules[fns[f].module], .node = fns[f].decl_node.int() };
    }

    /// The method the call headed by member node `fa` selected, or the bound method value
    /// `fa` binds, as `function` renders a fn. A user protocol's method shows as written,
    /// like the protocol-decl hover.
    fn callee(d: *Describer, fa: u32) !bool {
        const sig = (try d.an.calleeAt(d.a, fa)) orelse return false;
        if (d.an.spelledFromDecl(fa)) if (sig.decl) |decl| {
            const dm = &d.an.graph.modules[decl.module];
            if (d.an.callees()[fa].templateFn() != null) try render.appendFnHeader(d.a, &d.buf, dm, decl.node, sig.name) else try d.appendNodeText(dm, decl.node);
            return true;
        };
        try d.buf.print(d.a, "fn {s}(", .{sig.name});
        for (sig.params, 0..) |p, k| {
            if (k != 0) try d.buf.appendSlice(d.a, ", ");
            if (toyc.Modes.isMut(sig.mut_mask, k)) try d.buf.appendSlice(d.a, "mut ");
            // A receiver inside a bounded template has no concrete type (its node types are
            // scratch): the name alone beats a `?`.
            if (k < sig.param_names.len) {
                try d.buf.appendSlice(d.a, sig.param_names[k]);
                if (!render.isRenderableNamed(p, sig.var_names)) continue;
                try d.buf.appendSlice(d.a, ": ");
            }
            try render.renderTypeNamed(d.a, &d.buf, p, d.tc().layouts, d.tc().enum_layouts, sig.var_names);
        }
        try d.buf.appendSlice(d.a, ") -> ");
        try render.renderTypeNamed(d.a, &d.buf, sig.ret, d.tc().layouts, d.tc().enum_layouts, sig.var_names);
        return true;
    }

    /// Parameter names from `f`'s declaration, in whatever module declares it; empty for a
    /// builtin (no source) so the signature falls back to bare types.
    fn paramNames(d: *const Describer, f: u32) ![]const []const u8 {
        const fns = d.an.res().fns;
        if (f >= fns.len) return &.{};
        return d.an.declParamNames(d.a, fns[f].module, fns[f].decl_node);
    }

    fn structNamed(d: *const Describer, name: []const u8) ?*const Layout {
        for (d.tc().layouts) |*l| if (std.mem.eql(u8, bareName(l.name), name)) return l;
        return null;
    }

    fn enumNamed(d: *const Describer, name: []const u8) ?*const EnumLayout {
        for (d.tc().enum_layouts) |*e| if (std.mem.eql(u8, bareName(e.name), name)) return e;
        return null;
    }

    /// A type's declaration as written (generic parameters and all), found in whichever
    /// module declares it; the entry module wins a name clash.
    fn typeDefinition(d: *Describer, name: []const u8) !bool {
        const td = d.an.typeDecl(name) orelse {
            for (native_enums) |e| if (std.mem.eql(u8, e.name, name)) {
                try d.buf.appendSlice(d.a, e.decl);
                return true;
            };
            return false;
        };
        try render.appendNodeText(d.a, &d.buf, &d.an.graph.modules[td.module], td.node);
        return true;
    }

    fn appendNodeText(d: *Describer, m: *const Graph.Module, i: u32) !void {
        try render.appendNodeText(d.a, &d.buf, m, i);
    }

    /// A variant as `Enum.Variant(payload)`. The enum is the node's own type (a construction
    /// or pattern), else the one enum that declares a variant of this name.
    fn variant(d: *Describer, i: u32, ty: ?Type) !bool {
        const name = d.text(d.module().nodes[i].main_token);
        const e: *const EnumLayout = blk: {
            if (ty) |t| if (t.kind == .@"enum" and t.enum_id < d.tc().enum_layouts.len) break :blk &d.tc().enum_layouts[t.enum_id];
            var found: ?*const EnumLayout = null;
            for (d.tc().enum_layouts) |*el| for (el.variants) |v| if (std.mem.eql(u8, v.name, name)) {
                if (found != null) return false; // ambiguous
                found = el;
            };
            break :blk found orelse return false;
        };
        for (e.variants) |v| if (std.mem.eql(u8, v.name, name)) {
            try renderNominal(d.a, &d.buf, e.name, d.tc().layouts, d.tc().enum_layouts);
            try d.buf.append(d.a, '.');
            try d.variantShape(v);
            return true;
        };
        return false;
    }

    fn variantShape(d: *Describer, v: toyc.Typecheck.VariantLayout) !void {
        try d.buf.appendSlice(d.a, v.name);
        switch (v.form) {
            .unit => {},
            .tuple => {
                try d.buf.append(d.a, '(');
                for (v.field_types, 0..) |t, k| {
                    if (k != 0) try d.buf.appendSlice(d.a, ", ");
                    try renderType(d.a, &d.buf, t, d.tc().layouts, d.tc().enum_layouts);
                }
                try d.buf.append(d.a, ')');
            },
            .@"struct" => {
                try d.buf.appendSlice(d.a, " { ");
                try d.fields(v.field_names, v.field_types);
                try d.buf.appendSlice(d.a, " }");
            },
        }
    }

    fn fields(d: *Describer, names: []const []const u8, types: []const Type) !void {
        for (names, types, 0..) |n, t, k| {
            if (k != 0) try d.buf.appendSlice(d.a, ", ");
            try d.buf.print(d.a, "{s}: ", .{n});
            try renderType(d.a, &d.buf, t, d.tc().layouts, d.tc().enum_layouts);
        }
    }
};

/// The prelude enums the compiler registers natively (see `Prelude.registerOptionResult`):
/// no module declares them, so there is no source to show.
const native_enums = [_]struct { name: []const u8, decl: []const u8 }{
    .{ .name = "Option", .decl = "enum Option[T] { some(T), none }" },
    .{ .name = "Result", .decl = "enum Result[T, E] { ok(T), err(E) }" },
};

fn fieldType(names: []const []const u8, types: []const Type, name: []const u8) ?Type {
    for (names, types) |n, t| if (std.mem.eql(u8, n, name)) return t;
    return null;
}
