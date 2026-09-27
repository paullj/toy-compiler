//! What a token IS, rendered for a hover: a callable's signature (with parameter names), a
//! binding as `name: Type`, a type's declaration as written, a field, or an enum variant.
//! Declarations render from their source text (generic parameters and all); uses render
//! from the checked, concrete types.

const std = @import("std");
const toyc = @import("toy_compiler");
const render = @import("render.zig");

const Ast = toyc.Ast;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Token = toyc.Token;
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
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    tc: *const TypecheckGraph.GraphResult,
    buf: std.ArrayList(u8) = .empty,

    fn module(d: *const Describer) *const Graph.Module {
        return d.graph.entry();
    }

    fn text(d: *const Describer, tok: u32) []const u8 {
        const t = d.module().tokens[tok];
        return d.module().source[t.start..t.end];
    }

    pub fn describe(d: *Describer, tok: u32) !bool {
        const m = d.module();
        const entry = d.graph.entry_index;
        const resolutions = d.res.resolutions[entry];
        const node_types = d.tc.node_types[entry];

        // func-first: a token shared by an identifier and its enclosing node (a callee and
        // its `call`, a `mod.fn` access and its receiver) should show the callable.
        var ty: ?Type = null;
        var local = false;
        var decl: ?u32 = null;
        for (m.nodes, 0..) |n, i| {
            if (n.main_token != tok) continue;
            if (resolutions[i] == .func) if (try d.function(resolutions[i].func)) return true;
            if (ty == null and isRenderable(node_types[i])) ty = node_types[i];
            switch (n.tag) {
                .identifier => if (resolutions[i] == .local) {
                    local = true;
                },
                .var_decl, .pattern_binding, .for_stmt, .for_in_stmt, .for_in2_stmt => local = true,
                .fn_decl, .param, .field_init, .struct_decl, .enum_decl, .protocol_decl, .enum_variant_unit, .enum_variant_tuple, .enum_variant_struct, .enum_init_unit, .enum_init_tuple, .enum_init_struct, .pattern_variant => {
                    if (decl == null) decl = @intCast(i);
                },
                else => {},
            }
        }

        if (decl) |i| if (try d.declaration(i, tok, ty)) return true;
        if (ty) |t| {
            if (local) try d.buf.print(d.a, "{s}: ", .{d.text(tok)});
            try renderType(d.a, &d.buf, t, d.tc.layouts, d.tc.enum_layouts);
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
        const node_types = d.tc.node_types[d.graph.entry_index];
        for (m.nodes, 0..) |n, i| {
            if (n.main_token != tok or n.tag != .field_access) continue;
            if (isRenderable(node_types[i])) return d.variant(@intCast(i), node_types[i]);
            for (m.nodes, 0..) |call, c| {
                if (call.tag == .call and call.lhs.int() == i) return d.variant(@intCast(i), node_types[c]);
            }
        }
        return false;
    }

    fn declaration(d: *Describer, i: u32, tok: u32, ty: ?Type) !bool {
        const m = d.module();
        const n = m.nodes[i];
        switch (n.tag) {
            .fn_decl => {
                for (d.res.fns, 0..) |gf, f| {
                    if (gf.module == d.graph.entry_index and gf.decl_node.int() == i) return d.function(@intCast(f));
                }
                // A protocol's method signature never enters the fn table: show it as written.
                return d.sourceUntil(d.declStart(n.main_token), .l_brace);
            },
            .param => {
                const t = ty orelse d.paramType(i);
                if (t != null and isRenderable(t.?)) {
                    try d.buf.print(d.a, "{s}: ", .{d.text(tok)});
                    try renderType(d.a, &d.buf, t.?, d.tc.layouts, d.tc.enum_layouts);
                    return true;
                }
                // A generic template's field (`left: A`) has no concrete type: show it as written.
                return d.sourceUntil(tok, .comma);
            },
            .field_init => {
                const t = d.tc.node_types[d.graph.entry_index][n.lhs.int()];
                if (!isRenderable(t)) return false;
                try d.buf.print(d.a, "{s}: ", .{d.text(tok)});
                try renderType(d.a, &d.buf, t, d.tc.layouts, d.tc.enum_layouts);
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
                    for (d.res.fns, 0..) |gf, f| {
                        if (gf.module != d.graph.entry_index or gf.decl_node.int() != o) continue;
                        const sig = d.tc.sigs[f];
                        return if (k < sig.params.len) sig.params[k] else null;
                    }
                    return null;
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

    fn function(d: *Describer, f: u32) !bool {
        if (f >= d.tc.sigs.len) return false;
        const sig = d.tc.sigs[f];
        const names = try d.paramNames(f);
        try d.buf.appendSlice(d.a, "fn ");
        try d.buf.appendSlice(d.a, unmangled(bareName(sig.name)));
        try d.buf.append(d.a, '(');
        for (sig.params, 0..) |p, k| {
            if (k != 0) try d.buf.appendSlice(d.a, ", ");
            if (k < names.len) try d.buf.print(d.a, "{s}: ", .{names[k]});
            try renderType(d.a, &d.buf, p, d.tc.layouts, d.tc.enum_layouts);
        }
        try d.buf.appendSlice(d.a, ") -> ");
        try renderType(d.a, &d.buf, sig.ret, d.tc.layouts, d.tc.enum_layouts);
        return true;
    }

    /// Parameter names from `f`'s declaration, in whatever module declares it; empty for a
    /// builtin (no source) so the signature falls back to bare types.
    fn paramNames(d: *const Describer, f: u32) ![]const []const u8 {
        if (f >= d.res.fns.len) return &.{};
        const gf = d.res.fns[f];
        if (gf.decl_node == Ast.none) return &.{};
        const tm = &d.graph.modules[gf.module];
        const proto = Ast.protoAt(tm.tree(), tm.nodes[gf.decl_node.int()].lhs.int());
        const out = try d.a.alloc([]const u8, proto.params.len);
        for (proto.params, out) |p, *name| {
            const t = tm.tokens[tm.nodes[p.int()].main_token];
            name.* = tm.source[t.start..t.end];
        }
        return out;
    }

    fn structNamed(d: *const Describer, name: []const u8) ?*const Layout {
        for (d.tc.layouts) |*l| if (std.mem.eql(u8, bareName(l.name), name)) return l;
        return null;
    }

    fn enumNamed(d: *const Describer, name: []const u8) ?*const EnumLayout {
        for (d.tc.enum_layouts) |*e| if (std.mem.eql(u8, bareName(e.name), name)) return e;
        return null;
    }

    /// A type's declaration as written (generic parameters and all), found in whichever
    /// module declares it; the entry module wins a name clash.
    fn typeDefinition(d: *Describer, name: []const u8) !bool {
        var found: ?struct { mod: *const Graph.Module, tok: u32 } = null;
        for (d.graph.modules, 0..) |*tm, mi| {
            for (tm.nodes) |n| switch (n.tag) {
                .struct_decl, .enum_decl, .protocol_decl => {
                    const t = tm.tokens[n.main_token];
                    if (!std.mem.eql(u8, tm.source[t.start..t.end], name)) continue;
                    if (found == null or mi == d.graph.entry_index) found = .{ .mod = tm, .tok = n.main_token };
                },
                else => {},
            };
        }
        const f = found orelse {
            for (native_enums) |e| if (std.mem.eql(u8, e.name, name)) {
                try d.buf.appendSlice(d.a, e.decl);
                return true;
            };
            return false;
        };
        const start = declStartIn(f.mod.tokens, f.tok);
        const end = closingBrace(f.mod.tokens, f.tok) orelse return false;
        try appendCollapsed(d.a, &d.buf, f.mod.source[f.mod.tokens[start].start..f.mod.tokens[end].end]);
        return true;
    }

    /// The first token of the declaration whose name is `name_tok`: its keyword, and a
    /// leading `pub`.
    fn declStart(d: *const Describer, name_tok: u32) u32 {
        return declStartIn(d.module().tokens, name_tok);
    }

    /// Entry-module source from `start` up to (not including) the first `stop` token at
    /// nesting depth 0 or the end of the line, whitespace collapsed.
    fn sourceUntil(d: *Describer, start: u32, stop: toyc.Tag) !bool {
        const m = d.module();
        var depth: u32 = 0;
        var last = start;
        var i = start;
        while (i < m.tokens.len) : (i += 1) {
            const t = m.tokens[i];
            if (i > start and std.mem.indexOfScalar(u8, m.source[m.tokens[last].end..t.start], '\n') != null) break;
            switch (t.tag) {
                .l_paren, .l_bracket => depth += 1,
                .r_paren, .r_bracket, .r_brace => {
                    if (depth == 0) break;
                    depth -= 1;
                },
                else => {},
            }
            if (depth == 0 and t.tag == stop) break;
            last = i;
        }
        try appendCollapsed(d.a, &d.buf, m.source[m.tokens[start].start..m.tokens[last].end]);
        return true;
    }

    /// A variant as `Enum.Variant(payload)`. The enum is the node's own type (a construction
    /// or pattern), else the one enum that declares a variant of this name.
    fn variant(d: *Describer, i: u32, ty: ?Type) !bool {
        const name = d.text(d.module().nodes[i].main_token);
        const e: *const EnumLayout = blk: {
            if (ty) |t| if (t.kind == .@"enum" and t.enum_id < d.tc.enum_layouts.len) break :blk &d.tc.enum_layouts[t.enum_id];
            var found: ?*const EnumLayout = null;
            for (d.tc.enum_layouts) |*el| for (el.variants) |v| if (std.mem.eql(u8, v.name, name)) {
                if (found != null) return false; // ambiguous
                found = el;
            };
            break :blk found orelse return false;
        };
        for (e.variants) |v| if (std.mem.eql(u8, v.name, name)) {
            try renderNominal(d.a, &d.buf, e.name, d.tc.layouts, d.tc.enum_layouts);
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
                    try renderType(d.a, &d.buf, t, d.tc.layouts, d.tc.enum_layouts);
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
            try renderType(d.a, &d.buf, t, d.tc.layouts, d.tc.enum_layouts);
        }
    }
};

/// The prelude enums the compiler registers natively (see `Prelude.registerOptionResult`):
/// no module declares them, so there is no source to show.
const native_enums = [_]struct { name: []const u8, decl: []const u8 }{
    .{ .name = "Option", .decl = "enum Option[T] { some(T), none }" },
    .{ .name = "Result", .decl = "enum Result[T, E] { ok(T), err(E) }" },
};

fn declStartIn(tokens: []const Token, name_tok: u32) u32 {
    var i = name_tok;
    if (i > 0) switch (tokens[i - 1].tag) {
        .kw_struct, .kw_enum, .kw_protocol, .kw_fn => i -= 1,
        else => {},
    };
    if (i > 0 and tokens[i - 1].tag == .kw_pub) i -= 1;
    return i;
}

/// The `}` closing the first `{` after `from`.
fn closingBrace(tokens: []const Token, from: u32) ?u32 {
    var depth: u32 = 0;
    for (tokens[from..], from..) |t, i| switch (t.tag) {
        .l_brace => depth += 1,
        .r_brace => {
            if (depth == 0) return null;
            depth -= 1;
            if (depth == 0) return @intCast(i);
        },
        else => {},
    };
    return null;
}

/// `text` with each whitespace run (newlines included) collapsed to one space, and none
/// inside brackets: a declaration spread over lines reads as one signature.
fn appendCollapsed(a: std.mem.Allocator, buf: *std.ArrayList(u8), text: []const u8) !void {
    var gap = false;
    for (text) |c| {
        if (std.ascii.isWhitespace(c)) {
            gap = true;
            continue;
        }
        if (gap and buf.items.len > 0) {
            const prev = buf.items[buf.items.len - 1];
            if (prev != '(' and prev != '[' and c != ')' and c != ']' and c != ',') try buf.append(a, ' ');
        }
        gap = false;
        try buf.append(a, c);
    }
}

fn fieldType(names: []const []const u8, types: []const Type, name: []const u8) ?Type {
    for (names, types) |n, t| if (std.mem.eql(u8, n, name)) return t;
    return null;
}
