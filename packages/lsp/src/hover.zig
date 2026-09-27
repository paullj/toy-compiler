//! textDocument/hover: what the token under the cursor is — a callable's signature (with
//! parameter names), a binding as `name: Type`, a type's declaration as written, a field,
//! or an enum variant — rendered as a ```toy block the client highlights.
//!
//! Checks through the same `Workspace` as `diagnostics.checkBuffer`, but instead of mapping
//! diagnostics it RETAINS the graph/resolve/typecheck results to read `node_types` / `sigs` /
//! `resolutions`, renders the answer into an owned arena, THEN tears the compiler results
//! down. The render must precede those deinits: a `Sig.name` and a `Layout.name` are
//! BORROWED from the resolve/graph tables, so the arena copy is what outlives them.
//!
//! Selection is token-keyed because the AST stores only a `main_token` per node — no span,
//! no subtree-exit marker. The innermost node covering an offset is therefore "the node
//! whose `main_token` is the token under the cursor". A `{` / `(` / keyword token thus
//! hovers its enclosing node's type by design (a `block` -> `()`, a `call`'s `(` -> the
//! call's result); a whitespace/comment GAP carries no token and yields null.

const std = @import("std");
const toyc = @import("toy_compiler");
const Workspace = @import("Workspace.zig");

const Ast = toyc.Ast;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Token = toyc.Token;
const Type = toyc.Typecheck.Type;
const Layout = toyc.Typecheck.Layout;
const EnumLayout = toyc.Typecheck.EnumLayout;
const SourceMap = toyc.term.render.SourceMap;

/// Owns the arena backing `value`. `kind` is a static string. `value` stays valid until
/// `deinit`; the compiler results it was rendered from have already been torn down.
pub const Hover = struct {
    arena: std.heap.ArenaAllocator,
    value: []const u8,
    kind: []const u8,

    pub fn deinit(self: *Hover) void {
        self.arena.deinit();
    }
};

/// The type/signature at (`line`, `character`) in `source`, or null when there is nothing
/// meaningful under the cursor (a gap, an out-of-range or poisoned position, or a document
/// that does not parse). Resolve and type errors elsewhere do not suppress it: mid-edit, the
/// rest of the buffer still types. Never errors on a compile problem — only a hard I/O / OOM
/// fault propagates.
pub fn hoverAt(
    gpa: std.mem.Allocator,
    ws: Workspace,
    source: []const u8,
    line: u32,
    character: u32,
    doc_uri: []const u8,
) !?Hover {
    var graph = try ws.discover(gpa, doc_uri, source);
    defer graph.deinit(gpa);
    if (graph.err != null) return null;

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, ws.io, 0);
    defer tc.deinit(gpa);

    const m = graph.entry();

    var sm = try SourceMap.init(gpa, m.file, m.source);
    defer sm.deinit(gpa);

    const off = positionToOffset(&sm, line, character) orelse return null;
    const tok = tokenAt(m.tokens, off) orelse return null;

    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    var d: Describer = .{ .a = arena.allocator(), .graph = &graph, .res = &res, .tc = &tc };
    if (!try d.describe(tok)) {
        arena.deinit();
        return null;
    }
    // Fenced so the client highlights it as toy source.
    const value = try std.fmt.allocPrint(d.a, "```toy\n{s}\n```", .{d.buf.items});
    return .{ .arena = arena, .value = value, .kind = "markdown" };
}

/// Renders what the token under the cursor IS, in the entry module: a callable's signature
/// (with parameter names), a binding as `name: Type`, a type's definition, a field, or a
/// variant — falling back to the bare type of the expression the token heads.
const Describer = struct {
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

    fn describe(d: *Describer, tok: u32) !bool {
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

    /// `Enum.variant` spelled through the type name (`Option.some(x)`): the variant as the
    /// enum declares it, generic parameters included.
    fn qualifiedVariant(d: *Describer, tok: u32) !bool {
        const toks = d.module().tokens;
        if (tok < 2 or toks[tok - 1].tag != .dot or toks[tok - 2].tag != .identifier) return false;
        const enum_name = d.text(tok - 2);
        const mark = d.buf.items.len;
        if (!try d.typeDefinition(enum_name)) return false;
        // The declaration is scratch: cut it back off `buf` before writing the answer.
        const decl = try d.a.dupe(u8, d.buf.items[mark..]);
        d.buf.shrinkRetainingCapacity(mark);
        if (!std.mem.startsWith(u8, std.mem.trimStart(u8, decl, "pub "), "enum ")) return false;
        const item = variantIn(decl, d.text(tok)) orelse return false;
        try d.buf.print(d.a, "{s}.{s}", .{ enum_name, item });
        return true;
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
        const names = d.paramNames(f);
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
    fn paramNames(d: *const Describer, f: u32) []const []const u8 {
        if (f >= d.res.fns.len) return &.{};
        const gf = d.res.fns[f];
        if (gf.decl_node == Ast.none) return &.{};
        const tm = &d.graph.modules[gf.module];
        const proto = Ast.protoAt(tm.tree(), tm.nodes[gf.decl_node.int()].lhs.int());
        const out = d.a.alloc([]const u8, proto.params.len) catch return &.{};
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

/// The variant `name` as written in a collapsed enum declaration (`enum E { a(T), b }`):
/// the depth-0 comma-separated item inside the braces that starts with `name`.
fn variantIn(decl: []const u8, name: []const u8) ?[]const u8 {
    const open = std.mem.indexOfScalar(u8, decl, '{') orelse return null;
    const body = decl[open + 1 .. std.mem.lastIndexOfScalar(u8, decl, '}') orelse return null];
    var depth: u32 = 0;
    var start: usize = 0;
    for (body, 0..) |c, i| {
        switch (c) {
            '(', '[', '{' => depth += 1,
            ')', ']', '}' => depth -|= 1,
            else => {},
        }
        if ((c == ',' and depth == 0) or i + 1 == body.len) {
            const end = if (c == ',' and depth == 0) i else i + 1;
            const item = std.mem.trim(u8, body[start..end], " ");
            start = i + 1;
            if (!std.mem.startsWith(u8, item, name)) continue;
            if (item.len == name.len or item[name.len] == '(' or item[name.len] == ' ') return item;
        }
    }
    return null;
}

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

/// A symbol name minus the `$…` suffix the compiler mints for a generic instance or a
/// protocol method (`id$int`, `area$Area`): the user wrote neither.
fn unmangled(name: []const u8) []const u8 {
    return name[0 .. std.mem.indexOfScalar(u8, name, '$') orelse name.len];
}

fn fieldType(names: []const []const u8, types: []const Type, name: []const u8) ?Type {
    for (names, types) |n, t| if (std.mem.eql(u8, n, name)) return t;
    return null;
}

/// A type worth showing. The poison/check-only kinds (`invalid`, `type_var`, `app`) never
/// reach a hover answer — a slot left at the `@memset(.invalid)` default is "no info".
fn isRenderable(t: Type) bool {
    return t.kind != .invalid and t.kind != .type_var and t.kind != .app;
}

/// Exact inverse of `diagnostics.pointRange` (`off = line_starts[line] + character`). An
/// out-of-range line yields null; a character past the line's end clamps to the end (which
/// `tokenAt` then reads as a gap -> null). u64 arithmetic so the `line + 1` / `line + 2`
/// index used to bracket the line cannot overflow.
pub fn positionToOffset(sm: *const SourceMap, line: u32, character: u32) ?u32 {
    const lc = sm.lineCount();
    if (line >= lc) return null;
    const ls: u64 = sm.lineStart(@as(usize, line) + 1);
    const le: u64 = if (@as(usize, line) + 1 < lc) sm.lineStart(@as(usize, line) + 2) else sm.bytes.len;
    const want: u64 = ls + character;
    return @intCast(if (want >= le) le else want);
}

/// LSP (line,character) -> byte offset within `text`, via a throwaway SourceMap. Null for
/// an out-of-range line; a character past the line end clamps to the line end. This is the
/// SAME basis the feature handlers use, so a spliced edit and a later hover/definition agree.
pub fn offsetIn(gpa: std.mem.Allocator, text: []const u8, line: u32, character: u32) !?u32 {
    var sm = try SourceMap.init(gpa, "d", text);
    defer sm.deinit(gpa);
    return positionToOffset(&sm, line, character);
}

/// The token whose half-open `[start, end)` contains `off`, or null for a gap / EOF. Binary
/// search over the start-sorted, non-overlapping token stream.
pub fn tokenAt(tokens: []const Token, off: u32) ?u32 {
    var lo: usize = 0;
    var hi: usize = tokens.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (tokens[mid].start <= off) lo = mid + 1 else hi = mid;
    }
    if (lo == 0) return null;
    const i = lo - 1;
    return if (off < tokens[i].end) @intCast(i) else null;
}

/// Render a type as the compiler names it elsewhere. `layouts`/`enum_layouts` supply the
/// nominal name; the append copies that borrowed name into the caller's arena.
/// A symbol name minus its leading module qualifier. The qualifier is the entry file's
/// stem, which the user never wrote at the declaration — hover shows the bare declared name. Toy identifiers contain no '.', so the last '.' is the
/// module separator.
pub fn bareName(name: []const u8) []const u8 {
    return if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot| name[dot + 1 ..] else name;
}

pub fn renderType(a: std.mem.Allocator, buf: *std.ArrayList(u8), t: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) !void {
    switch (t.kind) {
        .int => try buf.appendSlice(a, t.intName()),
        .bool => try buf.appendSlice(a, "bool"),
        .str => try buf.appendSlice(a, "str"),
        .float => try buf.appendSlice(a, "float"),
        .unit => try buf.appendSlice(a, "()"),
        .rawptr => try buf.appendSlice(a, "rawptr"),
        .never => try buf.appendSlice(a, "never"),
        .@"struct" => if (t.struct_id < layouts.len) try renderNominal(a, buf, layouts[t.struct_id].name, layouts, enum_layouts) else try buf.appendSlice(a, "struct"),
        .@"enum" => if (t.enum_id < enum_layouts.len) try renderNominal(a, buf, enum_layouts[t.enum_id].name, layouts, enum_layouts) else try buf.appendSlice(a, "enum"),
        // Filtered out by `isRenderable` before we get here; kept total for the compiler.
        .invalid, .type_var, .app => try buf.appendSlice(a, "?"),
    }
}

/// A nominal type's name as the user spells it. A generic instance is minted as
/// `Template$arg$…` (see `Mono.mangle`), with a struct/enum argument as `s<id>`/`e<id>`
/// into the same tables — rendered back as `Template[arg, …]`.
fn renderNominal(a: std.mem.Allocator, buf: *std.ArrayList(u8), name: []const u8, layouts: []const Layout, enum_layouts: []const EnumLayout) error{OutOfMemory}!void {
    var parts = std.mem.splitScalar(u8, bareName(name), '$');
    try buf.appendSlice(a, parts.first());
    var first = true;
    while (parts.next()) |arg| {
        try buf.appendSlice(a, if (first) "[" else ", ");
        first = false;
        const id = if (arg.len > 1) std.fmt.parseInt(u32, arg[1..], 10) catch null else null;
        if (arg[0] == 's' and id != null and id.? < layouts.len) {
            try renderNominal(a, buf, layouts[id.?].name, layouts, enum_layouts);
        } else if (arg[0] == 'e' and id != null and id.? < enum_layouts.len) {
            try renderNominal(a, buf, enum_layouts[id.?].name, layouts, enum_layouts);
        } else {
            try buf.appendSlice(a, if (std.mem.eql(u8, arg, "unit")) "()" else arg);
        }
    }
    if (!first) try buf.append(a, ']');
}

/// `fn name(P0, P1, ...) -> R`. `sig` is duck-typed (`Typecheck.Sig` is not re-exported)
/// so this stays decoupled from the symbol module's surface.
pub fn renderSig(a: std.mem.Allocator, buf: *std.ArrayList(u8), sig: anytype, layouts: []const Layout, enum_layouts: []const EnumLayout) !void {
    try buf.appendSlice(a, "fn ");
    try buf.appendSlice(a, bareName(sig.name));
    try buf.append(a, '(');
    for (sig.params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        try renderType(a, buf, p, layouts, enum_layouts);
    }
    try buf.appendSlice(a, ") -> ");
    try renderType(a, buf, sig.ret, layouts, enum_layouts);
}

const testing = std.testing;

test "tokenAt: hit, gap, past-last-end, empty" {
    const toks = [_]Token{
        .{ .tag = .identifier, .start = 0, .end = 3 },
        .{ .tag = .identifier, .start = 5, .end = 8 },
    };
    try testing.expectEqual(@as(?u32, 0), tokenAt(&toks, 0));
    try testing.expectEqual(@as(?u32, 0), tokenAt(&toks, 2));
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 3)); // gap: end is exclusive
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 4)); // whitespace gap
    try testing.expectEqual(@as(?u32, 1), tokenAt(&toks, 7));
    try testing.expectEqual(@as(?u32, null), tokenAt(&toks, 8)); // past last end
    try testing.expectEqual(@as(?u32, null), tokenAt(&.{}, 0)); // empty stream
}

test "positionToOffset: basis, OOB line, char past EOL clamps" {
    const gpa = testing.allocator;
    const src = "ab\ncde\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    try testing.expectEqual(@as(?u32, 0), positionToOffset(&sm, 0, 0));
    try testing.expectEqual(@as(?u32, 1), positionToOffset(&sm, 0, 1));
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 1, 0)); // start of "cde"
    try testing.expectEqual(@as(?u32, 5), positionToOffset(&sm, 1, 2));
    // A char past the line end clamps to the line's end (the trailing '\n' index).
    try testing.expectEqual(@as(?u32, 3), positionToOffset(&sm, 0, 50));
    // OOB line -> null.
    try testing.expectEqual(@as(?u32, null), positionToOffset(&sm, 99, 0));
}

test "positionToOffset is the exact inverse of lineCol" {
    const gpa = testing.allocator;
    const src = "fn f() -> int {\n  x := 1\n  return x\n}\n";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    var off: u32 = 0;
    while (off <= src.len) : (off += 1) {
        const lc = sm.lineCol(off);
        const round = positionToOffset(&sm, @intCast(lc.line - 1), @intCast(lc.col - 1));
        try testing.expectEqual(@as(?u32, off), round);
    }
}

test "hover still types a binding when another line has a resolve error" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = "fn main() -> int {\n    count := 3\n    return count + missing\n}\n";
    var h = (try hoverAt(gpa, ws, src, 1, 5, "file:///b/main.toy")) orelse return error.HoverSuppressed;
    defer h.deinit();
    try testing.expectEqualStrings("```toy\ncount: int\n```", h.value);
}

test "hover describes declarations, bindings, type names, fields, and variants" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\pub struct Square { side: int }
        \\
        \\pub fn grow(sq: Square, by: int) -> Square {
        \\    bigger := Square { side: sq.side + by }
        \\    return bigger
        \\}
        \\
        \\pub enum Shape { Circle(int), Box(int) }
        \\
        \\pub fn area(shape: Shape) -> int {
        \\    match shape {
        \\        .Circle(r) -> 3 * r * r,
        \\        .Box(side) -> side * side,
        \\    }
        \\}
        \\
    ;
    const cases = [_]struct { line: u32, needle: []const u8, want: []const u8 }{
        .{ .line = 0, .needle = "Square", .want = "pub struct Square { side: int }" },
        .{ .line = 0, .needle = "side", .want = "side: int" },
        .{ .line = 2, .needle = "grow", .want = "fn grow(sq: Square, by: int) -> Square" },
        .{ .line = 2, .needle = "sq", .want = "sq: Square" },
        .{ .line = 2, .needle = "Square", .want = "pub struct Square { side: int }" },
        .{ .line = 3, .needle = "bigger", .want = "bigger: Square" },
        .{ .line = 4, .needle = "bigger", .want = "bigger: Square" },
        .{ .line = 7, .needle = "Shape", .want = "pub enum Shape { Circle(int), Box(int) }" },
        .{ .line = 7, .needle = "Circle", .want = "Shape.Circle(int)" },
        .{ .line = 10, .needle = "shape", .want = "shape: Shape" },
        .{ .line = 11, .needle = "Circle", .want = "Shape.Circle(int)" },
        .{ .line = 11, .needle = "r)", .want = "r: int" },
        .{ .line = 11, .needle = "r * r", .want = "r: int" },
    };
    for (cases) |c| {
        var it = std.mem.splitScalar(u8, src, '\n');
        var i: u32 = 0;
        const col: u32 = while (it.next()) |l| : (i += 1) {
            if (i == c.line) break @intCast(std.mem.indexOf(u8, l, c.needle).?);
        } else unreachable;
        var h = (try hoverAt(gpa, ws, src, c.line, col, "file:///b/shapes.toy")) orelse {
            std.debug.print("no hover for '{s}' on line {d}\n", .{ c.needle, c.line });
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}

test "hover shows generic, protocol, and prelude types as the user spells them" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Pair[A, B] { left: A, right: B }
        \\protocol Area {
        \\    fn area(self) -> int
        \\}
        \\struct Sq { side: int }
        \\impl Sq has Area {
        \\    fn area(self) -> int { self.side * self.side }
        \\}
        \\fn main() -> int {
        \\    point := Pair[int, int] { left: 3, right: 39 }
        \\    first: Option[int] = Option.some(point.left)
        \\    return Sq { side: 2 }.area()
        \\}
        \\
    ;
    const cases = [_]struct { line: u32, needle: []const u8, want: []const u8 }{
        .{ .line = 0, .needle = "Pair", .want = "struct Pair[A, B] { left: A, right: B }" },
        .{ .line = 0, .needle = "left", .want = "left: A" },
        .{ .line = 1, .needle = "Area", .want = "protocol Area { fn area(self) -> int }" },
        .{ .line = 2, .needle = "area", .want = "fn area(self) -> int" },
        .{ .line = 5, .needle = "Area", .want = "protocol Area { fn area(self) -> int }" },
        .{ .line = 6, .needle = "area", .want = "fn area(self: Sq) -> int" },
        .{ .line = 9, .needle = "point", .want = "point: Pair[int, int]" },
        .{ .line = 9, .needle = "left", .want = "left: int" },
        .{ .line = 10, .needle = "Option", .want = "enum Option[T] { some(T), none }" },
        .{ .line = 10, .needle = "first", .want = "first: Option[int]" },
        .{ .line = 10, .needle = "some", .want = "Option.some(T)" },
    };
    for (cases) |c| {
        var it = std.mem.splitScalar(u8, src, '\n');
        var i: u32 = 0;
        const col: u32 = while (it.next()) |l| : (i += 1) {
            if (i == c.line) break @intCast(std.mem.indexOf(u8, l, c.needle).?);
        } else unreachable;
        var h = (try hoverAt(gpa, ws, src, c.line, col, "file:///b/main.toy")) orelse {
            std.debug.print("no hover for '{s}' on line {d}\n", .{ c.needle, c.line });
            return error.NoHover;
        };
        defer h.deinit();
        const want = try std.fmt.allocPrint(gpa, "```toy\n{s}\n```", .{c.want});
        defer gpa.free(want);
        try testing.expectEqualStrings(want, h.value);
    }
}
