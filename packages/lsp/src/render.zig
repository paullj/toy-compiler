//! Types and signatures as the user spells them.

const std = @import("std");
const toyc = @import("toy_compiler");

const Type = toyc.Typecheck.Type;
const Layout = toyc.Typecheck.Layout;
const EnumLayout = toyc.Typecheck.EnumLayout;

/// A type worth showing. The poison/check-only kinds (`invalid`, `type_var`, `app`,
/// `func`) never reach a hover answer — a slot left at the `@memset(.invalid)` default is
/// "no info".
pub fn isRenderable(t: Type) bool {
    return t.kind != .invalid and t.kind != .type_var and t.kind != .app and t.kind != .func;
}

/// A symbol name minus its leading module qualifier. The qualifier is the entry file's
/// stem, which the user never wrote at the declaration. Toy identifiers contain no '.', so
/// the last '.' is the module separator.
pub fn bareName(name: []const u8) []const u8 {
    return if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot| name[dot + 1 ..] else name;
}

/// Render a type as the compiler names it elsewhere. `layouts`/`enum_layouts` supply the
/// nominal name; the append copies that borrowed name into the caller's arena.
pub fn renderType(a: std.mem.Allocator, buf: *std.ArrayList(u8), t: Type, layouts: []const Layout, enum_layouts: []const EnumLayout) error{OutOfMemory}!void {
    switch (t.kind) {
        .int => try buf.appendSlice(a, t.intName()),
        .bool => try buf.appendSlice(a, "bool"),
        .str => try buf.appendSlice(a, "str"),
        .float => try buf.appendSlice(a, "float"),
        .unit => try buf.appendSlice(a, "()"),
        .rawptr => try buf.appendSlice(a, "rawptr"),
        .never => try buf.appendSlice(a, "never"),
        .@"struct" => if (t.struct_id < layouts.len) {
            if (fnSigOf(t, layouts)) |fs| try renderFnType(a, buf, fs, layouts, enum_layouts) else try renderNominal(a, buf, layouts[t.struct_id].name, layouts, enum_layouts);
        } else try buf.appendSlice(a, "struct"),
        .@"enum" => if (t.enum_id < enum_layouts.len) try renderNominal(a, buf, enum_layouts[t.enum_id].name, layouts, enum_layouts) else try buf.appendSlice(a, "enum"),
        // Filtered out by `isRenderable` before we get here; kept total for the compiler.
        .invalid, .type_var, .app, .func => try buf.appendSlice(a, "?"),
    }
}

/// The signature of a function type, which reifies to a `{code, env}` pair struct; null
/// for any other type.
pub fn fnSigOf(t: Type, layouts: []const Layout) ?toyc.LayoutEngine.FnSig {
    if (t.kind != .@"struct" or t.struct_id >= layouts.len) return null;
    return layouts[t.struct_id].fn_sig;
}

/// `fn(A, B) -> R`, the spelling of the pair struct a function type reifies to. A unit
/// return is left off, as in the source.
fn renderFnType(a: std.mem.Allocator, buf: *std.ArrayList(u8), fs: toyc.LayoutEngine.FnSig, layouts: []const Layout, enum_layouts: []const EnumLayout) error{OutOfMemory}!void {
    try buf.appendSlice(a, "fn(");
    for (fs.params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        try renderType(a, buf, p, layouts, enum_layouts);
    }
    try buf.append(a, ')');
    if (fs.ret.kind == .unit) return;
    try buf.appendSlice(a, " -> ");
    try renderType(a, buf, fs.ret, layouts, enum_layouts);
}

/// `renderType`, except a `type_var(i)` with `i < var_names.len` prints `var_names[i]`: a
/// protocol method's signature is written over `Self` and the protocol's own params.
pub fn renderTypeNamed(a: std.mem.Allocator, buf: *std.ArrayList(u8), t: Type, layouts: []const Layout, enum_layouts: []const EnumLayout, var_names: []const []const u8) !void {
    if (t.kind == .type_var and t.typeVarOrd() < var_names.len) return buf.appendSlice(a, var_names[t.typeVarOrd()]);
    return renderType(a, buf, t, layouts, enum_layouts);
}

/// Whether `renderTypeNamed` has a name for `t` (see `isRenderable`).
pub fn isRenderableNamed(t: Type, var_names: []const []const u8) bool {
    return isRenderable(t) or (t.kind == .type_var and t.typeVarOrd() < var_names.len);
}

/// `text` with each whitespace run (newlines included) collapsed to one space, and none
/// inside brackets: a declaration spread over lines reads as one signature.
pub fn appendCollapsed(a: std.mem.Allocator, buf: *std.ArrayList(u8), text: []const u8) !void {
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

/// A nominal type's name as the user spells it. A generic instance is minted as
/// `Template$arg$…` (see `Mangle.instance`), with a struct/enum argument as `s<id>`/`e<id>`
/// into the same tables — rendered back as `Template[arg, …]`.
pub fn renderNominal(a: std.mem.Allocator, buf: *std.ArrayList(u8), name: []const u8, layouts: []const Layout, enum_layouts: []const EnumLayout) error{OutOfMemory}!void {
    var parts = std.mem.splitScalar(u8, bareName(name), '$');
    try buf.appendSlice(a, parts.first());
    var first = true;
    while (parts.next()) |arg| {
        try buf.appendSlice(a, if (first) "[" else ", ");
        first = false;
        const id = if (arg.len > 1) std.fmt.parseInt(u32, arg[1..], 10) catch null else null;
        if (arg[0] == 's' and id != null and id.? < layouts.len) {
            try renderType(a, buf, Type.structT(id.?), layouts, enum_layouts);
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
        if (toyc.Modes.isMut(sig.mut_mask, i)) try buf.appendSlice(a, "mut ");
        try renderType(a, buf, p, layouts, enum_layouts);
    }
    try buf.appendSlice(a, ") -> ");
    try renderType(a, buf, sig.ret, layouts, enum_layouts);
}

/// A symbol name minus the `$…` suffix the compiler mints for a generic instance or a
/// protocol method (`id$int`, `area$Area`): the user wrote neither.
pub fn unmangled(name: []const u8) []const u8 {
    return name[0 .. std.mem.indexOfScalar(u8, name, '$') orelse name.len];
}

const testing = std.testing;

test "renderTypeNamed spells type vars and falls back to ?" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const names: []const []const u8 = &.{ "Self", "U" };
    const cases = [_]struct { t: Type, want: []const u8 }{
        .{ .t = Type.typeVar(0), .want = "Self" },
        .{ .t = Type.typeVar(1), .want = "U" },
        .{ .t = Type.typeVar(2), .want = "?" },
        .{ .t = Type.app(0), .want = "?" },
        .{ .t = Type.int, .want = "int" },
        .{ .t = Type.str, .want = "str" },
    };
    for (cases) |c| {
        var named: std.ArrayList(u8) = .empty;
        try renderTypeNamed(a, &named, c.t, &.{}, &.{}, names);
        try testing.expectEqualStrings(c.want, named.items);
        try testing.expectEqual(!std.mem.eql(u8, c.want, "?"), isRenderableNamed(c.t, names));
        if (c.t.kind == .type_var) continue;
        var plain: std.ArrayList(u8) = .empty;
        try renderType(a, &plain, c.t, &.{}, &.{});
        try testing.expectEqualStrings(plain.items, named.items);
    }
}

test "renders a fn-pair struct as fn(A) -> R and inside Vec[..]" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pair_names = [_][]const u8{ "code", "env" };
    const pair_types = [_]Type{ Type.int, Type.rawptr };
    const pair_offs = [_]u32{ 0, 8 };
    const pair: Layout = .{ .name = "fn$int$int", .field_names = @constCast(&pair_names), .field_types = @constCast(&pair_types), .offsets = @constCast(&pair_offs), .size = 16, .@"align" = 8, .fn_sig = .{ .params = &.{Type.int}, .ret = Type.int } };
    var unit_ret = pair;
    unit_ret.name = "fn$str$unit";
    unit_ret.fn_sig = .{ .params = &.{ Type.str, Type.structT(0) }, .ret = Type.unit };
    var vec = pair;
    vec.name = "Vec$s0";
    vec.fn_sig = null;
    const layouts = [_]Layout{ pair, unit_ret, vec };
    const cases = [_]struct { t: Type, want: []const u8 }{
        .{ .t = Type.structT(0), .want = "fn(int) -> int" },
        .{ .t = Type.structT(1), .want = "fn(str, fn(int) -> int)" },
        .{ .t = Type.structT(2), .want = "Vec[fn(int) -> int]" },
    };
    for (cases) |c| {
        var buf: std.ArrayList(u8) = .empty;
        try renderType(a, &buf, c.t, &layouts, &.{});
        try testing.expectEqualStrings(c.want, buf.items);
    }
    try testing.expect(!isRenderable(Type.func(0)));
}
