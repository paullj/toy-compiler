//! Types and signatures as the user spells them.

const std = @import("std");
const toyc = @import("toy_compiler");

const Type = toyc.Typecheck.Type;
const Layout = toyc.Typecheck.Layout;
const EnumLayout = toyc.Typecheck.EnumLayout;

/// A type worth showing. The poison/check-only kinds (`invalid`, `type_var`, `app`) never
/// reach a hover answer — a slot left at the `@memset(.invalid)` default is "no info".
pub fn isRenderable(t: Type) bool {
    return t.kind != .invalid and t.kind != .type_var and t.kind != .app;
}

/// A symbol name minus its leading module qualifier. The qualifier is the entry file's
/// stem, which the user never wrote at the declaration. Toy identifiers contain no '.', so
/// the last '.' is the module separator.
pub fn bareName(name: []const u8) []const u8 {
    return if (std.mem.lastIndexOfScalar(u8, name, '.')) |dot| name[dot + 1 ..] else name;
}

/// Render a type as the compiler names it elsewhere. `layouts`/`enum_layouts` supply the
/// nominal name; the append copies that borrowed name into the caller's arena.
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
pub fn renderNominal(a: std.mem.Allocator, buf: *std.ArrayList(u8), name: []const u8, layouts: []const Layout, enum_layouts: []const EnumLayout) error{OutOfMemory}!void {
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

/// A symbol name minus the `$…` suffix the compiler mints for a generic instance or a
/// protocol method (`id$int`, `area$Area`): the user wrote neither.
pub fn unmangled(name: []const u8) []const u8 {
    return name[0 .. std.mem.indexOfScalar(u8, name, '$') orelse name.len];
}
