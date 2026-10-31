//! The libraries a program links: every `-l` the command line names, then every
//! library an `extern "name" { .. }` block names, each loaded once, libSystem first.
//! It also binds each extern's C symbol to the ordinal of its block's library.
//!
//! It reads only the module ASTs and the file system, so `check` reports the same
//! library errors `build` does, and a warm build recomputes the set from the
//! current sources and flags (the linked image is never cached).

const std = @import("std");
const Io = std.Io;
const Ast = @import("../ast/Ast.zig");
const Graph = @import("Graph.zig");
const Dylib = @import("../link/Dylib.zig");
const MachO = @import("../link/MachO.zig");
const codes = @import("../diagnostics/codes.zig");
const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;
const runtime = @import("../codegen/runtime/runtime.zig");
const literal = @import("../types/literal.zig");

/// The command line's `-l` names and `-L` folders, in order.
pub const Request = struct {
    libs: []const []const u8 = &.{},
    dirs: []const []const u8 = &.{},
};

/// A problem with a `-l` value: a usage error, since it is not in any source file.
pub const CliError = struct { code: codes.Code, message: []const u8 };

/// Everything is allocated in `arena`, so one `deinit` frees it.
pub const Result = struct {
    arena: std.heap.ArenaAllocator,
    linking: Dylib.Linking,
    cli_errors: []const CliError,
    /// At the library string of an `extern "name"` block, or at a conflicting extern.
    diags: []const Diagnostic,

    pub fn deinit(self: *Result) void {
        self.arena.deinit();
    }

    pub fn ok(self: *const Result) bool {
        return self.cli_errors.len == 0 and self.diags.len == 0;
    }
};

const Builder = struct {
    a: std.mem.Allocator,
    io: Io,
    dirs: []const []const u8,
    dylibs: std.ArrayList(MachO.Dylib) = .empty,
    rpaths: std.ArrayList([]const u8) = .empty,
    /// Each name looked up so far, to its ordinal (null when it failed).
    seen: std.StringHashMapUnmanaged(?u8) = .empty,

    /// The library a resolved name adds, deduplicated by install name, as its ordinal.
    fn add(b: *Builder, found: Dylib.Found) !u8 {
        const d = switch (found) {
            .libsystem => return 1,
            .dylib => |d| d,
            else => unreachable,
        };
        for (b.dylibs.items, 0..) |have, i| {
            if (std.mem.eql(u8, have.install_name, d.dylib.install_name)) return @intCast(i + 2);
        }
        if (b.dylibs.items.len == MachO.Libraries.max_dylibs) return error.TooManyLibraries;
        try b.dylibs.append(b.a, d.dylib);
        if (d.rpath) |r| {
            for (b.rpaths.items) |have| {
                if (std.mem.eql(u8, have, r)) break;
            } else try b.rpaths.append(b.a, r);
        }
        return @intCast(b.dylibs.items.len + 1);
    }

    /// Why `name` cannot be linked, or null with `ordinal` set.
    fn resolve(b: *Builder, name: []const u8, ordinal: *u8) !?struct { code: codes.Code, message: []const u8 } {
        const found = try Dylib.find(b.a, b.io, name, b.dirs);
        switch (found) {
            .libsystem, .dylib => {
                ordinal.* = try b.add(found);
                return null;
            },
            .static => |path| return .{ .code = .R0015, .message = if (std.mem.eql(u8, path, name))
                try std.fmt.allocPrint(b.a, "library '{s}' is a static library; toy links only dynamic libraries (.dylib)", .{name})
            else
                try std.fmt.allocPrint(b.a, "library '{s}' is only a static library ('{s}'); toy links only dynamic libraries (.dylib)", .{ name, path }) },
            .not_found => return .{ .code = .R0016, .message = try b.notFound(name) },
            .not_dylib => |nd| return .{ .code = .R0017, .message = try std.fmt.allocPrint(b.a, "library '{s}' cannot be linked: '{s}': {s}", .{ name, nd.path, nd.why }) },
        }
    }

    fn notFound(b: *Builder, name: []const u8) ![]const u8 {
        if (Dylib.isPath(name)) return std.fmt.allocPrint(b.a, "library '{s}' not found: no such file", .{name});
        var list: std.ArrayList(u8) = .empty;
        for (b.dirs) |d| try list.print(b.a, "{s}, ", .{d});
        for (Dylib.default_dirs) |d| try list.print(b.a, "{s}, ", .{d});
        try list.appendSlice(b.a, "the dyld shared cache");
        return std.fmt.allocPrint(b.a, "library '{s}' not found: no lib{s}.dylib in {s}", .{ name, name, list.items });
    }
};

/// A library string's text with its escapes decoded. A malformed escape keeps the raw
/// text, which then fails to resolve with the name as written.
fn libraryName(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    if (std.mem.indexOfScalar(u8, raw, '\\') == null) return raw;
    return switch (try literal.decodeString(a, try std.fmt.allocPrint(a, "\"{s}\"", .{raw}))) {
        .ok => |bytes| bytes,
        else => raw,
    };
}

/// Where one extern binds its C symbol. `site` is null for the runtime's own imports.
const Bound = struct {
    ordinal: u8,
    lib: []const u8,
    site: ?struct { module: u32, start: u32, end: u32 },
};

/// R0018 for `sym`, bound by `first` and again by `second` to another library. It sits
/// on a declaration the user can edit: the second, unless that one is in a bundled module
/// and the first is not.
fn conflict(a: std.mem.Allocator, graph: *const Graph.Graph, sym: []const u8, first: Bound, second: Bound) !Diagnostic {
    const s2 = second.site.?;
    const swap = if (first.site) |s1| graph.modules[s2.module].bundled and !graph.modules[s1.module].bundled else false;
    const at = if (swap) first else second;
    const other = if (swap) second else first;
    const where = if (other.site) |o|
        try std.fmt.allocPrint(a, "in module '{s}'", .{graph.modules[o.module].path})
    else
        "by the toy runtime";
    return .{
        .byte_offset = at.site.?.start,
        .end = at.site.?.end,
        .message = try std.fmt.allocPrint(a, "C symbol '{s}' is bound to library '{s}' here but to '{s}' {s}; a symbol comes from one library", .{ sym, at.lib, other.lib, where }),
        .scope = at.site.?.module,
        .code = .R0018,
    };
}

/// Collect and resolve the program's libraries. `graph` must have discovered cleanly.
pub fn collect(gpa: std.mem.Allocator, io: Io, graph: *const Graph.Graph, req: Request) !Result {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var b: Builder = .{ .a = a, .io = io, .dirs = req.dirs };
    var cli_errors: std.ArrayList(CliError) = .empty;
    var diags: std.ArrayList(Diagnostic) = .empty;

    for (req.libs) |name| {
        if (b.seen.contains(name)) continue;
        var ordinal: u8 = 0;
        if (try b.resolve(name, &ordinal)) |e| {
            try cli_errors.append(a, .{ .code = e.code, .message = try std.fmt.allocPrint(a, "{s} (-l {s})", .{ e.message, name }) });
            try b.seen.put(a, name, null);
        } else try b.seen.put(a, name, ordinal);
    }

    // An import binds by bare C name, so the runtime's own imports are bound first: a
    // block that names one for another library would move the runtime's call there.
    var bound: std.StringArrayHashMapUnmanaged(Bound) = .empty;
    for (runtime.libsystem_imports) |sym| try bound.put(a, sym, .{ .ordinal = 1, .lib = "libSystem", .site = null });
    for (graph.modules, 0..) |*m, mi| {
        const tree = m.tree();
        const prog = tree.nodes[Ast.root(tree.nodes).int()];
        if (prog.tag != .program) continue;
        for (Ast.rangeSlice(tree, prog.lhs.int())) |idx| {
            const n = tree.nodes[idx.int()];
            if (n.tag != .extern_fn_decl) continue;
            const lib = if (Ast.externLibrary(tree, m.tokens, m.source, idx)) |raw| try libraryName(a, raw) else null;
            var ordinal: u8 = 1;
            if (lib) |name| {
                const gop = try b.seen.getOrPut(a, name);
                if (!gop.found_existing) {
                    gop.value_ptr.* = null;
                    var o: u8 = 0;
                    if (try b.resolve(name, &o)) |e| {
                        const tok = m.tokens[tree.nodes[n.rhs.int()].main_token];
                        try diags.append(a, .{ .byte_offset = tok.start, .end = tok.end, .message = e.message, .scope = @intCast(mi), .code = e.code });
                    } else gop.value_ptr.* = o;
                }
                ordinal = gop.value_ptr.* orelse continue;
            }
            const sym_tok = m.tokens[n.main_token];
            const sym = sym_tok.text(m.source);
            const here: Bound = .{ .ordinal = ordinal, .lib = lib orelse "libSystem", .site = .{ .module = @intCast(mi), .start = sym_tok.start, .end = sym_tok.end } };
            const gop = try bound.getOrPut(a, sym);
            if (!gop.found_existing) {
                gop.value_ptr.* = here;
            } else if (gop.value_ptr.ordinal != ordinal) {
                try diags.append(a, try conflict(a, graph, sym, gop.value_ptr.*, here));
            }
        }
    }

    var bindings: std.ArrayList(Dylib.Binding) = .empty;
    for (bound.keys(), bound.values()) |sym, v| {
        if (v.ordinal != 1) try bindings.append(a, .{ .name = sym, .ordinal = v.ordinal });
    }
    std.mem.sort(Dylib.Binding, bindings.items, {}, struct {
        fn lt(_: void, x: Dylib.Binding, y: Dylib.Binding) bool {
            return std.mem.lessThan(u8, x.name, y.name);
        }
    }.lt);

    return .{
        .arena = arena,
        .linking = .{ .dylibs = b.dylibs.items, .rpaths = b.rpaths.items, .bindings = bindings.items },
        .cli_errors = cli_errors.items,
        .diags = diags.items,
    };
}

const testing = std.testing;

fn graphOf(gpa: std.mem.Allocator, src: []const u8) !struct { tokens: []@import("../ast/Token.zig").Token, tree: Ast.Tree, g: Graph.Graph } {
    const tokens = try @import("../lex.zig").tokenize(gpa, src);
    const tree = try @import("../parse.zig").expectTree(gpa, tokens, src);
    const g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    return .{ .tokens = tokens, .tree = tree, .g = g };
}

test "externs bind to their block's library; libSystem names and bare externs stay ordinal 1" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var pa: std.heap.ArenaAllocator = .init(gpa);
    defer pa.deinit();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", pa.allocator());
    var buf: [128]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyq.dylib", .data = Dylib.fakeDylib(&buf, "@rpath/libtoyq.dylib", 6) });

    const src =
        \\extern "toyq" {
        \\    fn q_one() -> int
        \\    fn q_two(x: int) -> int
        \\}
        \\extern "m" { fn sqrt(x: float) -> float }
        \\extern fn labs(x: int) -> int
        \\fn main() -> int { 0 }
        \\
    ;
    var p = try graphOf(gpa, src);
    defer {
        p.g.deinit(gpa);
        gpa.free(p.tree.nodes);
        gpa.free(p.tree.extra);
        gpa.free(p.tokens);
    }
    var r = try collect(gpa, io, &p.g, .{ .dirs = &.{dir} });
    defer r.deinit();
    try testing.expect(r.ok());
    try testing.expectEqual(@as(usize, 1), r.linking.dylibs.len);
    try testing.expectEqualStrings(dir, r.linking.rpaths[0]);
    try testing.expectEqual(@as(u8, 2), r.linking.ordinalOf("q_one"));
    try testing.expectEqual(@as(u8, 2), r.linking.ordinalOf("q_two"));
    try testing.expectEqual(@as(u8, 1), r.linking.ordinalOf("sqrt"));
    try testing.expectEqual(@as(u8, 1), r.linking.ordinalOf("labs"));
}

test "a missing or static library is one error at the block; -l problems are usage errors" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const src =
        \\extern "toy-no-such-lib" {
        \\    fn a() -> int
        \\    fn b() -> int
        \\}
        \\extern "libfoo.a" { fn c() }
        \\fn main() -> int { 0 }
        \\
    ;
    var p = try graphOf(gpa, src);
    defer {
        p.g.deinit(gpa);
        gpa.free(p.tree.nodes);
        gpa.free(p.tree.extra);
        gpa.free(p.tokens);
    }
    var r = try collect(gpa, threaded.io(), &p.g, .{ .libs = &.{ "z.o", "toy-no-such-lib" } });
    defer r.deinit();
    try testing.expectEqual(@as(usize, 2), r.cli_errors.len);
    try testing.expectEqual(codes.Code.R0015, r.cli_errors[0].code);
    try testing.expectEqual(codes.Code.R0016, r.cli_errors[1].code);
    try testing.expect(std.mem.indexOf(u8, r.cli_errors[1].message, "/usr/local/lib") != null);
    // The block's library was already reported for `-l`; the archive is reported once.
    try testing.expectEqual(@as(usize, 1), r.diags.len);
    try testing.expectEqual(codes.Code.R0015, r.diags[0].code);
    try testing.expectEqualStrings("\"libfoo.a\"", src[r.diags[0].byte_offset..r.diags[0].end]);
}

test "one C symbol bound to two libraries is R0018 at the second extern" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var pa: std.heap.ArenaAllocator = .init(gpa);
    defer pa.deinit();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", pa.allocator());
    var buf: [128]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyz.dylib", .data = Dylib.fakeDylib(&buf, "/opt/libtoyz.dylib", 6) });
    const src =
        \\extern "toyz" { fn crc32(c: int, p: rawptr, n: int) -> int }
        \\extern fn crc32(c: int, p: rawptr, n: int) -> int
        \\fn main() -> int { 0 }
        \\
    ;
    var p = try graphOf(gpa, src);
    defer {
        p.g.deinit(gpa);
        gpa.free(p.tree.nodes);
        gpa.free(p.tree.extra);
        gpa.free(p.tokens);
    }
    var r = try collect(gpa, io, &p.g, .{ .dirs = &.{dir} });
    defer r.deinit();
    try testing.expectEqual(@as(usize, 1), r.diags.len);
    try testing.expectEqual(codes.Code.R0018, r.diags[0].code);
}

fn graphOfMods(gpa: std.mem.Allocator, mods: []const struct { path: []const u8, src: []const u8, bundled: bool = false }) !Graph.Graph {
    const modules = try gpa.alloc(Graph.Module, mods.len);
    for (mods, modules) |md, *m| {
        const tokens = try @import("../lex.zig").tokenize(gpa, md.src);
        const tree = try @import("../parse.zig").expectTree(gpa, tokens, md.src);
        m.* = .{ .path = md.path, .file = "", .source = @constCast(md.src), .tokens = tokens, .nodes = tree.nodes, .extra = tree.extra, .pub_bits = tree.pub_bits, .bundled = md.bundled };
    }
    return .{ .ownership = .borrowed, .modules = modules };
}

fn freeGraphOfMods(gpa: std.mem.Allocator, g: *Graph.Graph) void {
    for (g.modules) |m| {
        gpa.free(m.tokens);
        gpa.free(m.nodes);
        gpa.free(m.extra);
        if (m.pub_bits.len != 0) gpa.free(m.pub_bits);
    }
    g.deinit(gpa);
}

test "a block that binds a symbol the runtime imports is R0018 at the block's extern" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var pa: std.heap.ArenaAllocator = .init(gpa);
    defer pa.deinit();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", pa.allocator());
    var buf: [128]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyz.dylib", .data = Dylib.fakeDylib(&buf, "/opt/libtoyz.dylib", 6) });
    const src =
        \\extern "toyz" { fn malloc(n: int) -> rawptr }
        \\fn main() -> int { 0 }
        \\
    ;
    var g = try graphOfMods(gpa, &.{.{ .path = "main", .src = src }});
    defer freeGraphOfMods(gpa, &g);
    var r = try collect(gpa, io, &g, .{ .dirs = &.{dir} });
    defer r.deinit();
    try testing.expectEqual(@as(usize, 1), r.diags.len);
    try testing.expectEqual(codes.Code.R0018, r.diags[0].code);
    try testing.expectEqualStrings("malloc", src[r.diags[0].byte_offset..r.diags[0].end]);
    try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "toy runtime") != null);
}

test "R0018 against a bundled module's extern sits at the user's extern" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var pa: std.heap.ArenaAllocator = .init(gpa);
    defer pa.deinit();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", pa.allocator());
    var buf: [128]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyz.dylib", .data = Dylib.fakeDylib(&buf, "/opt/libtoyz.dylib", 6) });
    const user =
        \\extern "toyz" { fn strlen(p: rawptr) -> int }
        \\fn main() -> int { 0 }
        \\
    ;
    const core = "pub extern fn strlen(p: rawptr) -> int\n";
    for ([_]bool{ false, true }) |core_first| {
        var g = if (core_first)
            try graphOfMods(gpa, &.{ .{ .path = "core/os", .src = core, .bundled = true }, .{ .path = "main", .src = user } })
        else
            try graphOfMods(gpa, &.{ .{ .path = "main", .src = user }, .{ .path = "core/os", .src = core, .bundled = true } });
        defer freeGraphOfMods(gpa, &g);
        var r = try collect(gpa, io, &g, .{ .dirs = &.{dir} });
        defer r.deinit();
        try testing.expectEqual(@as(usize, 1), r.diags.len);
        try testing.expectEqual(codes.Code.R0018, r.diags[0].code);
        try testing.expectEqual(@as(u32, if (core_first) 1 else 0), r.diags[0].scope);
        try testing.expectEqualStrings("strlen", user[r.diags[0].byte_offset..r.diags[0].end]);
        try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "core/os") != null);
    }
}

test "a block's library string has its escapes decoded" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var pa: std.heap.ArenaAllocator = .init(gpa);
    defer pa.deinit();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", pa.allocator());
    var buf: [128]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyq.dylib", .data = Dylib.fakeDylib(&buf, "/opt/libtoyq.dylib", 6) });
    var g = try graphOfMods(gpa, &.{.{ .path = "main", .src = "extern \"toy\\x71\" { fn q_one() -> int }\n" }});
    defer freeGraphOfMods(gpa, &g);
    var r = try collect(gpa, io, &g, .{ .dirs = &.{dir} });
    defer r.deinit();
    try testing.expect(r.ok());
    try testing.expectEqual(@as(u8, 2), r.linking.ordinalOf("q_one"));
}
