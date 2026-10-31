//! Whole-graph typecheck (types layer).
//!
//! Adapts the discovered module `Graph` + the whole-graph name-resolution result
//! (`resolve_graph.GraphResult`) into the program-wide type checker
//! (`types.checkGraph`). It:
//!
//!   * builds ONE program-wide layout table — global struct/enum ids assigned in
//!     (module-id order, then decl order); same-named types in DIFFERENT modules
//!     are DISTINCT ids (`Type.eql` compares ids → nominal distinctness for free);
//!   * resolves a qualified `mod.Type` type-ref (a `field_access` in type
//!     position whose receiver binds to a `.module`) to the OWNING module's global
//!     layout id — the cross-module form of the param/ret hole, so a pub-type
//!     layout change reaches the importer's fingerprint (closed by feeding the
//!     real global layout into `layouts`/`enum_layouts`);
//!   * type-checks every fn body against the resolver's GLOBAL fn table (`.func`
//!     ids index it directly); and
//!   * enforces pub-signature coherence (a `pub` fn may not name a non-pub type).
//!
//! This module owns the bridging tables (the per-module bare-name → global-id
//! maps + import namespaces) the type checker borrows; the returned
//! `types.GraphResult` is owned by the caller. This is the ONE type checker: a
//! lone source file is checked as the trivial one-module graph (`Graph.single` →
//! `checkGraph`), and its `GraphResult` is consumed WHOLE everywhere (no separate
//! single-source carrier).

const std = @import("std");
const Ast = @import("ast/Ast.zig");
const Graph = @import("driver/Graph.zig");
const ResolveGraph = @import("resolve_graph.zig");
const Typecheck = @import("types.zig");

pub const GraphResult = Typecheck.GraphResult;
const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;

/// Type-check the whole module `graph` against the resolver result `res`. Builds
/// the program-wide layout tables, checks every module's fn bodies cross-module,
/// and returns the per-module `node_types` + program-wide `sigs`/`layouts`/
/// `enum_layouts` + module-tagged diagnostics. Caller owns the result.
pub fn checkGraph(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    /// Pass-C worker pool, or `null` to force serial (see `Typecheck.Jobs.io`).
    /// The single-source path (per-file pipeline + inline test helpers) passes null.
    io: ?std.Io,
    /// The `-j` jobs knob for the Pass-C body fan-out (0 => host cpus). Ignored when
    /// `io == null`. See `Typecheck.Jobs.ncpu`.
    ncpu: usize,
) !GraphResult {
    return checkGraphWith(gpa, graph, res, .{ .io = io, .ncpu = ncpu });
}

/// `checkGraph` with every Pass-C dispatch setting spelled out, the fan-out threshold
/// included.
pub fn checkGraphWith(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *const ResolveGraph.GraphResult,
    jobs: Typecheck.Jobs,
) !GraphResult {
    const n = graph.modules.len;

    var ctx: Typecheck.GraphCtx = .{ .mods = try gpa.alloc(Typecheck.GraphCtx.ModuleCtx, n) };
    defer {
        for (ctx.mods) |*mc| {
            mc.struct_ids.deinit(gpa);
            mc.enum_ids.deinit(gpa);
            mc.protocol_ids.deinit(gpa);
            mc.namespaces.deinit(gpa);
            mc.pub_namespaces.deinit(gpa);
            mc.alias_ids.deinit(gpa);
            gpa.free(mc.selections);
        }
        gpa.free(ctx.mods);
    }
    for (graph.modules, 0..) |*m, i| {
        ctx.mods[i] = .{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[i],
            .closures = if (i < res.closures.len) res.closures[i] else &.{},
            .bundled = m.bundled,
            .path = m.path,
            .file = m.displayName(),
            .imports = m.imports,
        };
    }

    // Bind each module's import namespaces (namespace name → imported module id)
    // by rebuilding the `/`-joined import path and matching a module's canonical
    // name. Mirrors resolve_graph's binding so the type checker sees the same
    // namespaces (an unused or duplicate import is harmless here — resolve already
    // diagnosed collisions). Determinism: decls walked in source order.
    for (ctx.mods, 0..) |*mc, mi| {
        try bindNamespaces(gpa, graph, mc, @intCast(mi));
    }

    const fns = try gpa.alloc(Typecheck.GraphFnInput, res.fns.len);
    defer gpa.free(fns);
    for (res.fns, 0..) |gf, i| {
        fns[i] = .{
            .decl_node = gf.decl_node,
            .kind = gf.kind,
            .module = gf.module,
            .is_pub = gf.is_pub,
            .name = gf.name,
            .recv_type = gf.recv_type,
        };
    }

    const mods = try gpa.alloc(Typecheck.GraphModuleInput, n);
    defer gpa.free(mods);
    for (graph.modules, 0..) |*m, i| {
        mods[i] = .{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[i],
            .namespaces = ctx.mods[i].namespaces,
        };
    }

    return Typecheck.checkGraph(gpa, &ctx, mods, fns, graph.entry_index, jobs);
}

/// Bind module `mod`'s import namespaces into `mc.namespaces` (namespace name →
/// imported module id). The namespace name is the `as` alias when present, else
/// the import path's last segment. A selected name is followed to what it names.
fn bindNamespaces(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    mc: *Typecheck.GraphCtx.ModuleCtx,
    mod: u32,
) !void {
    const m = &graph.modules[mod];
    if (m.nodes.len == 0) return;
    const tree = m.tree();
    const prog = m.nodes[Ast.root(m.nodes).int()];
    if (prog.tag != .program) return;
    var selections: std.ArrayList(Typecheck.GraphCtx.Selection) = .empty;
    defer selections.deinit(gpa);
    for (Ast.rangeSlice(tree, prog.lhs.int())) |decl_idx| {
        const decl = m.nodes[decl_idx.int()];
        if (!Ast.isImport(decl.tag)) continue;
        const target = graph.importTarget(mod, decl) orelse continue;
        const is_pub = tree.isPub(decl_idx);
        if (decl.tag == .import_select_decl) {
            for (Ast.importItems(tree, decl)) |item| {
                const local = m.tokens[item.bound().int()].text(m.source);
                const found = graph.exportOf(target, m.tokens[item.name.int()].text(m.source)) orelse continue;
                if (!found.exported) continue;
                if (found.decl == Ast.none) {
                    const gop = try mc.namespaces.getOrPut(gpa, local);
                    if (gop.found_existing) continue;
                    gop.value_ptr.* = found.mod;
                    if (is_pub) try mc.pub_namespaces.put(gpa, local, found.mod);
                    continue;
                }
                const owner = &graph.modules[found.mod];
                const item_decl = owner.nodes[found.decl.int()];
                const kind: @FieldType(Typecheck.GraphCtx.Selection, "kind") = switch (item_decl.tag) {
                    .struct_decl, .tuple_struct_decl, .enum_decl, .type_alias_decl => .type,
                    .protocol_decl => .protocol,
                    else => continue,
                };
                try selections.append(gpa, .{
                    .local = local,
                    .name = owner.tokens[item_decl.main_token].text(owner.source),
                    .target = found.mod,
                    .kind = kind,
                    .exported = is_pub,
                });
            }
            continue;
        }
        const ns_tok: u32 = if (Ast.importAliasTok(decl).unwrap()) |alias| alias.int() else decl.main_token;
        const ns_name = m.tokens[ns_tok].text(m.source);
        // First binding wins (resolve_graph already diagnosed collisions).
        const gop = try mc.namespaces.getOrPut(gpa, ns_name);
        if (gop.found_existing) continue;
        gop.value_ptr.* = target;
        if (is_pub) try mc.pub_namespaces.put(gpa, ns_name, target);
    }
    mc.selections = try selections.toOwnedSlice(gpa);
}

const testing = std.testing;
const Cache = @import("query/Cache.zig");
const Closure = @import("symbols/Closure.zig");
const codes = @import("diagnostics/codes.zig");
const Io = std.Io;

const FixtureFile = struct { path: []const u8, source: []const u8 };

/// A fixture program's discovered graph and its resolve result.
const Fixture = struct {
    graph: Graph.Graph,
    res: ResolveGraph.GraphResult,

    /// Write `files` under a fresh `dir_name` (the caller removes it), then discover the
    /// graph rooted at `entry` and resolve it.
    fn load(gpa: std.mem.Allocator, io: Io, dir_name: []const u8, files: []const FixtureFile, entry: []const u8) !Fixture {
        Io.Dir.cwd().deleteTree(io, dir_name) catch {};
        try Io.Dir.cwd().createDirPath(io, dir_name);

        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        for (files) |f| {
            const full = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir_name, f.path });
            if (std.mem.lastIndexOfScalar(u8, full, '/')) |i|
                try Io.Dir.cwd().createDirPath(io, full[0..i]);
            try Io.Dir.cwd().writeFile(io, .{ .sub_path = full, .data = f.source });
        }

        var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
        const cache_dir = try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir_name});
        const cache = try Cache.init(io, cache_dir);

        var entry_buf: [std.fs.max_path_bytes]u8 = undefined;
        const entry_path = try std.fmt.bufPrint(&entry_buf, "{s}/{s}", .{ dir_name, entry });

        var graph = try Graph.discover(gpa, io, cache, "native", entry_path, .{});
        errdefer graph.deinit(gpa);
        try testing.expect(graph.err == null);
        return .{ .graph = graph, .res = try ResolveGraph.resolveGraph(gpa, &graph) };
    }

    fn deinit(f: *Fixture, gpa: std.mem.Allocator) void {
        f.res.deinit(gpa);
        f.graph.deinit(gpa);
    }
};

/// Write the fixtures, discover the graph, resolve it, typecheck it, and hand
/// `(graph, resolve_result, type_result)` to `check`.
fn withCheckedGraph(
    comptime dir_name: []const u8,
    files: []const FixtureFile,
    entry: []const u8,
    check: *const fn (g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void,
) !void {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    var fx = try Fixture.load(gpa, io, dir_name, files, entry);
    defer fx.deinit(gpa);

    var tr = try checkGraph(gpa, &fx.graph, &fx.res, io, 8);
    defer tr.deinit(gpa);

    try check(&fx.graph, &fx.res, &tr);
}

fn modId(g: *const Graph.Graph, path: []const u8) u32 {
    for (g.modules, 0..) |m, i| if (std.mem.eql(u8, m.path, path)) return @intCast(i);
    unreachable;
}

/// Find the (first) node of a tag in a module.
fn nodeOfTag(g: *const Graph.Graph, mod: u32, tag: Ast.Node.Tag) Ast.Index {
    for (g.modules[mod].nodes, 0..) |n, i| if (n.tag == tag) return Ast.Index.from(@intCast(i));
    unreachable;
}

test "value-poison: a bare module name used as a value reports (poison() asserted)" {
    // `util` is an import namespace misused as a value → "module ... is not a value".
    // Routed through BodyChecker.poison(), whose assert (Debug/ReleaseSafe) proves the
    // poison co-occurs with this fn's reported error; a silent poison would trip it.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int {
        \\ x := util
        \\ return x
        \\}
        \\
        },
        .{ .path = "util.toy", .source = "pub fn helper() -> int { return 1 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            try testing.expect(tr.diags.len >= 1);
            var saw = false;
            for (tr.diags) |d| if (std.mem.indexOf(u8, d.message, "is not a value") != null) {
                saw = true;
            };
            try testing.expect(saw);
        }
    };
    try withCheckedGraph(".toy-test-typ-modval", files, "main.toy", Check.run);
}

test "a method pipe from a module or a module's type is one 'not a value' error" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int {
        \\ _x := util |> .helper()
        \\ _y := util.Shape |> .circle(3)
        \\ return 0
        \\}
        \\
        },
        .{ .path = "util.toy", .source = "pub fn helper() -> int { return 1 }\npub enum Shape { circle(int) }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            try testing.expectEqual(@as(usize, 2), tr.diags.len);
            try testing.expectEqualStrings("module 'util' is not a value, so it cannot be piped; use 'util.helper' directly", tr.diags[0].message);
            try testing.expectEqualStrings("type 'util.Shape' is not a value, so it cannot be piped; use 'util.Shape.circle' directly", tr.diags[1].message);
        }
    };
    try withCheckedGraph(".toy-test-typ-modpipe", files, "main.toy", Check.run);
}

test "another module's struct or enum named as a value is T0056 at the qualified name" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn id[T](x: T) -> T { x }
        \\fn main() -> int {
        \\ _a := util.Shape
        \\ _b := id(util.P)
        \\ _c := util.Shape.circle(1)
        \\ return 0
        \\}
        \\
        },
        .{ .path = "util.toy", .source = "pub enum Shape { circle(int) }\npub struct P { x: int }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            try testing.expectEqual(@as(usize, 2), tr.diags.len);
            try testing.expectEqual(codes.Code.T0056, tr.diags[0].code);
            try testing.expectEqualStrings("type 'util.Shape' is not a value; name one of its variants, as in 'util.Shape.circle(..)'", tr.diags[0].message);
            try testing.expectEqualStrings("type 'util.P' is not a value", tr.diags[1].message);
        }
    };
    try withCheckedGraph(".toy-test-typ-modtype", files, "main.toy", Check.run);
}

test "another module's bare generic type names its associated fns, which T0016 spells qualified" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib as l
        \\fn main() -> int {
        \\ a := l.O.mk(4)
        \\ f: fn(int) -> l.S[int] = l.S.mk
        \\ _g := l.S.mk
        \\ return a.get(0) + f(1).v + l.S.mk(2).v
        \\}
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub enum O[T] { a(T), b }
        \\impl O[T] {
        \\ fn mk(v: T) -> O[T] { O.a(v) }
        \\ fn get(self, d: T) -> T { match self { O.a(v) -> v, O.b -> d } }
        \\}
        \\pub struct S[T] { pub v: T }
        \\impl S[T] { fn mk(v: T) -> S[T] { S { v: v } } }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            try testing.expectEqual(@as(usize, 1), tr.diags.len);
            try testing.expectEqual(codes.Code.T0016, tr.diags[0].code);
            try testing.expectEqualStrings("cannot infer type parameter 'T' for 'l.S.mk'; add explicit type arguments, e.g. l.S[int].mk", tr.diags[0].message);
        }
    };
    try withCheckedGraph(".toy-test-typ-modgeneric", files, "main.toy", Check.run);
}

test "an enum fn named like a variant of another module's enum names the enum qualified" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a as m
        \\protocol Q { fn B(self) -> int }
        \\impl m.E has Q { fn B(self) -> int { 5 } }
        \\fn main() -> int { m.E.A(1).B() }
        \\
        },
        .{ .path = "a.toy", .source = "pub enum E { A(int), B }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            try testing.expectEqual(@as(usize, 1), tr.diags.len);
            try testing.expectEqual(codes.Code.W0016, tr.diags[0].code);
            try testing.expectEqualStrings("method 'B' on 'm.E' has the name of its variant 'B': '.B(..)' on a value calls the method and 'm.E.B' is the variant; rename one", tr.diags[0].message);
        }
    };
    try withCheckedGraph(".toy-test-typ-modvariantfn", files, "main.toy", Check.run);
}

test "cross-module call typechecks clean and yields the callee return type" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int { return util.helper() }
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub fn helper() -> int { return 7 }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = r;
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
            const entry = modId(g, "main");
            // The call node's inferred type is int (helper's return).
            const call = nodeOfTag(g, entry, .call);
            try testing.expectEqual(Typecheck.Kind.int, tr.node_types[entry][call.int()].kind);
        }
    };
    try withCheckedGraph(".toy-test-typ-xcall", files, "main.toy", Check.run);
}

test "same-named structs in two modules get DISTINCT global ids" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a
        \\import b
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "a.toy", .source =
        \\pub struct T { x: int }
        \\
        },
        .{ .path = "b.toy", .source =
        \\pub struct T { x: int, y: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
            // Two USER structs at ids 0/1, then the prelude structs appended last:
            // `char` (id 2), then the managed-box templates `Ref`/`gc_array` (ids 3/4).
            try testing.expectEqual(@as(usize, 5), tr.layouts.len);
            try testing.expectEqualStrings("char", tr.layouts[2].name);
            try testing.expectEqualStrings("Ref", tr.layouts[3].name);
            try testing.expectEqualStrings("gc_array", tr.layouts[4].name);
            // The user layout sizes differ (1 int vs 2 ints) → distinct nominal types.
            const sizes = [_]u32{ tr.layouts[0].size, tr.layouts[1].size };
            try testing.expect(sizes[0] != sizes[1]);
            try testing.expect(sizes[0] == 8 or sizes[1] == 8); // a.T = 8
            try testing.expect(sizes[0] == 16 or sizes[1] == 16); // b.T = 16
        }
    };
    try withCheckedGraph(".toy-test-typ-2t", files, "main.toy", Check.run);
}

test "qualified pub type in a signature resolves to the owning module's layout" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import geometry/rect as r
        \\fn use(b: r.Rect) -> int { return b.w + b.h }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "geometry/rect.toy", .source =
        \\pub struct Rect { pub w: int, pub h: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            // `b.w + b.h` typechecks: the param type `r.Rect` resolved to the rect
            // module's struct layout (fields w, h are int).
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
            const entry = modId(g, "main");
            // The `use` fn's param type is the rect module's struct; confirm the
            // sig param is a @"struct" type whose layout has fields w/h.
            var use_id: ?u32 = null;
            for (r.fns, 0..) |gf, i| if (std.mem.endsWith(u8, gf.name, ".use")) {
                use_id = @intCast(i);
            };
            try testing.expect(use_id != null);
            const sig = tr.sigs[use_id.?];
            try testing.expectEqual(@as(usize, 1), sig.params.len);
            try testing.expectEqual(Typecheck.Kind.@"struct", sig.params[0].kind);
            const lay = tr.layouts[sig.params[0].struct_id];
            try testing.expectEqual(@as(usize, 2), lay.field_names.len);
            _ = entry;
        }
    };
    try withCheckedGraph(".toy-test-typ-qtype", files, "main.toy", Check.run);
}

test "a private field is T0078 at each read, write, literal and call outside its module" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn main() -> int {
        \\ p := lib.mk()
        \\ p.y = 3
        \\ p.y += 1
        \\ q := lib.P { x: 1, y: 2 }
        \\ w := lib.w()
        \\ r := w.0 + w.1
        \\ h := lib.h()
        \\ r += h.f() + h.g()
        \\ b := lib.B { v: 1, n: 2 }
        \\ r += b.v + b.n
        \\ return p.x + q.x + r
        \\}
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub struct P { pub x: int, y: int }
        \\pub struct W(pub int, int)
        \\pub struct H { f: fn() -> int, pub g: fn() -> int }
        \\pub struct B[T] { pub v: T, n: int }
        \\pub fn mk() -> P { P { x: 1, y: 2 } }
        \\pub fn w() -> W { W(1, 2) }
        \\fn one() -> int { 1 }
        \\pub fn h() -> H { H { f: one, g: one } }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            const want = [_][]const u8{
                "field 'y' of 'P' is private to module 'lib'; mark it pub or use a constructor",
                "field 'y' of 'P' is private to module 'lib'; mark it pub or use a constructor",
                "field 'y' of 'P' is private to module 'lib'; mark it pub or use a constructor",
                "field '1' of 'W' is private to module 'lib'; mark it pub or use a constructor",
                "field 'f' of 'H' is private to module 'lib'; mark it pub or use a constructor",
                "field 'n' of 'B' is private to module 'lib'; mark it pub or use a constructor",
                "field 'n' of 'B' is private to module 'lib'; mark it pub or use a constructor",
            };
            try testing.expectEqual(want.len, tr.diags.len);
            const main_mod = modId(g, "main");
            for (want, tr.diags) |w, d| {
                try testing.expectEqual(codes.Code.T0078, d.code);
                try testing.expectEqual(main_mod, d.scope);
                try testing.expectEqualStrings(w, d.message);
            }
        }
    };
    try withCheckedGraph(".toy-test-typ-privfield", files, "main.toy", Check.run);
}

test "outside its module a private field named like a method is T0078 that points at the call" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn take(f: fn() -> int) -> int { f() }
        \\fn apply(f: fn(int, bool) -> int) -> int { f(1, true) }
        \\fn main() -> int {
        \\ h := lib.mk()
        \\ f := h.len
        \\ b := lib.box(2)
        \\ g := b.v
        \\ t := take(h.len) + apply(h.add)
        \\ return h.len() + b.v() + lib.peek(h) + t
        \\}
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub struct H { len: int, add: int }
        \\impl H {
        \\ fn len(self) -> int { self.len * 10 }
        \\ fn add(self, n: int, _b: bool) -> int { self.add + n }
        \\}
        \\pub fn mk() -> H { H { len: 4, add: 1 } }
        \\pub fn peek(h: H) -> int { h.len }
        \\pub struct B[T] { v: T }
        \\impl B[T] { fn v(self) -> T { self.v } }
        \\pub fn box(v: int) -> B[int] { B { v: v } }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            const want = [_][]const u8{
                "field 'len' of 'H' is private to module 'lib'; call 'h.len()' to use the method",
                "field 'v' of 'B' is private to module 'lib'; call 'b.v()' to use the method",
                "field 'len' of 'H' is private to module 'lib'; call the method in a lambda, as in 'fn() { h.len() }'",
                "field 'add' of 'H' is private to module 'lib'; call the method in a lambda, as in 'fn(a, b) { h.add(a, b) }'",
            };
            try testing.expectEqual(want.len, tr.diags.len);
            const main_mod = modId(g, "main");
            for (want, tr.diags) |w, d| {
                try testing.expectEqual(codes.Code.T0078, d.code);
                try testing.expectEqual(main_mod, d.scope);
                try testing.expectEqualStrings(w, d.message);
            }
        }
    };
    try withCheckedGraph(".toy-test-typ-privfield-method", files, "main.toy", Check.run);
}

test "a literal leaving out a private field asks for a constructor, and std's say only private" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\import std/map
        \\fn main() -> int {
        \\ p := lib.P { x: 1 }
        \\ m := Map[int, int].new()
        \\ _h := m.h
        \\ return p.x
        \\}
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub struct P { pub x: int, y: int, z: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            const want = [_][]const u8{
                "'P' has private field 'y', so only module 'lib' can build it; use a constructor",
                "field 'h' of 'Map' is private to module 'std/map'; use the methods of 'Map' instead",
            };
            try testing.expectEqual(want.len, tr.diags.len);
            for (want, tr.diags) |w, d| {
                try testing.expectEqual(codes.Code.T0078, d.code);
                try testing.expectEqualStrings(w, d.message);
            }
        }
    };
    try withCheckedGraph(".toy-test-typ-privfield-missing", files, "main.toy", Check.run);
}

test "a struct's own module, its methods and its derives see every field" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn main() -> int {
        \\ a := lib.mk(1)
        \\ b := lib.mk(1)
        \\ s := a.to_string()
        \\ return if a == b && a.hidden() == 2 { s.len() } else { 0 }
        \\}
        \\
        },
        .{ .path = "lib.toy", .source =
        \\struct Inner { v: int }
        \\pub struct P { x: int, i: Inner }
        \\impl P { fn hidden(self) -> int { self.x + self.i.v } }
        \\pub fn mk(x: int) -> P { P { x: x, i: Inner { v: 1 } } }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            errdefer for (tr.diags) |d| std.debug.print("got: {s}\n", .{d.message});
            // No T0078, and a private field of non-pub `Inner` is no coherence leak.
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
        }
    };
    try withCheckedGraph(".toy-test-typ-privfield-own", files, "main.toy", Check.run);
}

test "pub fn exposing a non-pub return type is a coherence error" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source =
        \\struct Secret { v: int }
        \\pub fn make() -> Secret { return Secret { v: 1 } }
        \\
        },
    };
    // lib is only referenced if imported; import it so it joins the graph.
    const files2 = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source =
        \\struct Secret { v: int }
        \\pub fn make() -> Secret { return Secret { v: 1 } }
        \\
        },
    };
    _ = files;
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            // Exactly one coherence error: pub `make` returns non-pub `Secret`.
            try testing.expectEqual(@as(usize, 1), tr.diags.len);
            try testing.expect(std.mem.indexOf(u8, tr.diags[0].message, "non-pub type") != null);
            try testing.expect(std.mem.indexOf(u8, tr.diags[0].message, "Secret") != null);
        }
    };
    try withCheckedGraph(".toy-test-typ-coherence", files2, "main.toy", Check.run);
}

test "a pub fn naming a pub type in its signature is coherent (no error)" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub struct Ok { v: int }
        \\pub fn make() -> Ok { return Ok { v: 1 } }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
        }
    };
    try withCheckedGraph(".toy-test-typ-coherent-ok", files, "main.toy", Check.run);
}

test "cross-module argument type mismatch is reported against the importer" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int { return util.takes(true) }
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub fn takes(x: int) -> int { return x }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = r;
            try testing.expectEqual(@as(usize, 1), tr.diags.len);
            try testing.expectEqual(modId(g, "main"), tr.diags[0].scope);
            try testing.expect(std.mem.indexOf(u8, tr.diags[0].message, "expected int") != null);
        }
    };
    try withCheckedGraph(".toy-test-typ-argmismatch", files, "main.toy", Check.run);
}

test "3-level cross-module mod.Enum.Variant typechecks to the owning enum type" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import palette as p
        \\fn main() -> int {
        \\ c := p.Color.Red
        \\ return 0
        \\}
        \\
        },
        .{ .path = "palette.toy", .source =
        \\pub enum Color { Red, Green, Blue }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = r;
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
            const entry = modId(g, "main");
            // The OUTER field_access `(p.Color).Red` is typed as the palette enum.
            var saw_enum = false;
            for (g.modules[entry].nodes, 0..) |nn, i| {
                if (nn.tag == .field_access and tr.node_types[entry][i].kind == .@"enum")
                    saw_enum = true;
            }
            try testing.expect(saw_enum);
        }
    };
    try withCheckedGraph(".toy-test-typ-3level", files, "main.toy", Check.run);
}

test "a cross-module struct field type resolves through a qualified field annotation" {
    // A struct in module main has a field of a cross-module pub struct type. The
    // global layout must size it correctly (8 for the nested 1-int struct).
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib as l
        \\struct Wrap { inner: l.Cell, tag: int }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub struct Cell { v: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
            // Find Wrap's layout (the 2-field one): inner (8) + tag (8) = 16.
            var saw_wrap = false;
            for (tr.layouts) |lay| {
                if (std.mem.eql(u8, lay.name, "Wrap")) {
                    saw_wrap = true;
                    try testing.expectEqual(@as(u32, 16), lay.size);
                    // inner is a @"struct" type → its layout is Cell (size 8).
                    try testing.expectEqual(Typecheck.Kind.@"struct", lay.field_types[0].kind);
                    try testing.expectEqual(@as(u32, 8), tr.layouts[lay.field_types[0].struct_id].size);
                }
            }
            try testing.expect(saw_wrap);
        }
    };
    try withCheckedGraph(".toy-test-typ-xfield", files, "main.toy", Check.run);
}

test "a re-exported type is bare through a whole-module import, as if declared there" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int {
        \\    s: Shape = util.mk()
        \\    t := Shape { w: 2 }
        \\    return s.w + t.w
        \\}
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub import shapes.{Shape}
        \\pub fn mk() -> Shape { Shape { w: 3 } }
        \\
        },
        .{ .path = "shapes.toy", .source = "pub struct Shape { pub w: int }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
        }
    };
    try withCheckedGraph(".toy-test-typ-reexport-bare", files, "main.toy", Check.run);
}

test "a type a module only imports is not its member, in any type position" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int {
        \\    x: util.Rect = util.mk()
        \\    y: Option[util.Rect] = Option.none
        \\    match y { Option.some(r) -> r.w, Option.none -> x.w }
        \\}
        \\
        },
        .{ .path = "util.toy", .source =
        \\import util/shapes.{Rect}
        \\pub fn mk() -> Rect { Rect { w: 7 } }
        \\
        },
        .{ .path = "util/shapes.toy", .source = "pub struct Rect { pub w: int }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            // An invalid type argument still cascades (T0013), as an unknown one does.
            var n: usize = 0;
            for (tr.diags) |d| if (d.code == .R0005) {
                try testing.expectEqualStrings("struct 'Rect' is not exported by module 'util'", d.message);
                n += 1;
            };
            try testing.expectEqual(@as(usize, 2), n);
        }
    };
    try withCheckedGraph(".toy-test-typ-plain-select-qualified", files, "main.toy", Check.run);
}

test "a pub type alias is nameable through its module in every type position" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import geo/shapes
        \\import geo
        \\type Span = shapes.Meters
        \\struct Wall { len: shapes.Meters }
        \\pub fn f(m: shapes.Meters, g: fn(geo.Len) -> Span) -> shapes.Meters { g(m) }
        \\fn main() -> int {
        \\    x: Option[geo.shapes.Meters] = Option.some(Wall { len: 2 }.len)
        \\    match x { Option.some(n) -> n, Option.none -> 0 }
        \\}
        \\
        },
        .{ .path = "geo.toy", .source =
        \\pub import geo/shapes.{Meters as Len}
        \\pub import geo/shapes
        \\
        },
        .{ .path = "geo/shapes.toy", .source = "pub type Meters = int\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
            var f_id: ?u32 = null;
            for (r.fns, 0..) |gf, i| if (std.mem.endsWith(u8, gf.name, ".f")) {
                f_id = @intCast(i);
            };
            try testing.expectEqual(Typecheck.Kind.int, tr.sigs[f_id.?].ret.kind);
        }
    };
    try withCheckedGraph(".toy-test-typ-qalias", files, "main.toy", Check.run);
}

test "a private or merely imported type alias is not nameable through the module" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int {
        \\    x: util.Secret = 1
        \\    y: util.Meters = 2
        \\    x + y
        \\}
        \\
        },
        .{ .path = "util.toy", .source =
        \\import util/shapes.{Meters}
        \\type Secret = int
        \\pub fn mk() -> Meters { 1 }
        \\
        },
        .{ .path = "util/shapes.toy", .source = "pub type Meters = int\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            var msgs: [2][]const u8 = undefined;
            var n: usize = 0;
            for (tr.diags) |d| if (d.code == .R0005) {
                if (n < 2) msgs[n] = d.message;
                n += 1;
            };
            try testing.expectEqual(@as(usize, 2), n);
            try testing.expectEqualStrings("type alias 'Secret' is not exported by module 'util'", msgs[0]);
            try testing.expectEqualStrings("type alias 'Meters' is not exported by module 'util'", msgs[1]);
        }
    };
    try withCheckedGraph(".toy-test-typ-qalias-private", files, "main.toy", Check.run);
}

test "a module-qualified variant pattern checks its qualifier like a qualified type" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int {
        \\    match util.mk() {
        \\        util.Other.up -> 1,
        \\        util.S.x -> 4,
        \\        nope.Dir.up -> 2,
        \\        util.Dir.sideways -> 6,
        \\        util.Dir.up -> 7,
        \\        util.Dir.zz.down -> 8,
        \\        zz.Dir.up -> 9,
        \\        _ -> 3,
        \\    }
        \\}
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub enum Dir { up, down }
        \\pub enum Other { up, down }
        \\pub struct S { pub x: int }
        \\pub fn mk() -> Dir { Dir.up }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            const want = [_]struct { codes.Code, []const u8 }{
                .{ .T0067, "pattern enum 'util.Other' does not match scrutinee 'Dir'" },
                .{ .T0062, "'util.S' is not an enum type" },
                .{ .T0002, "unknown module 'nope'" },
                .{ .T0061, "enum 'Dir' has no variant 'sideways'" },
                .{ .T0002, "'util.Dir' is not a module" },
                .{ .T0002, "unknown module 'zz'" },
            };
            var n: usize = 0;
            // The arms after `util.Dir.up` repeat a variant, so W0006 joins the errors.
            for (tr.diags) |d| if (d.code != .W0006) {
                try testing.expect(n < want.len);
                try testing.expectEqual(want[n][0], d.code);
                try testing.expectEqualStrings(want[n][1], d.message);
                n += 1;
            };
            try testing.expectEqual(want.len, n);
        }
    };
    try withCheckedGraph(".toy-test-typ-qpattern", files, "main.toy", Check.run);
}

test "a type alias of an enum or an enum instance qualifies a variant pattern" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\type B = util.Box[int]
        \\type D = util.Dir
        \\fn main() -> int {
        \\    z: B = util.Box.some(4)
        \\    d: D = util.Dir.up
        \\    w: util.B = util.Box.some(5)
        \\    a := match z { B.some(n) -> n, B.none -> 0 }
        \\    b := match d { D.up -> 1, util.D.down -> 0 }
        \\    c := match w { util.B.some(n) -> n, util.B.none -> 0 }
        \\    a + b + c
        \\}
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub enum Box[T] { some(T), none }
        \\pub type B = Box[int]
        \\pub enum Dir { up, down }
        \\pub type D = Dir
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
        }
    };
    try withCheckedGraph(".toy-test-typ-alias-pattern", files, "main.toy", Check.run);
}

test "type arguments on a qualified non-generic type name the type as written" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int {
        \\    x: util.M[int] = 1
        \\    x
        \\}
        \\pub fn b(s: util.SA[int]) -> int { s.x }
        \\pub fn c(s: util.S[int]) -> int { s.x }
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub type M = int
        \\pub struct S { pub x: int }
        \\pub type SA = S
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            const want = [_][]const u8{
                "'M' is not generic; drop the type arguments",
                "'SA' is not generic; drop the type arguments",
                "'S' is not generic; drop the type arguments",
            };
            var n: usize = 0;
            for (tr.diags) |d| if (d.code == .T0001) {
                try testing.expect(n < want.len);
                try testing.expectEqualStrings(want[n], d.message);
                n += 1;
            };
            try testing.expectEqual(want.len, n);
        }
    };
    try withCheckedGraph(".toy-test-typ-qualified-not-generic", files, "main.toy", Check.run);
}

test "an alias a pub import fails to re-export is reported at that import only" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import geo
        \\pub fn f(x: int) -> geo.L { x }
        \\fn main() -> int {
        \\    x: geo.L = 1
        \\    x
        \\}
        \\
        },
        .{ .path = "geo.toy", .source = "pub import shapes.{M as L}\n" },
        .{ .path = "shapes.toy", .source = "type M = int\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            var n: usize = 0;
            for ([_][]const Diagnostic{ r.diags, tr.diags }) |ds| for (ds) |d| if (d.code == .R0005) {
                try testing.expectEqualStrings("type alias 'M' is not exported by module 'shapes'", d.message);
                n += 1;
            };
            try testing.expectEqual(@as(usize, 1), n);
        }
    };
    try withCheckedGraph(".toy-test-typ-alias-withheld", files, "main.toy", Check.run);
}

test "single-module graph typechecks like the single-file checker" {
    const files = &[_]FixtureFile{
        .{ .path = "solo.toy", .source =
        \\struct P { x: int, y: int }
        \\fn area(p: P) -> int { p.x * p.y }
        \\fn main() -> int { q := P { x: 6, y: 7 }
        \\ return area(q) }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            _ = g;
            _ = r;
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
            // The user `struct P` at id 0, then the prelude structs appended last:
            // `char` (id 1) and the managed-box templates `Ref`/`gc_array` (ids 2/3).
            try testing.expectEqual(@as(usize, 4), tr.layouts.len);
            try testing.expectEqual(@as(u32, 16), tr.layouts[0].size);
            try testing.expectEqualStrings("char", tr.layouts[1].name);
            try testing.expectEqualStrings("Ref", tr.layouts[2].name);
            try testing.expectEqualStrings("gc_array", tr.layouts[3].name);
        }
    };
    try withCheckedGraph(".toy-test-typ-solo", files, "solo.toy", Check.run);
}

/// The workers of the -jN check, and the bodies a determinism program gives them: one each.
const racers = 8;

/// `racers` copies of `fmt`, its `{[k]d}` the copy's index. Call at comptime.
fn perRacer(comptime fmt: []const u8) []const u8 {
    @setEvalBranchQuota(100_000);
    var s: []const u8 = "";
    for (0..racers) |k| s = s ++ std.fmt.comptimePrint(fmt, .{ .k = k });
    return s;
}

/// `std.testing.allocator` captures a stack trace on every allocation, which made each
/// determinism test take up to minutes (#34). This one still reports every leak, without
/// the allocation site.
const LeakCheck = std.heap.DebugAllocator(.{ .stack_trace_frames = 0 });

/// Renders one check's result to bytes, and may assert on it. `expectJobInvariant` runs it on
/// the -j1 result and on the -jN result alike.
const Projection = fn (gpa: std.mem.Allocator, g: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) anyerror!void;

/// PARALLEL == SERIAL: discover + resolve `files` ONCE, then type-check the SAME graph twice —
/// under a `.limited(0)` pool (serial, the -j1 baseline) and under a `racers`-worker pool
/// (-jN) — and expect `project` to render both results to the same bytes. The fan-out
/// threshold is 0, so the bodies of even a small program split across the workers.
fn expectJobInvariant(dir_name: []const u8, files: []const FixtureFile, comptime project: Projection) !void {
    var leak_check: LeakCheck = .init;
    const result = jobInvariant(leak_check.allocator(), dir_name, files, project);
    const leaked = leak_check.deinit() == .leak;
    try result;
    try testing.expect(!leaked);
}

fn jobInvariant(gpa: std.mem.Allocator, dir_name: []const u8, files: []const FixtureFile, comptime project: Projection) !void {
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    var fx = try Fixture.load(gpa, io, dir_name, files, "main.toy");
    defer fx.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), fx.res.diags.len);

    const serial = try projectUnder(gpa, &fx, .limited(0), project);
    defer gpa.free(serial);
    const parallel = try projectUnder(gpa, &fx, .limited(racers), project);
    defer gpa.free(parallel);
    try testing.expectEqualSlices(u8, serial, parallel);
}

fn projectUnder(gpa: std.mem.Allocator, fx: *const Fixture, limit: Io.Limit, comptime project: Projection) ![]u8 {
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = limit });
    defer threaded.deinit();
    var tr = try checkGraphWith(gpa, &fx.graph, &fx.res, .{ .io = threaded.io(), .ncpu = racers, .threshold = 0 });
    defer tr.deinit(gpa);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try project(gpa, &fx.graph, &tr, &out);
    return out.toOwnedSlice(gpa);
}

fn expectNoErrors(tr: *const GraphResult) !void {
    for (tr.diags) |d| try testing.expect(d.severity != .err);
}

/// The diagnostic stream: (module, byte_offset, message) at every index. It must also be
/// (module, byte_offset)-monotone, the stable merge key the parallel path has to reproduce.
fn diagStream(gpa: std.mem.Allocator, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
    for (tr.diags) |d| {
        try out.appendSlice(gpa, std.mem.asBytes(&d.scope));
        try out.appendSlice(gpa, std.mem.asBytes(&d.byte_offset));
        try out.appendSlice(gpa, d.message);
        try out.append(gpa, 0);
    }
    try testing.expect(std.sort.isSorted(Diagnostic, tr.diags, {}, struct {
        fn lt(_: void, a: Diagnostic, b: Diagnostic) bool {
            return a.scope < b.scope or (a.scope == b.scope and a.byte_offset < b.byte_offset);
        }
    }.lt));
}

/// Every module's callee slice, then each instance's template id, name and callee slice.
/// Returns the slots counted by tag.
fn calleeTables(gpa: std.mem.Allocator, tr: *const GraphResult, out: *std.ArrayList(u8)) !std.EnumArray(Typecheck.Callee.Tag, usize) {
    var tags: std.EnumArray(Typecheck.Callee.Tag, usize) = .initFill(0);
    for (tr.callees) |cs| {
        try out.appendSlice(gpa, std.mem.sliceAsBytes(cs));
        for (cs) |c| tags.getPtr(c.tag).* += 1;
    }
    for (tr.instances) |inst| {
        try out.appendSlice(gpa, std.mem.asBytes(&inst.template_gid));
        if (inst.name) |name| try out.appendSlice(gpa, name);
        try out.append(gpa, 0);
        try out.appendSlice(gpa, std.mem.asBytes(&inst.node_base));
        try out.appendSlice(gpa, std.mem.sliceAsBytes(inst.callees));
        for (inst.callees) |c| tags.getPtr(c.tag).* += 1;
    }
    return tags;
}

/// The adapter table: each adapter's key, name and target.
fn adapterTable(gpa: std.mem.Allocator, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
    for (tr.adapters) |a| {
        try out.appendSlice(gpa, std.mem.asBytes(&a.key));
        try out.appendSlice(gpa, a.name);
        try out.append(gpa, 0);
        try out.appendSlice(gpa, a.target.name);
        try out.append(gpa, 0);
    }
}

/// The reified layout table: every layout's name, then its fn signature's kind/id bytes when
/// it is a fn pair; then every instance name. Returns the number of fn pairs.
fn fnLayouts(gpa: std.mem.Allocator, tr: *const GraphResult, out: *std.ArrayList(u8)) !usize {
    var pairs: usize = 0;
    for (tr.layouts) |l| {
        try out.appendSlice(gpa, l.name);
        try out.append(gpa, 0);
        if (l.fn_sig) |fs| {
            pairs += 1;
            for (fs.params) |p| try p.appendKeyBytes(gpa, out);
            try fs.ret.appendKeyBytes(gpa, out);
        }
    }
    for (tr.instances) |inst| {
        try out.appendSlice(gpa, inst.name.?);
        try out.append(gpa, 0);
    }
    return pairs;
}

fn typeLabel(tr: *const GraphResult, t: Typecheck.Type) []const u8 {
    return if (t.isStruct()) tr.layouts[t.struct_id].name else @tagName(t.kind);
}

/// The lambda-unit table (with each frame env's layout name), the owners' unboxable and
/// escaping sets, and every lambda node's reified type (by layout name). Returns the number
/// of units with a frame env.
fn lambdaTables(gpa: std.mem.Allocator, g: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !usize {
    var frame_envs: usize = 0;
    for (tr.lambda_units) |u| {
        try out.appendSlice(gpa, u.name);
        try out.append(gpa, 0);
        try out.appendSlice(gpa, std.mem.asBytes(&u.node));
        try out.appendSlice(gpa, std.mem.sliceAsBytes(u.param_slots));
        for (u.sig.params) |p| try out.appendSlice(gpa, typeLabel(tr, p));
        try out.appendSlice(gpa, typeLabel(tr, u.sig.ret));
        try out.appendSlice(gpa, std.mem.asBytes(&u.env_size));
        for (u.env) |es| {
            try out.appendSlice(gpa, std.mem.asBytes(&es.slot));
            try out.appendSlice(gpa, std.mem.asBytes(&es.off));
            try out.appendSlice(gpa, @tagName(Closure.heapStorage(tr.bodies.items[u.owner]).hold(es.slot)));
            try out.appendSlice(gpa, typeLabel(tr, es.ty));
        }
        if (u.frame_env) |sid| {
            try out.appendSlice(gpa, tr.layouts[sid].name);
            frame_envs += 1;
        }
    }
    for (tr.owner_closures) |oc| {
        try out.appendSlice(gpa, std.mem.sliceAsBytes(oc.unboxable));
        try out.append(gpa, 0xff);
    }
    for (g.modules, 0..) |m, mi| for (m.nodes, 0..) |n, k| {
        if (n.tag != .lambda) continue;
        const t = tr.node_types[mi][k];
        if (t.isStruct()) try out.appendSlice(gpa, tr.layouts[t.struct_id].name);
    };
    for (tr.instances) |inst| for (g.modules[inst.mod].nodes[inst.node_base..][0..inst.node_types.len], inst.node_types) |n, t| {
        if (n.tag != .lambda) continue;
        if (t.isStruct()) try out.appendSlice(gpa, tr.layouts[t.struct_id].name);
    };
    return frame_envs;
}

/// Each owner's boxed and unboxable slots, then each lambda unit's env (slot, hold under frame
/// storage, offset) and frame env.
fn ownerClosures(gpa: std.mem.Allocator, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
    for (tr.owner_closures) |oc| {
        try out.print(gpa, "{s}:", .{tr.bodies.items[oc.owner].sig.name});
        for (tr.bodies.items[oc.owner].boxed) |sl| try out.print(gpa, "{d},", .{sl});
        try out.appendSlice(gpa, "u");
        for (oc.unboxable) |sl| try out.print(gpa, "{d},", .{sl});
    }
    for (tr.lambda_units) |u| {
        try out.appendSlice(gpa, u.name);
        const st = Closure.ownerStorage(tr.bodies, tr.owner_closures, u.owner, true);
        for (u.env) |es| try out.print(gpa, "({d} {s} {d})", .{ es.slot, @tagName(st.hold(es.slot)), es.off });
        if (u.frame_env) |sid| try out.print(gpa, "[{s}]", .{tr.layouts[sid].name});
    }
}

test "parallel Pass-C diagnostics are byte-identical to serial (-j1 == -jN)" {
    // Several fns each carrying ONE error, in deliberately source-disordered positions so a
    // thread-arrival ordering would scramble the stream.
    try expectJobInvariant(".toy-test-typ-s4det", &.{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn _a() -> int { return true }
        \\fn _b() -> int { return 1 + true }
        \\fn _e() -> int { return true |> lib.h(1, _) }
        \\fn _c() -> int { return lib.f(true) }
        \\fn _d() -> int { _x: bool = 3
        \\ return 0 }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub fn f(x: int) -> int { return x }
        \\pub fn h(a: int, b: int) -> int { return a + b }
        \\pub fn g() -> bool { return 1 }
        \\
        },
    }, struct {
        fn project(gpa: std.mem.Allocator, _: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
            // The four body errors surface (g's `1`-to-bool is in an UNUSED lib fn but still
            // checked): _a/_b/_c/_d in main + g in lib.
            try testing.expect(tr.diags.len >= 4);
            try diagStream(gpa, tr, out);
        }
    }.project);
}

test "callee tables are byte-identical at -j1 and -jN" {
    const src = "import std/vec\n" ++ Typecheck.dispatch_shapes_src ++
        \\fn _vecs() -> int {
        \\    v := Vec[int].new()
        \\    v.push(1)
        \\    return v.len()
        \\}
        \\
    ++ comptime perRacer("fn _pad{[k]d}(a: Acc) -> int {{ a.plus({[k]d}, 1) + (a |> .plus({[k]d}, 2)) + \"x\".len() + ({[k]d}).hash() }}\n");
    try expectJobInvariant(".toy-test-callees-jn", &.{.{ .path = "main.toy", .source = src }}, struct {
        fn project(gpa: std.mem.Allocator, _: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
            try expectNoErrors(tr);
            const tags = try calleeTables(gpa, tr, out);
            try testing.expect(tags.get(.method) > 0);
            try testing.expect(tags.get(.builtin) > 0);
            try testing.expect(tags.get(.protocol) > 0);
            try testing.expectEqual(@as(usize, 0), tags.get(.pending_method) + tags.get(.pending_bound) + tags.get(.pending_call));
        }
    }.project);
}

const own_generic_src =
    \\import std/vec
    \\struct Box[T] { v: T }
    \\impl Box[T] {
    \\    fn map[U](self, f: fn(T) -> U) -> Box[U] { Box { v: f(self.v) } }
    \\    fn pick[U](self, a: U, _b: U) -> U { a }
    \\    fn with[U](v: T, _tag: U) -> Box[T] { Box { v: v } }
    \\    fn get(self) -> T { self.v }
    \\}
    \\fn _vecs() -> int {
    \\    v := Vec[int].new()
    \\    v.push(1)
    \\    b := Box.with(v.len(), true)
    \\    return b.map(fn(x) { x + 1 }).get() + Box[int].with(2, "s").get()
    \\}
    \\
;

test "callee tables with own-generic instances are byte-identical at -j1 and -jN" {
    // Each body seals calls of one method to three instance rows.
    const src = own_generic_src ++ comptime perRacer("fn _pad{[k]d}(b: Box[int]) -> int {{ b.pick({[k]d}, 1) + (if b.pick(true, false) {{ 1 }} else {{ 0 }}) + b.pick(\"s\", \"t\").len() }}\n");
    try expectJobInvariant(".toy-test-callees-own-jn", &.{.{ .path = "main.toy", .source = src }}, struct {
        fn project(gpa: std.mem.Allocator, _: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
            try expectNoErrors(tr);
            const tags = try calleeTables(gpa, tr, out);
            // Every member call seals to a row: three `pick`s per body over three instances,
            // and 8 in `_vecs` (no instance body makes a method call).
            try testing.expectEqual(@as(usize, racers * 3 + 8), tags.get(.method));
            try testing.expectEqual(@as(usize, 0), tags.get(.pending_method) + tags.get(.pending_bound) + tags.get(.pending_call));
        }
    }.project);
}

test "fn-type reification is identical at -j1 and -jN" {
    // Each body interns one distinct and one shared fn type during the parallel Pass C
    // (inside a `Vec[..]` annotation, so the composite index order depends on which worker
    // gets there first).
    const src = comptime blk: {
        var s: []const u8 =
            \\import std/vec
            \\pub struct H { id: int, run: fn(str) -> bool }
            \\pub fn keep(h: H, _g: fn(H) -> int) -> int { h.id }
            \\
        ;
        const scalars = [_][]const u8{ "int", "bool", "str" };
        for (0..racers) |k| {
            var params: []const u8 = "";
            var rest = k;
            for (0..4) |i| {
                if (i != 0) params = params ++ ", ";
                params = params ++ scalars[rest % 3];
                rest /= 3;
            }
            s = s ++ std.fmt.comptimePrint(
                \\fn _f{d}() -> int {{
                \\    v: Vec[fn({s}) -> int] = []
                \\    w: Vec[fn(int) -> Vec[int]] = []
                \\    return v.len() + w.len()
                \\}}
                \\
            , .{ k, params });
        }
        break :blk s ++ "fn main() -> int { return 0 }\n";
    };
    try expectJobInvariant(".toy-test-fntypes-jn", &.{.{ .path = "main.toy", .source = src }}, struct {
        fn project(gpa: std.mem.Allocator, _: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
            try expectNoErrors(tr);
            // One distinct per body, the shared one, H's field and keep's param.
            try testing.expectEqual(@as(usize, racers + 3), try fnLayouts(gpa, tr, out));
        }
    }.project);
}

test "Vec[fn(int) -> int] mints a trace unit and no hash/eq unit for the pair" {
    try withCheckedGraph(".toy-test-fn-trace", &.{
        .{ .path = "main.toy", .source =
        \\import std/vec
        \\fn main() -> int {
        \\    v: Vec[fn(int) -> int] = []
        \\    return v.len()
        \\}
        \\
        },
    }, "main.toy", struct {
        fn check(_: *const Graph.Graph, _: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            for (tr.diags) |d| try testing.expect(d.severity != .err);
            var traces: usize = 0;
            for (tr.erased_units) |u| {
                if (!@import("layout/Engine.zig").isFnStruct(u.ty, tr.layouts)) continue;
                try testing.expectEqual(@import("symbols/Derive.zig").ErasedKind.trace, u.kind);
                traces += 1;
            }
            try testing.expectEqual(@as(usize, 1), traces);
            var descs: usize = 0;
            for (tr.descriptor_types) |d| if (@import("layout/Engine.zig").isFnStruct(d.ty, tr.layouts)) {
                try testing.expect(!d.hashable);
                descs += 1;
            };
            try testing.expectEqual(@as(usize, 1), descs);
        }
    }.check);
}

test "fn_ref/value callee tables and the adapter table are identical at -j1 and -jN" {
    // Each body names a distinct and a shared fn as a value and calls both; the generic
    // `pass` names one inside its instance body, so the table also depends on the instance
    // slices.
    const src =
        \\import std/vec
        \\fn shared(x: int) -> int { x }
        \\fn pass[T](v: T) -> T {
        \\    f := shared
        \\    f(1)
        \\    v
        \\}
        \\
    ++ comptime perRacer("fn _t{[k]d}(x: int) -> int {{ x + {[k]d} }}\n") ++ perRacer(
        \\fn _u{[k]d}() -> int {{
        \\    fs: Vec[fn(int) -> int] = []
        \\    fs.push(_t{[k]d})
        \\    g := shared
        \\    fs[0](1) + g(2) + pass({[k]d})
        \\}}
        \\
    ) ++ "fn main() -> int { return 0 }\n";
    try expectJobInvariant(".toy-test-fnvalues-jn", &.{.{ .path = "main.toy", .source = src }}, struct {
        fn project(gpa: std.mem.Allocator, _: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
            try expectNoErrors(tr);
            const tags = try calleeTables(gpa, tr, out);
            try adapterTable(gpa, tr, out);
            try testing.expectEqual(@as(usize, racers + 1), tr.adapters.len);
            // A push and a `g := shared` per body, plus one ref per `pass` instance.
            try testing.expect(tags.get(.fn_ref) >= 2 * racers + 1);
            try testing.expect(tags.get(.value) >= 2 * racers);
        }
    }.project);
}

test "closure tables and lambda node types are identical at -j1 and -jN" {
    // Each body has a lambda of a signature of its own (so fn types intern in racing order)
    // and a nested one that captures through its parent, plus two `_` captures typed from
    // their calls; the generic `wrap` holds a capturing lambda, so its units (and their env
    // layouts) follow the instance order. The generic `keep` pushes its arg, so the lambda
    // each body hands it escapes through that instance's summary while the others keep
    // frame envs.
    const src =
        \\import std/vec
        \\pub struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\fn add(a: int, b: int) -> int { a + b }
        \\fn wrap[T](x: T) -> T {
        \\    id := fn(y: T) -> T {
        \\        _k := x
        \\        y
        \\    }
        \\    id(x)
        \\}
        \\fn keep[T](xs: Vec[T], x: T) { xs.push(x) }
        \\
    ++ comptime perRacer("pub struct S{[k]d} {{ v: int }}\n") ++ perRacer(
        \\fn _u{[k]d}() -> int {{
        \\    base := S{[k]d} {{ v: {[k]d} }}
        \\    f := fn(s: S{[k]d}) -> Vec[S{[k]d}] {{
        \\        xs: Vec[S{[k]d}] = []
        \\        xs.push(s)
        \\        xs.push(base)
        \\        xs
        \\    }}
        \\    g := fn() -> fn(int) -> int {{ fn(x) {{ x + base.v }} }}
        \\    acc := Acc {{ n: {[k]d} }}
        \\    h := add(base.v, _)
        \\    p := acc.plus(_)
        \\    ks: Vec[fn() -> int] = []
        \\    keep(ks, fn() -> int {{ base.v }})
        \\    f(S{[k]d} {{ v: 1 }}).len() + g()(1) + wrap({[k]d}) + wrap(S{[k]d} {{ v: 2 }}).v + h(1) + p(2) + ks[0]()
        \\}}
        \\
    ) ++ "fn main() -> int { return 0 }\n";
    try expectJobInvariant(".toy-test-lambdas-jn", &.{.{ .path = "main.toy", .source = src }}, struct {
        fn project(gpa: std.mem.Allocator, g: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
            try expectNoErrors(tr);
            const frame_envs = try lambdaTables(gpa, g, tr, out);
            // Six per `_u` body, plus one per `wrap` instance (int and each S{k}).
            try testing.expectEqual(@as(usize, racers * 6 + racers + 1), tr.lambda_units.len);
            // Per `_u` body every lambda but `g`'s returned one and the kept one; every `id`.
            try testing.expectEqual(@as(usize, racers * 4 + racers + 1), frame_envs);
        }
    }.project);
}

test "owner_closures are identical at -j1 and -jN" {
    // Each body boxes through an assignment, a field store, a `mut self` call and a for-in
    // variable (and copies a read-only capture); the generic `keep` boxes its param in each
    // instance.
    const src =
        \\import std/vec
        \\pub struct C { n: int }
        \\impl C { fn bump(mut self) { self.n = self.n + 1 } }
        \\fn keep[T](x: T, y: T) -> T {
        \\    set := fn() { x = y }
        \\    set()
        \\    x
        \\}
        \\
    ++ comptime perRacer(
        \\pub struct S{[k]d} {{ v: int }}
        \\fn _u{[k]d}() -> int {{
        \\    total := {[k]d}
        \\    ro := 1
        \\    s := S{[k]d} {{ v: 0 }}
        \\    c := C {{ n: 0 }}
        \\    add := fn(v: int) {{
        \\        total = total + v + ro
        \\        s.v = v
        \\        c.bump()
        \\    }}
        \\    xs: Vec[int] = [1, 2]
        \\    fs: Vec[fn() -> int] = []
        \\    for x in xs {{ fs.push(fn() {{
        \\        x = x + 1
        \\        x
        \\    }}) }}
        \\    add(1)
        \\    total + s.v + c.n + fs[0]() + keep({[k]d}, 1) + keep(S{[k]d} {{ v: 1 }}, s).v
        \\}}
        \\
    ) ++ "fn main() -> int { return 0 }\n";
    try expectJobInvariant(".toy-test-owner-closures-jn", &.{.{ .path = "main.toy", .source = src }}, struct {
        fn project(gpa: std.mem.Allocator, _: *const Graph.Graph, tr: *const GraphResult, out: *std.ArrayList(u8)) !void {
            try expectNoErrors(tr);
            try ownerClosures(gpa, tr, out);
            // Every `_u` body, plus the `keep` instances (int and each S{k}).
            try testing.expectEqual(@as(usize, racers + racers + 1), tr.owner_closures.len);
            for (tr.owner_closures) |oc| switch (tr.bodies.items[oc.owner].of) {
                // A `_u` body boxes `total`, `s`, `c` and the for-in `x`, and only `x` is
                // captured by a closure that escapes (pushed).
                .base => {
                    try testing.expectEqual(@as(usize, 4), tr.bodies.items[oc.owner].boxed.len);
                    try testing.expectEqual(@as(usize, 3), oc.unboxable.len);
                },
                // A `keep` instance boxes its `x` for a called closure.
                .instance => {
                    try testing.expectEqualSlices(u32, &.{0}, tr.bodies.items[oc.owner].boxed);
                    try testing.expectEqualSlices(u32, &.{0}, oc.unboxable);
                },
            };
        }
    }.project);
}

test "push on a captured Vec is not a mutation" {
    try withCheckedGraph(".toy-test-capture-vec-push", &.{
        .{ .path = "main.toy", .source =
        \\import std/vec
        \\fn main() -> int {
        \\    xs: Vec[int] = []
        \\    add := fn(v: int) { xs.push(v) }
        \\    add(1)
        \\    xs.push(2)
        \\    return xs.len()
        \\}
        \\
        },
    }, "main.toy", struct {
        fn check(_: *const Graph.Graph, _: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            for (tr.diags) |d| try testing.expect(d.severity != .err);
            try testing.expectEqual(@as(usize, 1), tr.lambda_units.len);
            const u = tr.lambda_units[0];
            try testing.expectEqual(@as(usize, 1), u.env.len);
            try testing.expectEqual(@as(u32, 8), u.env_size);
            // The captured handle: a `Vec` struct whose one field is the shared backing.
            const vl = tr.layouts[u.env[0].ty.struct_id];
            try testing.expect(std.mem.startsWith(u8, vl.name, "Vec"));
            try testing.expect(Typecheck.isRefStruct(vl.field_types[0], tr.layouts));
        }
    }.check);
}

test "fn values in std containers and sinks: a Map key is T0023, Display renders; fs[i](x) calls the element" {
    try withCheckedGraph(".toy-test-fn-std", &.{
        .{ .path = "main.toy", .source =
        \\import std/vec
        \\import std/map
        \\import std/io
        \\type Op = fn(int) -> int
        \\fn inc(x: int) -> int { x + 1 }
        \\fn main() -> int {
        \\    m := Map[Op, int].new()
        \\    io.println(inc)
        \\    fs: Vec[Op] = []
        \\    fs.push(inc)
        \\    i := 0
        \\    return fs[i](1) + m.len()
        \\}
        \\
        },
    }, "main.toy", struct {
        fn check(g: *const Graph.Graph, _: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            var hashable: usize = 0;
            var display: usize = 0;
            for (tr.diags) |d| {
                if (d.severity != .err) continue;
                try testing.expectEqual(@import("diagnostics/codes.zig").Code.T0023, d.code);
                try testing.expect(std.mem.indexOf(u8, d.message, "function type 'fn(int) -> int'") != null);
                if (std.mem.indexOf(u8, d.message, "'Hashable'") != null) hashable += 1;
                if (std.mem.indexOf(u8, d.message, "'Display'") != null) display += 1;
            }
            try testing.expect(hashable >= 1);
            try testing.expectEqual(@as(usize, 0), display);
            const m = &g.modules[g.entry_index];
            var calls: usize = 0;
            for (m.nodes, 0..) |n, k| {
                if (n.tag != .type_app or tr.callees[g.entry_index][k].view() != .value) continue;
                calls += 1;
            }
            try testing.expectEqual(@as(usize, 1), calls);
        }
    }.check);
}

test "W0011 fires at a program's method that a std module in the graph also declares, imported directly or not" {
    // std/str reaches the program only through `lib`, and still owns a `trim` on `str`.
    // `Vec.each` and `Option.or_else` shadow; `Vec.total` and the program's own `Bag.each`
    // shadow nothing.
    try withCheckedGraph(".toy-test-std-shadow", &.{
        .{ .path = "main.toy", .source =
        \\import std/vec
        \\import std/option
        \\import lib
        \\struct Bag { n: int }
        \\impl Bag { fn each(self) -> int { self.n } }
        \\impl Vec[T] {
        \\    fn each(self, f: fn(T)) { f(self.get(0).expect("no value")) }
        \\    fn total(self) -> int { self.len() }
        \\}
        \\impl Option[T] { fn or_else(self, f: fn() -> T) -> T { f() } }
        \\impl str { fn trim(self) -> str { self } }
        \\fn main() -> int {
        \\    xs: Vec[int] = [1]
        \\    b := Bag { n: 1 }
        \\    return xs.total() + b.each() + lib.words("a b".trim())
        \\}
        \\
        },
        .{ .path = "lib.toy", .source =
        \\import std/str
        \\pub fn words(s: str) -> int { s.split(" ").len() }
        \\
        },
    }, "main.toy", struct {
        fn check(g: *const Graph.Graph, _: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            errdefer for (tr.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
            const ds = tr.diags;
            try testing.expectEqual(@as(usize, 3), ds.len);
            for (ds) |d| {
                try testing.expectEqual(@import("diagnostics/codes.zig").Code.W0011, d.code);
                try testing.expectEqual(@import("diagnostics/model.zig").Severity.warning, d.severity);
                try testing.expectEqual(g.entry_index, d.scope);
            }
            try testing.expectEqualStrings("method 'each' on 'Vec' shadows the one in std/vec: the program's calls reach this one, std's own calls keep std's; rename it to keep both", ds[0].message);
            try testing.expect(std.mem.startsWith(u8, ds[1].message, "method 'or_else' on 'Option' shadows the one in std/option:"));
            try testing.expect(std.mem.startsWith(u8, ds[2].message, "method 'trim' on 'str' shadows the one in std/str:"));
            const m = &g.modules[g.entry_index];
            for (ds, [_][]const u8{ "each", "or_else", "trim" }) |d, name|
                try testing.expectEqualStrings(name, m.source[d.byte_offset..d.end]);
            for (tr.templates) |tm| try testing.expectEqual(!tm.bundled and !std.mem.eql(u8, tm.name, "total"), tm.shadows);
            for (tr.methods) |mth| if (mth.instance == null)
                try testing.expectEqual(!mth.bundled and std.mem.eql(u8, mth.name, "trim"), mth.shadows);
        }
    }.check);
    // With no std/str in the graph, the same `impl str { fn trim }` shadows nothing.
    try withCheckedGraph(".toy-test-std-shadow-none", &.{
        .{ .path = "main.toy", .source =
        \\impl str { fn trim(self) -> str { self } }
        \\fn main() -> int { return "a".trim().len() }
        \\
        },
    }, "main.toy", struct {
        fn check(_: *const Graph.Graph, _: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            try testing.expectEqual(@as(usize, 0), tr.diags.len);
        }
    }.check);
}

test "W0011 fires at a program's method named like a builtin of a builtin type, whatever the imports" {
    // `str.len`, `int.to_string` and `Option.or` are builtins; `str.shout` is a new
    // name, and `to_string` on the program's own `Tag` is how `Tag` implements it.
    try withCheckedGraph(".toy-test-builtin-shadow", &.{
        .{ .path = "main.toy", .source =
        \\struct Tag { n: int }
        \\impl Tag { fn to_string(self) -> str { "tag" } }
        \\impl str {
        \\    fn len(self) -> int { 3 }
        \\    fn shout(self) -> str { self }
        \\}
        \\impl int { fn to_string(self) -> str { "n" } }
        \\impl Option[T] { fn or(self, d: T) -> T { d } }
        \\fn main() -> int {
        \\    t := Tag { n: 1 }
        \\    return "ab".len() + "x".shout().len() + 7.to_string().len() + t.to_string().len() + Option.some(4).or(5)
        \\}
        \\
        },
    }, "main.toy", struct {
        fn check(g: *const Graph.Graph, _: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            errdefer for (tr.diags) |d| std.debug.print("got {s}: {s}\n", .{ @tagName(d.code), d.message });
            const ds = tr.diags;
            try testing.expectEqual(@as(usize, 3), ds.len);
            try testing.expectEqualStrings("method 'len' on 'str' shadows the built-in one: the program's calls reach this one, std's own calls keep the built-in; rename it to keep both", ds[0].message);
            try testing.expect(std.mem.startsWith(u8, ds[1].message, "method 'to_string' on 'int' shadows the built-in one:"));
            try testing.expect(std.mem.startsWith(u8, ds[2].message, "method 'or' on 'Option' shadows the built-in one:"));
            const m = &g.modules[g.entry_index];
            for (ds, [_][]const u8{ "len", "to_string", "or" }) |d, name| {
                try testing.expectEqual(@import("diagnostics/codes.zig").Code.W0011, d.code);
                try testing.expectEqualStrings(name, m.source[d.byte_offset..d.end]);
            }
        }
    }.check);
}

test "a call in a bundled module never reaches a program's shadowing method; the program's own call does" {
    // The program's `len` and `byte_at` on `str` and `or` on `Option` would steer
    // `trim`, `split` and `join` if std's calls reached them.
    try withCheckedGraph(".toy-test-bundled-site", &.{
        .{ .path = "main.toy", .source =
        \\import std/str
        \\impl str {
        \\    fn len(self) -> int { 100000 }
        \\    fn byte_at(self, _i: int) -> int { 120 }
        \\}
        \\impl Option[T] { fn or(self, d: T) -> T { d } }
        \\fn main() -> int {
        \\    return "ab".len() + " a ".trim().concat(",".join("a b".split(" "))).byte_at(0) + Option.some(1).or(2)
        \\}
        \\
        },
    }, "main.toy", struct {
        fn check(g: *const Graph.Graph, _: *ResolveGraph.GraphResult, tr: *GraphResult) anyerror!void {
            const shadowing = struct {
                fn any(t: *const GraphResult, cs: []const Typecheck.Callee) bool {
                    for (cs) |c| switch (c.view()) {
                        .method, .bound_method => |id| if (t.methods[id].shadows) return true,
                        else => {},
                    };
                    return false;
                }
            };
            var builtin_len: usize = 0;
            for (g.modules, tr.callees) |mod, cs| {
                if (!mod.bundled) continue;
                try testing.expect(!shadowing.any(tr, cs));
                for (cs) |c| if (std.meta.eql(c.view(), Typecheck.Callee.View{ .builtin = .str_len })) {
                    builtin_len += 1;
                };
            }
            try testing.expect(builtin_len > 0);
            for (tr.instances) |inst| if (g.modules[inst.mod].bundled) try testing.expect(!shadowing.any(tr, inst.callees));
            // The program's calls reach its own `len`, `byte_at` and `or`.
            var own: usize = 0;
            for (tr.callees[g.entry_index]) |c| switch (c.view()) {
                .method => |id| own += @intFromBool(tr.methods[id].shadows),
                else => {},
            };
            try testing.expectEqual(@as(usize, 3), own);
        }
    }.check);
}
