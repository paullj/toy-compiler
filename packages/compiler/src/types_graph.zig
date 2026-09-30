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
    /// Pass-C worker pool, or `null` to force serial (see `Typecheck.checkGraph`).
    /// The single-source path (per-file pipeline + inline test helpers) passes null.
    io: ?std.Io,
    /// The `-j` jobs knob for the Pass-C body fan-out (0 => host cpus). Ignored when
    /// `io == null`. See `Typecheck.ncpu`.
    ncpu: usize,
) !GraphResult {
    const n = graph.modules.len;

    var ctx: Typecheck.GraphCtx = .{ .mods = try gpa.alloc(Typecheck.GraphCtx.ModuleCtx, n) };
    defer {
        for (ctx.mods) |*mc| {
            mc.struct_ids.deinit(gpa);
            mc.enum_ids.deinit(gpa);
            mc.protocol_ids.deinit(gpa);
            mc.namespaces.deinit(gpa);
            mc.alias_ids.deinit(gpa);
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
        };
    }

    // Bind each module's import namespaces (namespace name → imported module id)
    // by rebuilding the `/`-joined import path and matching a module's canonical
    // name. Mirrors resolve_graph's binding so the type checker sees the same
    // namespaces (an unused or duplicate import is harmless here — resolve already
    // diagnosed collisions). Determinism: decls walked in source order.
    for (graph.modules, 0..) |*m, mi| {
        try bindNamespaces(gpa, graph, &ctx.mods[mi], m);
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

    return Typecheck.checkGraph(gpa, &ctx, mods, fns, graph.entry_index, io, ncpu);
}

/// Bind module `m`'s import namespaces into `mc.namespaces` (namespace name →
/// imported module id). The namespace name is the `as` alias when present, else
/// the import path's last segment.
fn bindNamespaces(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    mc: *Typecheck.GraphCtx.ModuleCtx,
    m: *const Graph.Module,
) !void {
    if (m.nodes.len == 0) return;
    const tree = m.tree();
    const prog = m.nodes[Ast.root(m.nodes).int()];
    if (prog.tag != .program) return;
    for (Ast.rangeSlice(tree, prog.lhs.int())) |decl_idx| {
        const decl = m.nodes[decl_idx.int()];
        if (decl.tag != .import_decl) continue;
        const target = importTarget(graph, m, decl) orelse continue;
        const ns_tok: u32 = if (Ast.importAliasTok(decl).unwrap()) |alias| alias.int() else decl.main_token;
        const ns_name = m.tokens[ns_tok].text(m.source);
        // First binding wins (resolve_graph already diagnosed collisions).
        const gop = try mc.namespaces.getOrPut(gpa, ns_name);
        if (!gop.found_existing) gop.value_ptr.* = target;
    }
}

/// The graph module id an `import_decl` in module `m` refers to (rebuild its
/// `/`-joined path, match a module's canonical name), or null if unresolved.
fn importTarget(graph: *const Graph.Graph, m: *const Graph.Module, decl: Ast.Node) ?u32 {
    const tree = m.tree();
    const segs = Ast.importPathToks(tree, decl);
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var len: usize = 0;
    for (segs, 0..) |seg, i| {
        const txt = m.tokens[seg.int()].text(m.source);
        if (i != 0) {
            if (len >= buf.len) return null;
            buf[len] = '/';
            len += 1;
        }
        if (len + txt.len > buf.len) return null;
        @memcpy(buf[len .. len + txt.len], txt);
        len += txt.len;
    }
    const path = buf[0..len];
    for (graph.modules, 0..) |om, oi| {
        if (std.mem.eql(u8, om.path, path)) return @intCast(oi);
    }
    return null;
}

const testing = std.testing;
const Cache = @import("query/Cache.zig");
const Io = std.Io;

const FixtureFile = struct { path: []const u8, source: []const u8 };

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

    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

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
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);

    var r = try ResolveGraph.resolveGraph(gpa, &graph);
    defer r.deinit(gpa);

    var tr = try checkGraph(gpa, &graph, &r, io, 8);
    defer tr.deinit(gpa);

    try check(&graph, &r, &tr);
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
        \\pub struct Rect { w: int, h: int }
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

/// PARALLEL == SERIAL: discover+resolve ONCE, then type-check the SAME graph
/// twice — under a `.limited(0)` (serial, the -j1 baseline) pool and under a
/// multi-worker (`.limited(8)`, the -jN) pool — and assert the diagnostic stream is
/// IDENTICAL: same count, same (module, byte_offset, message) at every index. A
/// multi-fn, multi-module reject program with errors in source-disordered fns is the
/// real test of the stable (module, byte_offset) merge sort vs thread-arrival order.
fn diagsUnder(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *ResolveGraph.GraphResult,
    limit: Io.Limit,
) ![]Diagnostic {
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = limit });
    defer threaded.deinit();
    var tr = try checkGraph(gpa, graph, res, threaded.io(), 8);
    defer tr.deinit(gpa);
    const copy = try gpa.alloc(Diagnostic, tr.diags.len);
    for (tr.diags, 0..) |d, i| copy[i] = .{
        .scope = d.scope,
        .byte_offset = d.byte_offset,
        .end = d.end,
        .message = try gpa.dupe(u8, d.message),
    };
    return copy;
}

test "parallel Pass-C diagnostics are byte-identical to serial (-j1 == -jN)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-typ-s4det";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const files = &[_]FixtureFile{
        // Several fns each carrying ONE error, in deliberately source-disordered
        // positions so a thread-arrival ordering would scramble the stream.
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn _a() -> int { return true }
        \\fn _b() -> int { return 1 + true }
        \\fn _c() -> int { return lib.f(true) }
        \\fn _d() -> int { _x: bool = 3
        \\ return 0 }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source =
        \\pub fn f(x: int) -> int { return x }
        \\pub fn g() -> bool { return 1 }
        \\
        },
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    for (files) |f| {
        const full = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ dir, f.path });
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = full, .data = f.source });
    }

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir});
    const cache = try Cache.init(io, cache_dir);

    var entry_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry_path = try std.fmt.bufPrint(&entry_buf, "{s}/main.toy", .{dir});

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path, .{});
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);

    var r = try ResolveGraph.resolveGraph(gpa, &graph);
    defer r.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), r.diags.len);

    const serial = try diagsUnder(gpa, &graph, &r, .limited(0));
    defer {
        for (serial) |d| gpa.free(@constCast(d.message));
        gpa.free(serial);
    }
    const parallel = try diagsUnder(gpa, &graph, &r, .limited(8));
    defer {
        for (parallel) |d| gpa.free(@constCast(d.message));
        gpa.free(parallel);
    }

    // The four body errors surface (g's `1`-to-bool is in an UNUSED lib fn but still
    // checked): _a/_b/_c/_d in main + g in lib. Same count, same order, same bytes.
    try testing.expect(serial.len >= 4);
    try testing.expectEqual(serial.len, parallel.len);
    for (serial, parallel) |s, p| {
        try testing.expectEqual(s.scope, p.scope);
        try testing.expectEqual(s.byte_offset, p.byte_offset);
        try testing.expectEqualStrings(s.message, p.message);
    }

    // And the serial stream is itself (scope, byte_offset)-monotone (the stable
    // merge key) — the property the parallel path must reproduce.
    for (serial[1..], serial[0 .. serial.len - 1]) |cur, prev| {
        try testing.expect(prev.scope < cur.scope or
            (prev.scope == cur.scope and prev.byte_offset <= cur.byte_offset));
    }
}

/// The callee tables of one check of `graph` under a `limit`-worker pool, as raw bytes:
/// every module slice, then each instance's `template_gid` and slice. `tags` counts
/// every slot by tag. Requires an error-free check.
fn calleesUnder(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *ResolveGraph.GraphResult,
    limit: Io.Limit,
    tags: *[@typeInfo(Typecheck.Callee.Tag).@"enum".fields.len]usize,
) ![]u8 {
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = limit });
    defer threaded.deinit();
    var tr = try checkGraph(gpa, graph, res, threaded.io(), 8);
    defer tr.deinit(gpa);
    for (tr.diags) |d| try testing.expect(d.severity != .err);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    @memset(tags, 0);
    for (tr.callees) |cs| {
        try out.appendSlice(gpa, std.mem.sliceAsBytes(cs));
        for (cs) |c| tags[@intFromEnum(c.tag)] += 1;
    }
    for (tr.instances) |inst| {
        try out.appendSlice(gpa, std.mem.asBytes(&inst.template_gid));
        try out.appendSlice(gpa, std.mem.sliceAsBytes(inst.callees));
        for (inst.callees) |c| tags[@intFromEnum(c.tag)] += 1;
    }
    return out.toOwnedSlice(gpa);
}

test "callee tables are byte-identical at -j1 and -jN" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-callees-jn";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Every Callee shape, plus enough fns (> the Pass-C fan-out threshold) that the
    // parallel check really splits the bodies across workers.
    var src: std.Io.Writer.Allocating = .init(gpa);
    defer src.deinit();
    try src.writer.writeAll("import std/vec\n");
    try src.writer.writeAll(Typecheck.dispatch_shapes_src);
    try src.writer.writeAll(
        \\fn _vecs() -> int {
        \\    v := Vec[int].new()
        \\    v.push(1)
        \\    return v.len()
        \\}
        \\
    );
    for (0..70) |k| try src.writer.print("fn _pad{d}(a: Acc) -> int {{ a.plus({d}, 1) + \"x\".len() + ({d}).hash() }}\n", .{ k, k, k });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry_path = try std.fmt.bufPrint(&path_buf, "{s}/main.toy", .{dir});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = entry_path, .data = src.written() });

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache = try Cache.init(io, try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir}));

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path, .{});
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);
    var r = try ResolveGraph.resolveGraph(gpa, &graph);
    defer r.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), r.diags.len);

    var serial_tags: [@typeInfo(Typecheck.Callee.Tag).@"enum".fields.len]usize = undefined;
    var parallel_tags: [@typeInfo(Typecheck.Callee.Tag).@"enum".fields.len]usize = undefined;
    const serial = try calleesUnder(gpa, &graph, &r, .limited(0), &serial_tags);
    defer gpa.free(serial);
    const parallel = try calleesUnder(gpa, &graph, &r, .limited(8), &parallel_tags);
    defer gpa.free(parallel);

    try testing.expectEqualSlices(u8, serial, parallel);
    try testing.expect(serial_tags[@intFromEnum(Typecheck.Callee.Tag.method)] > 0);
    try testing.expect(serial_tags[@intFromEnum(Typecheck.Callee.Tag.builtin)] > 0);
    try testing.expect(serial_tags[@intFromEnum(Typecheck.Callee.Tag.protocol)] > 0);
    try testing.expectEqual(@as(usize, 0), serial_tags[@intFromEnum(Typecheck.Callee.Tag.pending)]);
}

/// The reified layout table of one check of `graph` under a `limit`-worker pool, as
/// bytes: every layout's name, then its fn signature's kind/id bytes when it is a fn
/// pair; then every instance name. Requires an error-free check.
fn fnLayoutsUnder(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *ResolveGraph.GraphResult,
    limit: Io.Limit,
    pairs: *usize,
) ![]u8 {
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = limit });
    defer threaded.deinit();
    var tr = try checkGraph(gpa, graph, res, threaded.io(), 8);
    defer tr.deinit(gpa);
    for (tr.diags) |d| try testing.expect(d.severity != .err);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    pairs.* = 0;
    for (tr.layouts) |l| {
        try out.appendSlice(gpa, l.name);
        try out.append(gpa, 0);
        if (l.fn_sig) |fs| {
            pairs.* += 1;
            for (fs.params) |p| try p.appendKeyBytes(gpa, &out);
            try fs.ret.appendKeyBytes(gpa, &out);
        }
    }
    for (tr.instances) |inst| {
        try out.appendSlice(gpa, inst.name.?);
        try out.append(gpa, 0);
    }
    return out.toOwnedSlice(gpa);
}

test "fn-type reification is identical at -j1 and -jN" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-fntypes-jn";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // Each body interns one distinct and one shared fn type during the parallel Pass C
    // (inside a `Vec[..]` annotation, so the composite index order depends on which
    // worker gets there first), across more fns than the fan-out threshold.
    var src: std.Io.Writer.Allocating = .init(gpa);
    defer src.deinit();
    try src.writer.writeAll(
        \\import std/vec
        \\pub struct H { id: int, run: fn(str) -> bool }
        \\pub fn keep(h: H, _g: fn(H) -> int) -> int { h.id }
        \\
    );
    const scalars = [_][]const u8{ "int", "bool", "str" };
    for (0..70) |k| {
        try src.writer.print("fn _f{d}() -> int {{\n    v: Vec[fn(", .{k});
        var rest = k;
        for (0..4) |i| {
            if (i != 0) try src.writer.writeAll(", ");
            try src.writer.writeAll(scalars[rest % 3]);
            rest /= 3;
        }
        try src.writer.writeAll(
            \\) -> int] = []
            \\    w: Vec[fn(int) -> Vec[int]] = []
            \\    return v.len() + w.len()
            \\}
            \\
        );
    }
    try src.writer.writeAll("fn main() -> int { return 0 }\n");

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry_path = try std.fmt.bufPrint(&path_buf, "{s}/main.toy", .{dir});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = entry_path, .data = src.written() });

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache = try Cache.init(io, try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir}));

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path, .{});
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);
    var r = try ResolveGraph.resolveGraph(gpa, &graph);
    defer r.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), r.diags.len);

    var serial_pairs: usize = 0;
    var parallel_pairs: usize = 0;
    const serial = try fnLayoutsUnder(gpa, &graph, &r, .limited(0), &serial_pairs);
    defer gpa.free(serial);
    const parallel = try fnLayoutsUnder(gpa, &graph, &r, .limited(8), &parallel_pairs);
    defer gpa.free(parallel);

    try testing.expectEqualSlices(u8, serial, parallel);
    // 70 distinct, the shared one, H's field and keep's param.
    try testing.expectEqual(@as(usize, 73), serial_pairs);
    try testing.expectEqual(serial_pairs, parallel_pairs);
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

/// The callee tables plus the adapter table of one check of `graph` under a
/// `limit`-worker pool, as bytes. Requires an error-free check.
fn closureTablesUnder(
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,
    res: *ResolveGraph.GraphResult,
    limit: Io.Limit,
    tags: *[@typeInfo(Typecheck.Callee.Tag).@"enum".fields.len]usize,
    n_adapters: *usize,
) ![]u8 {
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = limit });
    defer threaded.deinit();
    var tr = try checkGraph(gpa, graph, res, threaded.io(), 8);
    defer tr.deinit(gpa);
    for (tr.diags) |d| try testing.expect(d.severity != .err);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    @memset(tags, 0);
    for (tr.callees) |cs| {
        try out.appendSlice(gpa, std.mem.sliceAsBytes(cs));
        for (cs) |c| tags[@intFromEnum(c.tag)] += 1;
    }
    for (tr.instances) |inst| {
        try out.appendSlice(gpa, inst.name.?);
        try out.appendSlice(gpa, std.mem.sliceAsBytes(inst.callees));
        for (inst.callees) |c| tags[@intFromEnum(c.tag)] += 1;
    }
    for (tr.adapters) |a| {
        try out.appendSlice(gpa, std.mem.asBytes(&a.key));
        try out.appendSlice(gpa, a.name);
        try out.append(gpa, 0);
        try out.appendSlice(gpa, a.target.name);
        try out.append(gpa, 0);
    }
    n_adapters.* = tr.adapters.len;
    return out.toOwnedSlice(gpa);
}

test "fn_ref/value callee tables and the adapter table are identical at -j1 and -jN" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-fnvalues-jn";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // More bodies than the Pass-C fan-out threshold, each naming a distinct and a
    // shared fn as a value and calling both; the generic `pass` names one inside its
    // instance body, so the table also depends on the instance slices.
    var src: std.Io.Writer.Allocating = .init(gpa);
    defer src.deinit();
    try src.writer.writeAll(
        \\import std/vec
        \\fn shared(x: int) -> int { x }
        \\fn pass[T](v: T) -> T {
        \\    f := shared
        \\    f(1)
        \\    v
        \\}
        \\
    );
    for (0..70) |k| try src.writer.print("fn _t{d}(x: int) -> int {{ x + {d} }}\n", .{ k, k });
    for (0..70) |k| try src.writer.print(
        \\fn _u{d}() -> int {{
        \\    fs: Vec[fn(int) -> int] = []
        \\    fs.push(_t{d})
        \\    g := shared
        \\    fs[0](1) + g(2) + pass({d})
        \\}}
        \\
    , .{ k, k, k });
    try src.writer.writeAll("fn main() -> int { return 0 }\n");

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry_path = try std.fmt.bufPrint(&path_buf, "{s}/main.toy", .{dir});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = entry_path, .data = src.written() });

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache = try Cache.init(io, try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir}));

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path, .{});
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);
    var r = try ResolveGraph.resolveGraph(gpa, &graph);
    defer r.deinit(gpa);

    const n_tags = @typeInfo(Typecheck.Callee.Tag).@"enum".fields.len;
    var serial_tags: [n_tags]usize = undefined;
    var parallel_tags: [n_tags]usize = undefined;
    var serial_n: usize = 0;
    var parallel_n: usize = 0;
    const serial = try closureTablesUnder(gpa, &graph, &r, .limited(0), &serial_tags, &serial_n);
    defer gpa.free(serial);
    const parallel = try closureTablesUnder(gpa, &graph, &r, .limited(8), &parallel_tags, &parallel_n);
    defer gpa.free(parallel);

    try testing.expectEqualSlices(u8, serial, parallel);
    try testing.expectEqual(@as(usize, 71), serial_n);
    try testing.expectEqual(serial_n, parallel_n);
    // 70 pushes + 70 `g := shared` + one ref per `pass` instance.
    try testing.expect(serial_tags[@intFromEnum(Typecheck.Callee.Tag.fn_ref)] >= 141);
    try testing.expect(serial_tags[@intFromEnum(Typecheck.Callee.Tag.value)] >= 140);
}

/// The lambda-unit table and every lambda node's reified type (by layout name) of one
/// check of `graph` under a `limit`-worker pool, as bytes. Requires an error-free check.
fn lambdaTablesUnder(gpa: std.mem.Allocator, graph: *const Graph.Graph, res: *ResolveGraph.GraphResult, limit: Io.Limit, n_units: *usize) ![]u8 {
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = limit });
    defer threaded.deinit();
    var tr = try checkGraph(gpa, graph, res, threaded.io(), 8);
    defer tr.deinit(gpa);
    for (tr.diags) |d| try testing.expect(d.severity != .err);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (tr.lambda_units) |u| {
        try out.appendSlice(gpa, u.name);
        try out.append(gpa, 0);
        try out.appendSlice(gpa, std.mem.asBytes(&u.node));
        try out.appendSlice(gpa, std.mem.sliceAsBytes(u.param_slots));
        for (u.sig.params) |p| try out.appendSlice(gpa, if (p.isStruct()) tr.layouts[p.struct_id].name else @tagName(p.kind));
        try out.appendSlice(gpa, if (u.sig.ret.isStruct()) tr.layouts[u.sig.ret.struct_id].name else @tagName(u.sig.ret.kind));
    }
    for (graph.modules, 0..) |m, mi| for (m.nodes, 0..) |n, k| {
        if (n.tag != .lambda) continue;
        const t = tr.node_types[mi][k];
        if (t.isStruct()) try out.appendSlice(gpa, tr.layouts[t.struct_id].name);
    };
    for (tr.instances) |inst| for (graph.modules[inst.mod].nodes, 0..) |n, k| {
        if (n.tag != .lambda) continue;
        const t = inst.node_types[k];
        if (t.isStruct()) try out.appendSlice(gpa, tr.layouts[t.struct_id].name);
    };
    n_units.* = tr.lambda_units.len;
    return out.toOwnedSlice(gpa);
}

test "closure tables and lambda node types are identical at -j1 and -jN" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-lambdas-jn";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    // More bodies than the Pass-C fan-out threshold, each with a lambda of a signature of
    // its own (so fn types intern in racing order) and a nested one; the generic `wrap`
    // holds a lambda, so its units follow the instance order.
    var src: std.Io.Writer.Allocating = .init(gpa);
    defer src.deinit();
    try src.writer.writeAll(
        \\import std/vec
        \\pub struct S0 { v: int }
        \\fn wrap[T](x: T) -> T {
        \\    id := fn(y: T) -> T { y }
        \\    id(x)
        \\}
        \\
    );
    for (1..70) |k| try src.writer.print("pub struct S{d} {{ v: int }}\n", .{k});
    for (0..70) |k| try src.writer.print(
        \\fn _u{d}() -> int {{
        \\    f := fn(s: S{d}) -> Vec[S{d}] {{
        \\        xs: Vec[S{d}] = []
        \\        xs.push(s)
        \\        xs
        \\    }}
        \\    g := fn() -> fn(int) -> int {{ fn(x) {{ x + {d} }} }}
        \\    f(S{d} {{ v: 1 }}).len() + g()(1) + wrap({d}) + wrap(S{d} {{ v: 2 }}).v
        \\}}
        \\
    , .{ k, k, k, k, k, k, k, k });
    try src.writer.writeAll("fn main() -> int { return 0 }\n");

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry_path = try std.fmt.bufPrint(&path_buf, "{s}/main.toy", .{dir});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = entry_path, .data = src.written() });

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache = try Cache.init(io, try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir}));

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path, .{});
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);
    var r = try ResolveGraph.resolveGraph(gpa, &graph);
    defer r.deinit(gpa);

    var serial_n: usize = 0;
    var parallel_n: usize = 0;
    const serial = try lambdaTablesUnder(gpa, &graph, &r, .limited(0), &serial_n);
    defer gpa.free(serial);
    const parallel = try lambdaTablesUnder(gpa, &graph, &r, .limited(8), &parallel_n);
    defer gpa.free(parallel);
    try testing.expectEqualSlices(u8, serial, parallel);
    // Three per `_u` body, plus one per `wrap` instance (int and each S{k}).
    try testing.expectEqual(@as(usize, 70 * 3 + 71), serial_n);
    try testing.expectEqual(serial_n, parallel_n);
}

test "fn values in std containers and sinks: Map key and Display are T0023; fs[i](x) calls the element" {
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
            try testing.expectEqual(@as(usize, 1), display);
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
