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
        }
        gpa.free(ctx.mods);
    }
    for (graph.modules, 0..) |*m, i| {
        ctx.mods[i] = .{
            .tree = m.tree(),
            .tokens = m.tokens,
            .source = m.source,
            .resolutions = res.resolutions[i],
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

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path, null);
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
            // Two structs, two distinct global layout ids; b's T has 2 fields.
            try testing.expectEqual(@as(usize, 2), tr.layouts.len);
            // The layout sizes differ (1 int vs 2 ints) → distinct nominal types.
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
            try testing.expectEqual(@as(usize, 1), tr.layouts.len);
            try testing.expectEqual(@as(u32, 16), tr.layouts[0].size);
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
        \\fn a() -> int { return true }
        \\fn b() -> int { return 1 + true }
        \\fn c() -> int { return lib.f(true) }
        \\fn d() -> int { x: bool = 3
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

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path, null);
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
    // checked): a/b/c/d in main + g in lib. Same count, same order, same bytes.
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
