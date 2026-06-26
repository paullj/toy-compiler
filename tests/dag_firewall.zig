//! M16 INTEGRATION TESTS — the recorded query DAG + the signature/body firewall,
//! exercised on a REAL whole-graph typecheck (not the pure Dag-mechanics unit tests
//! inside Dag.zig). These drive `TypecheckGraph.checkGraph(..., &dag)` over a
//! discovered module graph and assert the recorded structure:
//!
//!   * the body(fn)->signature(callee) edge EXISTS (the caller's body depends on the
//!     callee's signature);
//!   * a body-only edit to the callee leaves signature(callee)'s recorded fp STABLE
//!     while flipping body(callee)'s fp — THE FIREWALL, proven on a real program;
//!   * a signature edit (a param type change) FLIPS signature(callee)'s fp;
//!   * the layout(type) fp mirrors the codegen layout bytes (a field edit flips it);
//!   * the typecheck signature node id RECONCILES with codegen's (`Wyhash("SGNM",
//!     name)`) so a body->signature and a codegen->signature edge share one node.
//!
//! Black box through `@import("toy_compiler")`; runs in `toyc-integration-test`.

const std = @import("std");
const Io = std.Io;
const testing = std.testing;

const toyc = @import("toy_compiler");
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Dag = toyc.QueryDag;
const Cache = toyc.Cache;

const FixtureFile = struct { path: []const u8, source: []const u8 };

/// Result of a recorded whole-graph typecheck: the populated `Dag` + the discovered
/// graph (so a test can map a fn name to its global id). Caller deinits via `deinit`.
const Recorded = struct {
    dag: Dag,
    graph: Graph.Graph,
    res: ResolveGraph.GraphResult,
    tc: TypecheckGraph.GraphResult,
    threaded: *std.Io.Threaded,
    dir_name: []const u8,
    io: Io,
    gpa: std.mem.Allocator,

    fn deinit(self: *Recorded) void {
        self.tc.deinit(self.gpa);
        self.res.deinit(self.gpa);
        self.graph.deinit(self.gpa);
        self.dag.deinit(self.gpa);
        Io.Dir.cwd().deleteTree(self.io, self.dir_name) catch {};
        self.threaded.deinit();
        self.gpa.destroy(self.threaded);
    }

    /// The global fn id whose qualified name ENDS WITH `suffix` (e.g. ".add" or
    /// bare "main"); the resolver's `fns` order == the typecheck fn id order.
    fn fnId(self: *const Recorded, suffix: []const u8) u32 {
        for (self.res.fns, 0..) |gf, i| {
            if (std.mem.eql(u8, gf.name, suffix) or std.mem.endsWith(u8, gf.name, suffix))
                return @intCast(i);
        }
        std.debug.panic("no fn matching '{s}'", .{suffix});
    }

    /// The `signature(fid)` DAG node, folding the SAME name codegen folds.
    fn sigNode(self: *const Recorded, fid: u32) Dag.NodeKey {
        const name = self.res.fns[fid].name;
        return .{ .kind = .signature, .id = std.hash.Wyhash.hash(0x53_47_4e_4d, name) };
    }

    fn bodyNode(_: *const Recorded, fid: u32) Dag.NodeKey {
        return .{ .kind = .body, .id = fid };
    }

    /// How many DISTINCT `resolve_name` children `body(fid)` has. `resolve` is NOT
    /// engine-routed, so these edges are recorded directly from the typecheck read
    /// site (`typeOfCall`) — this counts them off the recorded `edges` map.
    fn resolveNameChildCount(self: *const Recorded, fid: u32) usize {
        const list = self.dag.edges.get(self.bodyNode(fid)) orelse return 0;
        var n: usize = 0;
        for (list.items) |child| if (child.kind == .resolve_name) {
            n += 1;
        };
        return n;
    }

    /// Whether `body(fid)` records ANY `resolve_name` edge.
    fn hasResolveNameChild(self: *const Recorded, fid: u32) bool {
        return self.resolveNameChildCount(fid) > 0;
    }

    /// How many DISTINCT `layout` children `body(fid)` records.
    fn layoutChildCount(self: *const Recorded, fid: u32) usize {
        const list = self.dag.edges.get(self.bodyNode(fid)) orelse return 0;
        var n: usize = 0;
        for (list.items) |child| if (child.kind == .layout) {
            n += 1;
        };
        return n;
    }

    /// The struct layout node id is the raw global struct id; the enum layout node id
    /// is tagged with the high bit (recordLayoutOf in src/types.zig) so struct(N) and
    /// enum(N) stay distinct nodes.
    fn structLayoutNode(_: *const Recorded, struct_id: u64) Dag.NodeKey {
        return .{ .kind = .layout, .id = struct_id };
    }
    fn enumLayoutNode(_: *const Recorded, enum_id: u64) Dag.NodeKey {
        return .{ .kind = .layout, .id = enum_id | (@as(u64, 1) << 63) };
    }
};

/// Write `files`, discover the graph from `entry`, resolve it, then typecheck it
/// with a per-build `Dag` threaded so every fine-grained projection is recorded.
fn record(comptime dir_name: []const u8, files: []const FixtureFile, entry: []const u8) !Recorded {
    const gpa = testing.allocator;
    const threaded = try gpa.create(std.Io.Threaded);
    errdefer gpa.destroy(threaded);
    threaded.* = std.Io.Threaded.init(gpa, .{});
    errdefer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    errdefer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

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

    var dag: Dag = .init(gpa);
    errdefer dag.deinit(gpa);

    var graph = try Graph.discover(gpa, io, cache, "native", entry_path);
    errdefer graph.deinit(gpa);
    try testing.expect(graph.err == null);

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    errdefer res.deinit(gpa);

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, &dag);
    errdefer tc.deinit(gpa);

    return .{
        .dag = dag,
        .graph = graph,
        .res = res,
        .tc = tc,
        .threaded = threaded,
        .dir_name = dir_name,
        .io = io,
        .gpa = gpa,
    };
}

test "body(caller) records a signature(callee) edge (the firewall edge exists)" {
    const files = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn main() -> int { return add(1, 2) }
    \\fn add(a: int, b: int) -> int { return a + b }
    \\
    }};
    var r = try record(".toyc-it-fw-edge", files, "m.toy");
    defer r.deinit();
    try testing.expectEqual(@as(usize, 0), r.tc.diags.len);

    const main_id = r.fnId("main");
    const add_id = r.fnId(".add");
    // main's body depends on add's SIGNATURE (not its body).
    try testing.expect(r.dag.hasEdge(r.bodyNode(main_id), r.sigNode(add_id)));
    // and there is NO body(main) -> body(add) edge (the firewall: never depend on a
    // callee's body).
    try testing.expect(!r.dag.hasEdge(r.bodyNode(main_id), r.bodyNode(add_id)));
}

test "FIREWALL: a body-only edit to the callee keeps signature(callee) fp STABLE, flips body(callee) fp" {
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn main() -> int { return add(1, 2) }
    \\fn add(a: int, b: int) -> int { return a + b }
    \\
    }};
    // SAME file path + SAME fn names + SAME signatures => identical signature node
    // ids; only add's BODY text differs.
    const body_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn main() -> int { return add(1, 2) }
    \\fn add(a: int, b: int) -> int { return a + b + 0 }
    \\
    }};

    var r0 = try record(".toyc-it-fw-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-fw-body", body_edit, "m.toy");
    defer r1.deinit();

    const add0 = r0.fnId(".add");
    const add1 = r1.fnId(".add");

    const sig0 = r0.dag.fingerprintOf(r0.sigNode(add0)) orelse return error.NoSigFp;
    const sig1 = r1.dag.fingerprintOf(r1.sigNode(add1)) orelse return error.NoSigFp;
    const body0 = r0.dag.fingerprintOf(r0.bodyNode(add0)) orelse return error.NoBodyFp;
    const body1 = r1.dag.fingerprintOf(r1.bodyNode(add1)) orelse return error.NoBodyFp;

    // The signature fp is STABLE across a body-only edit (the firewall).
    try testing.expectEqual(sig0, sig1);
    // The body fp FLIPS (the body content changed).
    try testing.expect(body0 != body1);
}

test "FIREWALL: a signature edit (param type change) FLIPS signature(callee) fp" {
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn main() -> int { return add(1, 2) }
    \\fn add(a: int, b: int) -> int { return a + b }
    \\
    }};
    const sig_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn main() -> int { return add(1, true) }
    \\fn add(a: int, b: bool) -> int { return a }
    \\
    }};

    var r0 = try record(".toyc-it-sig-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-sig-edit", sig_edit, "m.toy");
    defer r1.deinit();

    const add0 = r0.fnId(".add");
    const add1 = r1.fnId(".add");
    // Same name => same signature node id; the param-type change must FLIP its fp.
    try testing.expectEqual(r0.sigNode(add0).id, r1.sigNode(add1).id);
    const sig0 = r0.dag.fingerprintOf(r0.sigNode(add0)) orelse return error.NoSigFp;
    const sig1 = r1.dag.fingerprintOf(r1.sigNode(add1)) orelse return error.NoSigFp;
    try testing.expect(sig0 != sig1);
}

test "LAYOUT: a signature naming a struct records a layout edge; a field edit flips its fp" {
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct Point { x: int, y: int }
    \\fn dist(p: Point) -> int { return p.x + p.y }
    \\fn main() -> int {
    \\ q := Point { x: 6, y: 7 }
    \\ return dist(q)
    \\}
    \\
    }};
    // Add a field to Point: the layout bytes (size/offsets/field set) change.
    const field_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct Point { x: int, y: int, z: int }
    \\fn dist(p: Point) -> int { return p.x + p.y }
    \\fn main() -> int {
    \\ q := Point { x: 6, y: 7, z: 8 }
    \\ return dist(q)
    \\}
    \\
    }};

    var r0 = try record(".toyc-it-lay-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-lay-edit", field_edit, "m.toy");
    defer r1.deinit();

    try testing.expectEqual(@as(usize, 0), r0.tc.diags.len);
    // Point is global struct id 0; dist's signature names it => a layout edge.
    const lay: Dag.NodeKey = .{ .kind = .layout, .id = 0 };
    const dist0 = r0.fnId(".dist");
    try testing.expect(r0.dag.hasEdge(r0.sigNode(dist0), lay));
    // The recorded layout fp flips when a field is added (mirrors the codegen
    // layout bytes — no over/under-fold drift).
    const fp0 = r0.dag.fingerprintOf(lay) orelse return error.NoLayoutFp;
    const fp1 = r1.dag.fingerprintOf(.{ .kind = .layout, .id = 0 }) orelse return error.NoLayoutFp;
    try testing.expect(fp0 != fp1);
}

test "LAYOUT (body): a struct used ONLY in a body (named in no signature) records a body->layout edge" {
    // Regression for the M16 under-recording bug: an aggregate constructed/read
    // inside a body but named in NO signature recorded NO layout edge, so a
    // field-layout edit was invisible to the fine-grained DAG => a stale body /
    // codegen result under M17 (a miscompile). The fix records body->layout at the
    // body-level struct-init/field-access/enum-construct/match/pattern read sites.
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct P { a: int, b: int }
    \\fn main() -> int {
    \\ p := P { a: 1, b: 2 }
    \\ return p.a + p.b
    \\}
    \\
    }};
    const field_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct P { a: int, b: int, c: int }
    \\fn main() -> int {
    \\ p := P { a: 1, b: 2, c: 3 }
    \\ return p.a + p.b
    \\}
    \\
    }};

    var r0 = try record(".toyc-it-bodylay-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-bodylay-edit", field_edit, "m.toy");
    defer r1.deinit();

    try testing.expectEqual(@as(usize, 0), r0.tc.diags.len);
    // P is global struct id 0; main's BODY constructs + reads it, yet NO signature
    // names it => the body itself MUST record a body->layout(0) edge (the bug: it
    // recorded none, so a field reorder/add was invisible).
    const lay: Dag.NodeKey = .{ .kind = .layout, .id = 0 };
    const main0 = r0.fnId("main");
    try testing.expect(r0.dag.hasEdge(r0.bodyNode(main0), lay));
    // The layout fp flips on a field edit, so M17 re-runs the body via that edge.
    const fp0 = r0.dag.fingerprintOf(lay) orelse return error.NoLayoutFp;
    const fp1 = r1.dag.fingerprintOf(.{ .kind = .layout, .id = 0 }) orelse return error.NoLayoutFp;
    try testing.expect(fp0 != fp1);
}

test "cross-module body(caller) records a signature(callee) edge over a qualified name" {
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
    var r = try record(".toyc-it-fw-xmod", files, "main.toy");
    defer r.deinit();
    try testing.expectEqual(@as(usize, 0), r.tc.diags.len);

    const main_id = r.fnId("main");
    const helper_id = r.fnId(".helper");
    // main's body depends on helper's signature, recorded over the QUALIFIED name.
    try testing.expect(r.dag.hasEdge(r.bodyNode(main_id), r.sigNode(helper_id)));
}

test "RECURSIVE enum does not crash the layout fold (sentinel guard)" {
    // A directly-recursive enum is a typecheck ERROR (infinite size) — recording
    // its layout fold must NOT loop/crash; the .laying/poisoned sentinel guards it.
    const files = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\enum R { A(R), B }
    \\fn main() -> int { return 0 }
    \\
    }};
    var r = try record(".toyc-it-rec", files, "m.toy");
    defer r.deinit();
    // It diagnosed the recursion (did not hang/crash); the test reaching here is the
    // proof the layout fold terminated on the poisoned node.
    try testing.expect(r.tc.diags.len >= 1);
}

test "RESOLVE_NAME: a caller's body records a resolve_name edge for the consumed callee name" {
    // The firewall's name-resolution leg: when a body consumes a callee NAME, the
    // cross-item `t.resolutions[..]` read is routed through `resolve_name(name)` and
    // an edge body(caller)->resolve_name is recorded directly (resolve is COARSE and
    // NOT engine-routed). A NON-EMPTY resolve_name child set on the caller proves the
    // leg is LIVE, not dead.
    const files = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn main() -> int { return add(1, 2) }
    \\fn add(a: int, b: int) -> int { return a + b }
    \\
    }};
    var r = try record(".toyc-it-rn-edge", files, "m.toy");
    defer r.deinit();
    try testing.expectEqual(@as(usize, 0), r.tc.diags.len);

    const main_id = r.fnId("main");
    // main's body consumes add's name => exactly one resolve_name child.
    try testing.expect(r.hasResolveNameChild(main_id));
    try testing.expectEqual(@as(usize, 1), r.resolveNameChildCount(main_id));
}

test "RESOLVE_NAME: resolve stays COARSE — repeated calls to one callee share a single resolve_name node (deduped)" {
    // Two calls to `add` in one body must collapse to ONE resolve_name child: the
    // node id is a fold of the RESOLUTION (the bound func id), so identical bindings
    // dedup via recordEdge. This is the "resolve stays coarse" guarantee — one node
    // per consumed name, not one per syntactic call site.
    const files = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn main() -> int { return add(1, 2) + add(3, 4) }
    \\fn add(a: int, b: int) -> int { return a + b }
    \\
    }};
    var r = try record(".toyc-it-rn-coarse", files, "m.toy");
    defer r.deinit();
    try testing.expectEqual(@as(usize, 0), r.tc.diags.len);

    const main_id = r.fnId("main");
    // TWO syntactic calls to add, but ONE deduped resolve_name child.
    try testing.expectEqual(@as(usize, 1), r.resolveNameChildCount(main_id));
}

test "RESOLVE_NAME: a cross-module call records a resolve_name edge over the qualified name" {
    // The qualified-name leg: a `mod.helper()` call resolves to a global func id; the
    // body must record a resolve_name edge for it just as a bare-name call does.
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
    var r = try record(".toyc-it-rn-xmod", files, "main.toy");
    defer r.deinit();
    try testing.expectEqual(@as(usize, 0), r.tc.diags.len);

    const main_id = r.fnId("main");
    try testing.expect(r.hasResolveNameChild(main_id));
}

test "BODY/SIG LAYOUT: a pass-through fn records body->layout for its OWN param/ret aggregate (struct, called)" {
    // M16 under-recording: a pass-through `fn id(p:Point)->Point { p }` whose body
    // never constructs/accesses the aggregate had a body(id) fp blind to Point's
    // layout — recordSignature recorded the layout dep ONLY on signature(id), and
    // only at caller call-sites. Under M17 red-green keyed on body(fid), an ABI-
    // boundary layout edit would early-cutoff id as unchanged = miscompile. The fix
    // records body(fid)->layout for each param + the return in checkFn.
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct Point { x: int, y: int }
    \\fn id(p: Point) -> Point { p }
    \\fn main() -> int {
    \\ q := id(Point { x: 1, y: 2 })
    \\ return q.x
    \\}
    \\
    }};
    const field_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct Point { x: int, y: int, z: int }
    \\fn id(p: Point) -> Point { p }
    \\fn main() -> int {
    \\ q := id(Point { x: 1, y: 2, z: 3 })
    \\ return q.x
    \\}
    \\
    }};
    var r0 = try record(".toyc-it-pt-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-pt-edit", field_edit, "m.toy");
    defer r1.deinit();
    try testing.expectEqual(@as(usize, 0), r0.tc.diags.len);

    const id0 = r0.fnId(".id");
    const lay = r0.structLayoutNode(0);
    // id's BODY (which never constructs Point) must depend on Point's layout.
    try testing.expect(r0.dag.hasEdge(r0.bodyNode(id0), lay));
    // And that recorded layout fp flips when the aggregate layout is edited, even
    // though id's body source is unchanged.
    const fp0 = r0.dag.fingerprintOf(lay) orelse return error.NoLayoutFp;
    const fp1 = r1.dag.fingerprintOf(r1.structLayoutNode(0)) orelse return error.NoLayoutFp;
    try testing.expect(fp0 != fp1);
}

test "BODY LAYOUT (uncalled): an UNCALLED pass-through fn still records body->layout (no signature node exists)" {
    // When the pass-through fn is never called, recordSignature never fires, so NO
    // signature node exists — the body-level edge is the ONLY thing that ties id's
    // body to the aggregate's layout. This is the case the signature firewall cannot
    // cover at all.
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct Point { x: int, y: int }
    \\fn id(p: Point) -> Point { p }
    \\fn main() -> int { return 0 }
    \\
    }};
    const field_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct Point { x: int, y: int, z: int }
    \\fn id(p: Point) -> Point { p }
    \\fn main() -> int { return 0 }
    \\
    }};
    var r0 = try record(".toyc-it-unc-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-unc-edit", field_edit, "m.toy");
    defer r1.deinit();
    try testing.expectEqual(@as(usize, 0), r0.tc.diags.len);

    const id0 = r0.fnId(".id");
    const lay = r0.structLayoutNode(0);
    try testing.expect(r0.dag.hasEdge(r0.bodyNode(id0), lay));
    const fp0 = r0.dag.fingerprintOf(lay) orelse return error.NoLayoutFp;
    const fp1 = r1.dag.fingerprintOf(r1.structLayoutNode(0)) orelse return error.NoLayoutFp;
    try testing.expect(fp0 != fp1);
}

test "BODY LAYOUT (enum): a pass-through fn over an enum records body->layout for its param/ret" {
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\enum Opt { None, Some(int) }
    \\fn passthru(o: Opt) -> Opt { o }
    \\fn main() -> int { return 0 }
    \\
    }};
    const variant_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\enum Opt { None, Some(int), Other }
    \\fn passthru(o: Opt) -> Opt { o }
    \\fn main() -> int { return 0 }
    \\
    }};
    var r0 = try record(".toyc-it-en-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-en-edit", variant_edit, "m.toy");
    defer r1.deinit();
    try testing.expectEqual(@as(usize, 0), r0.tc.diags.len);

    const pid = r0.fnId(".passthru");
    const lay = r0.enumLayoutNode(0);
    try testing.expect(r0.dag.hasEdge(r0.bodyNode(pid), lay));
    const fp0 = r0.dag.fingerprintOf(lay) orelse return error.NoLayoutFp;
    const fp1 = r1.dag.fingerprintOf(r1.enumLayoutNode(0)) orelse return error.NoLayoutFp;
    try testing.expect(fp0 != fp1);
}

test "LAYOUT NODE IDENTITY: struct(id0) and enum(id0) read in one body are DISTINCT nodes (no masking)" {
    // The critical fingerprint-fidelity bug: struct ids and enum ids are separate
    // 0-based sequences, so without a kind tag struct(0) and enum(0) collapse to one
    // layout#0 node and recordEdge's last-writer-wins fp masks the loser's layout
    // edit (an invisible change => M17 miscompile). The fix tags the enum node id
    // with the high bit. Here S(id0) + E(id0) are both read in f's body; an enum-only
    // edit must flip the ENUM layout fp without being masked by the struct.
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct S { n: int }
    \\enum E { Ea, Eb(int) }
    \\fn f(s: S, e: E) -> int {
    \\ m := match e {
    \\  .Ea -> 0
    \\  .Eb(x) -> x
    \\ }
    \\ return s.n + m
    \\}
    \\fn main() -> int { return 0 }
    \\
    }};
    const enum_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\struct S { n: int }
    \\enum E { Ea, Eb(int), Ec }
    \\fn f(s: S, e: E) -> int {
    \\ m := match e {
    \\  .Ea -> 0
    \\  .Eb(x) -> x
    \\  .Ec -> 9
    \\ }
    \\ return s.n + m
    \\}
    \\fn main() -> int { return 0 }
    \\
    }};
    var r0 = try record(".toyc-it-coll-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-coll-edit", enum_edit, "m.toy");
    defer r1.deinit();
    try testing.expectEqual(@as(usize, 0), r0.tc.diags.len);

    const struct_lay = r0.structLayoutNode(0);
    const enum_lay = r0.enumLayoutNode(0);
    // The two aggregates occupy DISTINCT DAG nodes.
    try testing.expect(struct_lay.id != enum_lay.id);
    const s_fp = r0.dag.fingerprintOf(struct_lay) orelse return error.NoLayoutFp;
    const e_fp = r0.dag.fingerprintOf(enum_lay) orelse return error.NoLayoutFp;
    try testing.expect(s_fp != e_fp);
    // An enum-only edit flips the ENUM layout fp; the struct fp is NOT masking it.
    const e_fp1 = r1.dag.fingerprintOf(r1.enumLayoutNode(0)) orelse return error.NoLayoutFp;
    try testing.expect(e_fp != e_fp1);
    // The struct's own fp is unchanged by the enum edit.
    const s_fp1 = r1.dag.fingerprintOf(r1.structLayoutNode(0)) orelse return error.NoLayoutFp;
    try testing.expectEqual(s_fp, s_fp1);
}

test "TYPE_OF IDENTITY: two same-shape modules with distinct boundary types use DISTINCT type_of nodes" {
    // type_of node ids were the raw module-local Ast.Index, so two modules sharing an
    // index collapsed distinct boundary types to one node (last-writer-wins fp masks a
    // real type change => M17 miscompile). The fix folds the module id into the node
    // id. Two modules with identical `fn f; fn g { return f() }` shape but distinct
    // return types (int vs bool) must land their call-result type_of on DISTINCT nodes.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a
        \\import b
        \\fn main() -> int { return a.g() + (if b.g() { 1 } else { 0 }) }
        \\
        },
        .{ .path = "a.toy", .source =
        \\fn f() -> int { return 1 }
        \\pub fn g() -> int { return f() }
        \\
        },
        .{ .path = "b.toy", .source =
        \\fn f() -> bool { return true }
        \\pub fn g() -> bool { return f() }
        \\
        },
    };
    var r = try record(".toyc-it-tof-xmod", files, "main.toy");
    defer r.deinit();
    try testing.expectEqual(@as(usize, 0), r.tc.diags.len);

    // a.g and b.g are different modules; their bodies each record a type_of for the
    // f() call result (int vs bool). Collect the type_of children of each body and
    // assert NO node id is shared between the two modules.
    const ag = r.fnId("a.g");
    const bg = r.fnId("b.g");
    const a_list = r.dag.edges.get(r.bodyNode(ag)) orelse return error.NoEdges;
    const b_list = r.dag.edges.get(r.bodyNode(bg)) orelse return error.NoEdges;
    for (a_list.items) |ac| {
        if (ac.kind != .type_of) continue;
        for (b_list.items) |bc| {
            if (bc.kind != .type_of) continue;
            try testing.expect(ac.id != bc.id);
        }
    }
}
