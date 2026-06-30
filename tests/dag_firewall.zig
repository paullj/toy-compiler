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
//! Black box through `@import("toy_compiler")`; runs in `toy-integration-test`.

const std = @import("std");
const Io = std.Io;
const testing = std.testing;

const toyc = @import("toy_compiler");
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const Dag = toyc.QueryDag;
const Cache = toyc.Cache;
const Driver = toyc.Driver;

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

    /// The `signature(fid)` DAG node. M16 AUDIT FIX: the stable signature id is now
    /// the program-wide global fn id (NOT the bare name — two same-named pub fns in
    /// different modules collided under the old name-only scheme). Mirrors
    /// `Typecheck.sigNodeId`.
    fn sigNode(_: *const Recorded, fid: u32) Dag.NodeKey {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, fid, .little);
        return .{ .kind = .signature, .id = std.hash.Wyhash.hash(0x53_47_4e_4d, &b) };
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

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, &dag, io, 8);
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

test "BODY/SIG LAYOUT: a pass-through fn reaches its OWN param/ret aggregate layout via its signature (struct, called)" {
    // M16 under-recording: a pass-through `fn id(p:Point)->Point { p }` whose body
    // never constructs/accesses the aggregate had a body(id) fp blind to Point's
    // layout. Under red-green keyed on body(fid), an ABI-boundary layout edit would
    // early-cutoff id as unchanged = miscompile. The fix records body(fid)->
    // signature(fid) in checkFn, and recordSignature records signature(fid)->layout
    // for each aggregate param + the return, so id's body reaches Point's layout
    // TRANSITIVELY through its own signature node.
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
    // id's BODY (which never constructs Point) must reach Point's layout via its own
    // signature: body(id)->signature(id)->layout(Point).
    try testing.expect(r0.dag.hasEdge(r0.bodyNode(id0), r0.sigNode(id0)));
    try testing.expect(r0.dag.hasEdge(r0.sigNode(id0), lay));
    // And that recorded layout fp flips when the aggregate layout is edited, even
    // though id's body source is unchanged.
    const fp0 = r0.dag.fingerprintOf(lay) orelse return error.NoLayoutFp;
    const fp1 = r1.dag.fingerprintOf(r1.structLayoutNode(0)) orelse return error.NoLayoutFp;
    try testing.expect(fp0 != fp1);
}

test "BODY LAYOUT (uncalled): an UNCALLED pass-through fn reaches its param/ret layout via its OWN signature" {
    // An UNCALLED pass-through fn `fn id(p: Point) -> Point { p }` whose body never
    // constructs/accesses the aggregate must still tie its body to the aggregate's
    // layout, or an ABI-boundary layout edit would false-green the body (= miscompile,
    // no content-fp net). checkFn now records body(fid)->signature(fid) for EVERY fn
    // (not just pub/called ones), and recordSignature records signature->layout for
    // aggregate params/ret. So the body reaches the layout TRANSITIVELY through its own
    // signature node — the closure the firewall could not cover directly.
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
    try testing.expect(r0.dag.hasEdge(r0.bodyNode(id0), r0.sigNode(id0)));
    try testing.expect(r0.dag.hasEdge(r0.sigNode(id0), lay));
    const fp0 = r0.dag.fingerprintOf(lay) orelse return error.NoLayoutFp;
    const fp1 = r1.dag.fingerprintOf(r1.structLayoutNode(0)) orelse return error.NoLayoutFp;
    try testing.expect(fp0 != fp1);
}

test "BODY LAYOUT (enum): a pass-through fn over an enum reaches its param/ret layout via its OWN signature" {
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
    try testing.expect(r0.dag.hasEdge(r0.bodyNode(pid), r0.sigNode(pid)));
    try testing.expect(r0.dag.hasEdge(r0.sigNode(pid), lay));
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

test "RED-GREEN: a body-only edit cuts off the caller (strict-subset recompute on a REAL program)" {
    // The end-to-end cutoff gate on a recorded whole-graph typecheck: build a PRIOR
    // DAG, edit ONLY `add`'s body, build a FRESH DAG, then run `Dag.RedGreen` over
    // the fresh graph against the serialized prior. The firewall (caller -> callee
    // SIGNATURE, never body) must make `main`/`mul` cut off while `add` re-executes —
    // a STRICT SUBSET recompute, with green > 0 (liveness).
    const gpa = testing.allocator;
    const base = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn add(a: int, b: int) -> int { return a + b }
    \\fn mul(a: int, b: int) -> int { return a * b }
    \\fn main() -> int { return add(mul(2, 3), 4) }
    \\
    }};
    // SAME file/fn names/signatures => identical node ids; only add's BODY differs.
    const body_edit = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn add(a: int, b: int) -> int { return a + b + 0 }
    \\fn mul(a: int, b: int) -> int { return a * b }
    \\fn main() -> int { return add(mul(2, 3), 4) }
    \\
    }};

    var r0 = try record(".toyc-it-rg-base", base, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-rg-edit", body_edit, "m.toy");
    defer r1.deinit();
    try testing.expectEqual(@as(usize, 0), r0.tc.diags.len);
    try testing.expectEqual(@as(usize, 0), r1.tc.diags.len);

    // The PRIOR snapshot (serialize r0, deserialize a `Loaded`).
    const blob = try r0.dag.serialize(gpa);
    defer gpa.free(blob);
    var prior = (try Dag.deserialize(gpa, blob)).?;
    defer prior.deinit(gpa);

    // Walk the FRESH (edited) graph against the prior. Demand every recorded node.
    const roots = try r1.dag.nodeKeys(gpa);
    defer gpa.free(roots);
    var rg = try Dag.RedGreen.init(gpa, &prior, &r1.dag);
    defer rg.deinit();
    try rg.run(roots);

    // CUTOFF: a strict subset re-executed, and the walk verified some nodes green.
    try testing.expect(rg.red_count > 0);
    try testing.expect(rg.green_count > 0);
    try testing.expect(rg.red_count < rg.red_count + rg.green_count);

    // THE FIREWALL, the index-INDEPENDENT guarantee: add's BODY is RED (its content
    // changed) while add's SIGNATURE is GREEN (a body edit never touches the proto).
    // This is what makes a caller — whose codegen fp folds the callee SIGNATURE, not
    // its body ([C3], index-free) — cut off. NB the callers' typecheck `body` nodes
    // here ALSO depend on index-derived `type_of` nodes whose ids SHIFT when the edit
    // inserts Ast nodes, so they are RED in this TYPECHECK-only DAG; the real
    // body-level caller cutoff lives at the CODEGEN layer (index-free fingerprints),
    // proven by `--codegen-stats` (incremental compiled=1, full compiled=3) + the
    // byte-identical soundness cmp in the stage report.
    const add_id = r1.fnId(".add");
    try testing.expectEqual(Dag.Verdict.red, try rg.verdictOf(r1.bodyNode(add_id)));
    try testing.expectEqual(Dag.Verdict.green, try rg.verdictOf(r1.sigNode(add_id)));
}

test "RED-GREEN: an unedited rebuild of a real program is ALL GREEN (no recompute)" {
    const gpa = testing.allocator;
    const files = &[_]FixtureFile{.{ .path = "m.toy", .source =
    \\fn add(a: int, b: int) -> int { return a + b }
    \\fn main() -> int { return add(1, 2) }
    \\
    }};
    var r0 = try record(".toyc-it-rg-same0", files, "m.toy");
    defer r0.deinit();
    var r1 = try record(".toyc-it-rg-same1", files, "m.toy");
    defer r1.deinit();

    const blob = try r0.dag.serialize(gpa);
    defer gpa.free(blob);
    var prior = (try Dag.deserialize(gpa, blob)).?;
    defer prior.deinit(gpa);

    const roots = try r1.dag.nodeKeys(gpa);
    defer gpa.free(roots);
    var rg = try Dag.RedGreen.init(gpa, &prior, &r1.dag);
    defer rg.deinit();
    try rg.run(roots);

    // Nothing changed => zero recompute, all green.
    try testing.expectEqual(@as(usize, 0), rg.red_count);
    try testing.expect(rg.green_count > 0);
}

/// Discover `entry` under a FRESH cache dir (a cold build) with a `*Dag` threaded,
/// then return the recorded lex+parse node fingerprints keyed by NodeKey. Each call
/// uses an independent dir + cache + heap so two calls model two separate compiler
/// processes — the exact thing persisted red-green must round-trip.
fn coldFrontEndFps(comptime dir_name: []const u8, source: []const u8) !std.AutoHashMapUnmanaged(Dag.NodeKey, u64) {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const full = try std.fmt.bufPrint(&path_buf, "{s}/m.toy", .{dir_name});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = full, .data = source });

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir_name});
    const cache = try Cache.init(io, cache_dir);

    var dag: Dag = .init(gpa);
    defer dag.deinit(gpa);

    var graph = try Graph.discoverDag(gpa, io, cache, "native", full, &dag);
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);

    var out: std.AutoHashMapUnmanaged(Dag.NodeKey, u64) = .{};
    var it = dag.result_fp.iterator();
    while (it.next()) |e| {
        if (e.key_ptr.kind == .lex or e.key_ptr.kind == .parse)
            try out.put(gpa, e.key_ptr.*, e.value_ptr.*);
    }
    return out;
}

test "COLD-BUILD FP DETERMINISM: lex+parse result fps are byte-identical across two cold builds" {
    // The Stage-0 lead task: two fresh-process cold builds of the same source must
    // record IDENTICAL lex/parse fps. Before the fix, `Node`/`Token` extern-struct
    // padding leaked into the hashed bytes, flipping the parse fp run-to-run and
    // making persisted red-green inert (permanent miss) or unsound (false-green).
    const source =
        \\struct Point { x: int, y: int }
        \\enum Shape { circle(int), square }
        \\fn area(s: Shape) -> int { return match s { .circle(r) -> r, .square -> 1 } }
        \\fn main() -> int {
        \\  p := Point{ x: 1, y: 2 }
        \\  return area(Shape.circle(p.x))
        \\}
        \\
    ;
    var a = try coldFrontEndFps(".toyc-it-coldfp-a", source);
    defer a.deinit(testing.allocator);
    var b = try coldFrontEndFps(".toyc-it-coldfp-b", source);
    defer b.deinit(testing.allocator);

    try testing.expect(a.count() > 0);
    try testing.expectEqual(a.count(), b.count());
    var it = a.iterator();
    while (it.next()) |e| {
        const bv = b.get(e.key_ptr.*) orelse return error.MissingNodeInColdBuildB;
        try testing.expectEqual(e.value_ptr.*, bv);
    }
}

test "M17 PERSISTENCE: a build WRITES the DAG artifact; a fresh process LOADS it and the fps match" {
    // The Stage-1 round-trip gate: build 1 records a real front-end DAG, serializes
    // it, and `putDag`s it to disk. Build 2 — an independent heap + a freshly opened
    // Cache handle, modelling a separate compiler invocation — `getDag`s the blob,
    // `deserialize`s it, and every persisted lex/parse fp must equal what a fresh
    // recording of the SAME source produces (the prerequisite for red-green: ids AND
    // fps round-trip across processes).
    const gpa = testing.allocator;
    const source =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn main() -> int { return add(1, 2) }
        \\
    ;
    const dir = ".toyc-it-persist";

    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&path_buf, "{s}/m.toy", .{dir});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = entry, .data = source });

    var cache_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&cache_buf, "{s}/.cache", .{dir});
    const key = Cache.programKey(entry, "native");

    // --- build 1: record, serialize, WRITE the artifact ---
    {
        const cache = try Cache.init(io, cache_dir);
        var dag: Dag = .init(gpa);
        defer dag.deinit(gpa);
        var graph = try Graph.discoverDag(gpa, io, cache, "native", entry, &dag);
        defer graph.deinit(gpa);
        try testing.expect(graph.err == null);

        // No prior artifact on a cold build.
        try testing.expect((try cache.getDag(gpa, io, key)) == null);

        const blob = try dag.serialize(gpa);
        defer gpa.free(blob);
        try cache.putDag(io, key, blob);
    }

    // --- build 2: a fresh Cache handle + heap (a separate "process") LOADS it ---
    {
        const cache = try Cache.init(io, cache_dir);
        const blob = (try cache.getDag(gpa, io, key)) orelse return error.ArtifactNotLoaded;
        defer gpa.free(blob);
        var loaded = (try Dag.deserialize(gpa, blob)) orelse return error.ArtifactUnparseable;
        defer loaded.deinit(gpa);

        try testing.expect(loaded.nodes.len > 0);

        // Re-record the same source fresh and assert every persisted lex/parse fp
        // round-trips identically (the cold-stable-fp prerequisite, now persisted).
        var dag2: Dag = .init(gpa);
        defer dag2.deinit(gpa);
        var graph2 = try Graph.discoverDag(gpa, io, cache, "native", entry, &dag2);
        defer graph2.deinit(gpa);
        try testing.expect(graph2.err == null);

        var checked: usize = 0;
        var it = dag2.result_fp.iterator();
        while (it.next()) |e| {
            if (e.key_ptr.kind != .lex and e.key_ptr.kind != .parse) continue;
            const persisted = loaded.nodeFp(e.key_ptr.*) orelse return error.PersistedNodeMissing;
            try testing.expectEqual(e.value_ptr.*, persisted);
            checked += 1;
        }
        try testing.expect(checked > 0);
    }
}

/// Run the FULL `-o` build pipeline (discover -> resolve -> typecheck -> codegen)
/// with a `*Dag` threaded through every query — exactly what `--query-stats` does
/// — over `source` written to `entry` under a SHARED cache dir. Returns the
/// populated dag (caller deinits) plus the lowering's codegen compiled/cached
/// counters (the byte-level cutoff). Models one compiler invocation; calling it
/// twice against the SAME cache_dir models two `-o --query-stats` runs.
const WireBuild = struct {
    dag: Dag,
    compiled: usize,
    cached: usize,
    gpa: std.mem.Allocator,
    fn deinit(self: *WireBuild) void {
        self.dag.deinit(self.gpa);
    }
};

fn wireBuild(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    entry: []const u8,
    source: []const u8,
) !WireBuild {
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = entry, .data = source });

    var dag: Dag = .init(gpa);
    errdefer dag.deinit(gpa);

    var graph = try Graph.discoverDag(gpa, io, cache, "aarch64-macos", entry, &dag);
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, &dag, io, 8);
    defer tc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    var lowered = try Driver.lowerGraphProgram(gpa, io, cache, "aarch64-macos", &graph, &res, &tc, .normal, .O0, &dag, null, null, 8);
    switch (lowered) {
        .err => return error.TestUnexpectedResult,
        .ok => |*lp| {
            defer lp.deinit(gpa);
            try testing.expectEqual(@as(usize, 0), lp.diags.len);
            return .{ .dag = dag, .compiled = lp.codegen_compiled, .cached = lp.codegen_cached, .gpa = gpa };
        },
    }
}

test "M17 WIRE-AND-PROOF: a body edit through the real -o pipeline cuts off the caller (codegen node GREEN), persisted DAG round-trips" {
    // The end-to-end production proof of `--query-stats`: drive the WHOLE `-o`
    // pipeline (incl. codegen) with the dag threaded, persist the DAG through the
    // REAL Cache.programKey/putDag, then a fresh build of a body-edited source
    // loads it and the red-green walk over the CODEGEN nodes must:
    //   * mark the edited fn's codegen node RED (its content fingerprint flipped =>
    //     its node id is new w.r.t. the prior DAG), and
    //   * keep the caller's codegen node GREEN (its fingerprint folds the callee
    //     SIGNATURE, not its body — the firewall — so its node id is UNCHANGED and
    //     present in the prior DAG with the same fp).
    // The byte-level cutoff is corroborated by `codegen_compiled` from the shared
    // content cache (the edited build re-lowers exactly 1 fn).
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toyc-it-wire";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    var entry_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&entry_buf, "{s}/m.toy", .{dir});
    var cache_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&cache_buf, "{s}/.cache", .{dir});
    const cache = try Cache.init(io, cache_dir);
    const prog_key = Cache.programKey(entry, "aarch64-macos");

    const base =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn mul(a: int, b: int) -> int { return a * b }
        \\fn main() -> int { return add(mul(2, 3), 4) }
        \\
    ;
    // SAME signatures; ONLY mul's body changes.
    const edited =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn mul(a: int, b: int) -> int { return a * b + 0 }
        \\fn main() -> int { return add(mul(2, 3), 4) }
        \\
    ;

    // --- build 1: cold, record + persist the DAG ---
    var b0 = try wireBuild(gpa, io, cache, entry, base);
    defer b0.deinit();
    try testing.expectEqual(@as(usize, 3), b0.compiled); // cold: all 3 fns lowered
    try testing.expectEqual(@as(usize, 0), b0.cached);

    const blob0 = try b0.dag.serialize(gpa);
    defer gpa.free(blob0);
    try cache.putDag(io, prog_key, blob0);

    // --- build 2: a fresh process loads the prior DAG, body-edits mul ---
    const prior_blob = (try cache.getDag(gpa, io, prog_key)) orelse return error.ArtifactNotLoaded;
    defer gpa.free(prior_blob);
    var prior = (try Dag.deserialize(gpa, prior_blob)) orelse return error.ArtifactUnparseable;
    defer prior.deinit(gpa);

    var b1 = try wireBuild(gpa, io, cache, entry, edited);
    defer b1.deinit();
    // THE BYTE-LEVEL CUTOFF: exactly ONE fn re-lowered (mul); add + main are cache
    // hits (their content fingerprints are unchanged — the firewall).
    try testing.expectEqual(@as(usize, 1), b1.compiled);
    try testing.expectEqual(@as(usize, 2), b1.cached);

    // The red-green walk over the fresh DAG vs the prior: a STRICT SUBSET re-executes
    // and some nodes verify green (liveness).
    const roots = try b1.dag.nodeKeys(gpa);
    defer gpa.free(roots);
    var rg = try Dag.RedGreen.init(gpa, &prior, &b1.dag);
    defer rg.deinit();
    try rg.run(roots);
    try testing.expect(rg.red_count > 0);
    try testing.expect(rg.green_count > 0);
    try testing.expect(rg.red_count < rg.red_count + rg.green_count);

    // THE CODEGEN FIREWALL, the index-free guarantee that drives the byte cutoff:
    // the caller (main) and the untouched sibling (add) carry UNCHANGED codegen node
    // ids (their content fingerprints didn't move), so they are present in the prior
    // DAG with the same fp -> GREEN. The edited fn (mul) has a NEW codegen id -> RED.
    var green_codegen: usize = 0;
    var red_codegen: usize = 0;
    for (roots) |k| {
        if (k.kind != .codegen) continue;
        switch (try rg.verdictOf(k)) {
            .green => green_codegen += 1,
            .red => red_codegen += 1,
        }
    }
    // At least the two unchanged fns' codegen nodes cut off (GREEN); the edited fn's
    // codegen node is a fresh id (RED — counted among the recompute set).
    try testing.expect(green_codegen >= 2);
    try testing.expect(red_codegen >= 1);
}

// ===========================================================================
// CONTENT-FP CUTOFF: the content-fp cache is the SOLE reuse driver. Each test edits
// a source, rebuilds over a warm cache so unchanged fns are served from the cache,
// and asserts the emitted __text is BYTE-IDENTICAL to a cold full rebuild of the
// same edited source. Byte-identity is the non-negotiable correctness oracle: a
// stale-blob reuse would change a byte. (The DAG records the same nodes for the
// red-green REPORTER; it never gates these builds.)
// ===========================================================================

/// One full `-o`-shaped build returning the emitted __text (caller frees) + the
/// codegen compiled/cached counters + the fresh serialized DAG blob (caller frees).
/// Reuse is the content-fp cache (the sole driver); the DAG is recorded for
/// observability and serialized so a caller can run the red-green REPORTER over it.
const GateBuild = struct {
    text: []u8,
    dag_blob: []u8,
    compiled: usize,
    cached: usize,
    gpa: std.mem.Allocator,
    fn deinit(self: *GateBuild) void {
        self.gpa.free(self.text);
        self.gpa.free(self.dag_blob);
    }
};

fn gateBuild(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    entry: []const u8,
    files: []const FixtureFile,
) !GateBuild {
    for (files) |f| try Io.Dir.cwd().writeFile(io, .{ .sub_path = f.path, .data = f.source });

    var dag: Dag = .init(gpa);
    defer dag.deinit(gpa);

    var graph = try Graph.discoverDag(gpa, io, cache, "aarch64-macos", entry, &dag);
    defer graph.deinit(gpa);
    try testing.expect(graph.err == null);

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, &dag, io, 8);
    defer tc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    var lowered = try Driver.lowerGraphProgram(gpa, io, cache, "aarch64-macos", &graph, &res, &tc, .normal, .O0, &dag, null, null, 8);
    switch (lowered) {
        .err => return error.TestUnexpectedResult,
        .ok => |*lp| {
            defer lp.deinit(gpa);
            try testing.expectEqual(@as(usize, 0), lp.diags.len);
            const text = try gpa.dupe(u8, lp.text);
            errdefer gpa.free(text);
            const blob = try dag.serialize(gpa);
            return .{ .text = text, .dag_blob = blob, .compiled = lp.codegen_compiled, .cached = lp.codegen_cached, .gpa = gpa };
        },
    }
}

/// Cold full rebuild of `files` under a FRESH cache (no warm cache) — the
/// ground-truth __text every incremental build must match byte-for-byte. Uses a
/// distinct cache dir so no cached blob can leak in.
fn coldText(gpa: std.mem.Allocator, io: Io, dir: []const u8, entry: []const u8, files: []const FixtureFile) ![]u8 {
    var cache_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&cache_buf, "{s}/.cold-cache", .{dir});
    Io.Dir.cwd().deleteTree(io, cache_dir) catch {};
    const cache = try Cache.init(io, cache_dir);
    var b = try gateBuild(gpa, io, cache, entry, files);
    defer b.deinit();
    return gpa.dupe(u8, b.text);
}

/// Two-build driver over ONE warm cache: cold build of `base`, then an edited
/// rebuild of `edited` that reuses the unchanged fns from the content-fp cache.
/// Asserts the edited build's __text == a cold full rebuild of `edited` (the
/// byte-identity oracle for content-fp reuse). Returns the warm build's (compiled,
/// cached) so a caller can assert the cutoff shape.
const TwoBuild = struct { compiled: usize, cached: usize };

fn twoBuild(
    gpa: std.mem.Allocator,
    io: Io,
    dir: []const u8,
    entry: []const u8,
    base: []const FixtureFile,
    edited: []const FixtureFile,
) !TwoBuild {
    var cache_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&cache_buf, "{s}/.cache", .{dir});
    const cache = try Cache.init(io, cache_dir);

    // build 1: cold, warms the content-fp cache.
    var b0 = try gateBuild(gpa, io, cache, entry, base);
    defer b0.deinit();

    // build 2: edited source over the warm cache => unchanged fns are content-fp hits.
    var b1 = try gateBuild(gpa, io, cache, entry, edited);
    defer b1.deinit();

    // THE ORACLE: byte-identical to a cold full rebuild of the edited source.
    const cold = try coldText(gpa, io, dir, entry, edited);
    defer gpa.free(cold);
    try testing.expectEqualSlices(u8, cold, b1.text);

    return .{ .compiled = b1.compiled, .cached = b1.cached };
}

fn gateDir(io: Io, name: []const u8) !void {
    Io.Dir.cwd().deleteTree(io, name) catch {};
    try Io.Dir.cwd().createDirPath(io, name);
}

test "CONTENT-FP CUTOFF (a): a body-only edit reuses the caller from cache and stays byte-identical to a cold rebuild" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-body";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn mul(a: int, b: int) -> int { return a * b }
        \\fn main() -> int { return add(mul(2, 3), 4) }
        \\
    }};
    // ONLY mul's body changes; add + main signatures + bodies are unchanged.
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn mul(a: int, b: int) -> int { return a * b + 0 - 0 }
        \\fn main() -> int { return add(mul(2, 3), 4) }
        \\
    }};

    const r = try twoBuild(gpa, io, dir, entry, &base, &edited);
    // The content-fp cutoff fired: add + main are served from cache (their transitive
    // fingerprints are unchanged), only the edited mul re-lowered. The byte-identity
    // check inside twoBuild proves the cached bytes are correct.
    try testing.expectEqual(@as(usize, 1), r.compiled);
    try testing.expectEqual(@as(usize, 2), r.cached);
}

test "CONTENT-FP CUTOFF (b): a signature edit turns the caller RED but output is byte-identical to a cold rebuild" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-sig";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\fn id(a: int) -> int { return a }
        \\fn main() -> int { return id(7) }
        \\
    }};
    // `id`'s SIGNATURE changes (a second param): the caller `main`'s transitive
    // fingerprint flips (it folds the callee signature) so it re-lowers. Byte-identity
    // vs cold proves the content-fp cache did not falsely reuse a stale caller.
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\fn id(a: int, b: int) -> int { return a + b }
        \\fn main() -> int { return id(7, 0) }
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (c): an ABI layout edit to an aggregate used only in a callee body keeps the caller correct (closure guard)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-layout";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    // `area` constructs+reads a Rect ONLY in its own body; the layout flows to area
    // via codegen->body->layout (the CLOSURE edge under test). A field-order edit
    // changes the ABI; area MUST re-lower. main calls area but never names Rect.
    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\struct Rect { w: int, h: int }
        \\fn area() -> int {
        \\ r := Rect { w: 3, h: 4 }
        \\ return r.w * r.h
        \\}
        \\fn main() -> int { return area() }
        \\
    }};
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\struct Rect { h: int, w: int }
        \\fn area() -> int {
        \\ r := Rect { w: 3, h: 4 }
        \\ return r.w * r.h
        \\}
        \\fn main() -> int { return area() }
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (d): two same-named fns in different modules do not falsely reuse (sigNodeId collision fix)" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-collide";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    var ab: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});
    const apath = try std.fmt.bufPrint(&ab, "{s}/a.toy", .{dir});

    // BOTH modules define `helper`. With the old name-only signature id these two
    // collided on one DAG node (and the reporter could mis-attribute their fps). The
    // fix keys signatures by global fn id, so the ids are distinct. We edit a.helper's
    // body only; byte-identity vs cold proves the content-fp cache did not falsely reuse.
    const base = [_]FixtureFile{
        .{ .path = apath, .source =
        \\pub fn helper(x: int) -> int { return x + 1 }
        \\
        },
        .{ .path = entry, .source =
        \\import a
        \\fn helper(x: int) -> int { return x * 2 }
        \\fn main() -> int { return helper(2) + a.helper(3) }
        \\
        },
    };
    const edited = [_]FixtureFile{
        .{ .path = apath, .source =
        \\pub fn helper(x: int) -> int { return x + 1 + 0 }
        \\
        },
        .{ .path = entry, .source =
        \\import a
        \\fn helper(x: int) -> int { return x * 2 }
        \\fn main() -> int { return helper(2) + a.helper(3) }
        \\
        },
    };

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (e): a fn's own PARAMETER REORDER (body brace identical) is not falsely reused" {
    // The proto-only stale-hit class: `f`'s body brace span `{ return a + b }` is
    // BYTE-IDENTICAL across the edit; only the proto order flips (a,b -> b,a). The body
    // hash folds only the brace span (excludes the proto), so if the codegen fingerprint
    // did not also fold `f`'s OWN signature a content-fp cache HIT would serve the base
    // blob (a->x0, b->x1) for a proto that now wants a->x1,b->x0 = miscompile. The
    // fingerprint folds the proto (sigFingerprint folds param order), so `f`'s content
    // fp flips and it re-lowers. main also re-lowers (callee sig changed). Oracle:
    // byte-identity vs cold rebuild.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-reorder";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\fn f(a: int, b: int) -> int { return a + b }
        \\fn main() -> int { return f(1, 2) }
        \\
    }};
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\fn f(b: int, a: int) -> int { return a + b }
        \\fn main() -> int { return f(1, 2) }
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (f): a fn's own ARITY change (body brace identical, params unused) is not falsely reused" {
    // f's body `{ return 0 }` is byte-identical and references no param, so the body
    // hash is unchanged across the edit. Only the arity flips (1 -> 2 params), which the
    // authoritative codegen fp folds (sigFingerprint folds params.len) so f's content fp
    // flips and it re-lowers; without that fold a content-fp cache HIT would serve the
    // 1-param base blob for a 2-param f = miscompile.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-arity";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\fn f(a: int) -> int { return 0 }
        \\fn main() -> int { return f(0) }
        \\
    }};
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\fn f(a: int, b: int) -> int { return 0 }
        \\fn main() -> int { return f(0, 0) }
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (g): a fn's own SCALAR param-type change (body brace identical) is not falsely reused" {
    // f is UNCALLED (no caller to flip it). Its body `{ return 0 }` is byte-identical;
    // only the param type changes int -> str. scalarSize(str)=16 vs int=8, so the
    // emitted prologue/frame differs. The codegen fp folds the param KIND
    // (sigFingerprint), so f's content fp flips and it re-lowers; without that fold a
    // content-fp cache HIT would serve the int-ABI blob = a wrong-ABI miscompile.
    // Oracle: byte-identity vs cold rebuild.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-scalar";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\fn f(p: int) -> int { return 0 }
        \\fn main() -> int { return 0 }
        \\
    }};
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\fn f(p: str) -> int { return 0 }
        \\fn main() -> int { return 0 }
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (h): a fn RENAME at the same source position (body brace identical) is not falsely reused" {
    // foo->baz keeps the same global fn id (same source slot) and the same body brace
    // `{ return a }`, so the body hash is stable. The codegen cache key folds the
    // emitted SymName (symMix) AND the fingerprint folds the fn's own name
    // (sigFingerprint), so the renamed fn's content fp flips and it re-lowers. Without
    // that a content-fp cache HIT would serve foo's old blob (FnCode.sym = "foo") for
    // baz = wrong symbol / stale reloc. main also re-lowers (its call target name
    // changed). Oracle: byte-identity vs cold rebuild.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-rename";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\fn foo(a: int) -> int { return a }
        \\fn main() -> int { return foo(5) }
        \\
    }};
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\fn baz(a: int) -> int { return a }
        \\fn main() -> int { return baz(5) }
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (i): a PASS-THROUGH aggregate param/return layout edit is not falsely reused" {
    // The subtle one: `id` takes and returns P BY VALUE but its body never field-
    // accesses P. The proto TEXT `(p: P) -> P` is byte-identical across the edit (P's
    // NAME is unchanged) so the body hash does not flip. The codegen fingerprint MUST
    // fold P's LAYOUT (its touched-type walk over the signature's aggregates), so when P
    // gains a field id's content fp flips and it re-lowers. Without that fold a content-
    // fp cache HIT would serve the 1-field-ABI blob for a 2-field P = ABI miscompile.
    // Oracle: byte-identity vs cold rebuild. (Regression: a prior refactor dropped the
    // touched-layout fold, trusting the type identity alone — which a name-stable layout
    // edit slips past.)
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-passthrough";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\struct P { x: int }
        \\fn id(p: P) -> P { return p }
        \\fn main() -> int {
        \\    q := id(P{ x: 5 })
        \\    return q.x
        \\}
        \\
    }};
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\struct P { x: int, y: int }
        \\fn id(p: P) -> P { return p }
        \\fn main() -> int {
        \\    q := id(P{ x: 5, y: 9 })
        \\    return q.x
        \\}
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}

test "CONTENT-FP CUTOFF (j): a PASS-THROUGH enum payload-layout edit is not falsely reused" {
    // Same as (i) for the enum layout class (recordLayoutOf reserves the high id bit
    // for enums). The edit GROWS the R variant's payload (1 int -> 2 ints), which
    // enlarges E's by-value size = a real ABI change for the pass-through `route`
    // (whose body never matches on E). A bare variant-COUNT bump within one tag size
    // would NOT change `route`'s emitted bytes and so could not discriminate — the
    // payload growth is what makes the test meaningful. The codegen fingerprint folds
    // E's layout (the touched-type walk); without that fold a content-fp cache HIT
    // would falsely reuse route's old blob = ABI miscompile.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const dir = ".toyc-gate-passthrough-enum";
    try gateDir(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};
    var eb: [std.fs.max_path_bytes]u8 = undefined;
    const entry = try std.fmt.bufPrint(&eb, "{s}/m.toy", .{dir});

    const base = [_]FixtureFile{.{ .path = entry, .source =
        \\enum E { R { a: int }, N }
        \\fn route(e: E) -> E { return e }
        \\fn main() -> int {
        \\    x := route(E.R { a: 5 })
        \\    return match x { .R { a } -> a, .N -> 0 }
        \\}
        \\
    }};
    const edited = [_]FixtureFile{.{ .path = entry, .source =
        \\enum E { R { a: int, b: int }, N }
        \\fn route(e: E) -> E { return e }
        \\fn main() -> int {
        \\    x := route(E.R { a: 5, b: 9 })
        \\    return match x { .R { a, b } -> a + b, .N -> 0 }
        \\}
        \\
    }};

    _ = try twoBuild(gpa, io, dir, entry, &base, &edited);
}
