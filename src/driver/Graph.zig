//! M14 module-graph discovery.
//!
//! Given an ENTRY file, discover the whole module graph by following `import`
//! declarations transitively. A module IS a file, named by a `/`-path; an import
//! `a/b` resolves to `<root>/a/b.toy` where ROOT is the entry file's directory.
//!
//! Discovery is two-phase and SERIAL at the discover level (a stable, sorted DFS
//! over each module's imports) so the parse fan-out is never re-entered on a
//! cycle and the module ordering is deterministic ([C11]). Each file's
//! lex→parse runs through the existing on-disk content cache, so a warm rebuild
//! pays nothing.
//!
//! This stage produces the GRAPH (parsed modules + edges) and rejects the three
//! structural errors: a missing module, a path escaping the root, and an import
//! cycle (the graph must be a DAG). Cross-module name/type resolution, the global
//! id-space merge, and whole-graph lowering are LATER stages — this one stops at
//! "the graph is discovered and is a well-formed DAG".

const std = @import("std");
const Io = std.Io;
const Token = @import("../ast/Token.zig").Token;
const Ast = @import("../ast/Ast.zig");
const Cache = @import("../query/Cache.zig");
const Engine = @import("../query/Engine.zig");
const Dag = @import("../query/Dag.zig");

/// The `.toy` source extension that an import path maps onto.
pub const ext = ".toy";

/// One discovered module: its canonical import-path name, on-disk file path, and
/// the parsed tree (owned). The entry module's `path` is the bare stem of the
/// entry file (e.g. `main` for `src/main.toy`); imported modules carry their
/// `/`-joined import path (e.g. `geometry/rect`).
pub const Module = struct {
    /// Canonical module name = `/`-joined import path (entry = its file stem).
    /// This is the qualification prefix later stages prepend to symbol names.
    /// Owned.
    path: []const u8,
    /// On-disk file path actually read (root-relative or as given for entry).
    /// Owned.
    file: []const u8,
    /// Source bytes (owned).
    source: []u8 = &.{},
    /// Token stream (owned).
    tokens: []Token = &.{},
    /// Parsed tree node array (owned).
    nodes: []Ast.Node = &.{},
    /// Parse `extra` side array (owned).
    extra: []u32 = &.{},
    /// `pub` export bitset (owned; empty == no exports).
    pub_bits: []const u32 = &.{},
    /// Indices (into `Graph.modules`) of the modules this one imports, in stable
    /// (sorted-by-import-path) order. Owned.
    imports: []u32 = &.{},

    pub fn tree(m: *const Module) Ast.Tree {
        return .{ .nodes = m.nodes, .extra = m.extra, .pub_bits = m.pub_bits };
    }

    fn deinit(m: *Module, gpa: std.mem.Allocator) void {
        gpa.free(m.path);
        gpa.free(m.file);
        gpa.free(m.source);
        gpa.free(m.tokens);
        gpa.free(m.nodes);
        gpa.free(m.extra);
        if (m.pub_bits.len != 0) gpa.free(@constCast(m.pub_bits));
        gpa.free(m.imports);
        m.* = undefined;
    }
};

/// What went wrong during discovery. Carries the OWNING module's index (into
/// `modules`, or `entry_index` for the entry) so the driver can render the error
/// against that module's source at `byte_offset` (the offending import token).
/// `cycle` is a `/`-joined arrow path naming the import cycle; owned when present.
pub const Error = struct {
    pub const Kind = enum {
        /// `import a/b` named a file that does not exist under the root.
        missing,
        /// The import path escaped the root (`..`, absolute, or otherwise).
        escape,
        /// Following imports formed a cycle (the graph must be a DAG).
        cycle,
        /// A parse error in a discovered module.
        parse,
    };
    kind: Kind,
    message: []const u8,
    /// Index of the module the error is reported against (its source renders the
    /// location). `null` for the entry-not-found case (no module loaded yet).
    module: ?u32 = null,
    /// Byte offset into that module's source for `line:col`, or `null`.
    byte_offset: ?u32 = null,
    /// For `cycle`/`missing`/`escape`: a heap string (the cycle path or the
    /// resolved file path) the caller may want in the message. Owned when set.
    detail: []const u8 = &.{},

    fn deinit(e: *Error, gpa: std.mem.Allocator) void {
        gpa.free(e.message);
        if (e.detail.len != 0) gpa.free(e.detail);
    }
};

/// The discovered graph: every module reachable from the entry, in deterministic
/// discovery order with the ENTRY at `entry_index`. On a structural failure
/// `err` is set and `modules` holds whatever was discovered before the failure
/// (still owned + freed by `deinit`).
pub const Graph = struct {
    modules: []Module = &.{},
    entry_index: u32 = 0,
    err: ?Error = null,

    pub fn deinit(g: *Graph, gpa: std.mem.Allocator) void {
        for (g.modules) |*m| m.deinit(gpa);
        gpa.free(g.modules);
        if (g.err) |*e| e.deinit(gpa);
        g.* = undefined;
    }

    /// Tear down a graph built by `single`: free ONLY the one-element `modules`
    /// spine. Every module field is BORROWED from the caller (see `single`), so
    /// freeing them (as `deinit` does) would be a double-free. Use this — not
    /// `deinit` — for a `single`-built graph.
    pub fn deinitSingle(g: *Graph, gpa: std.mem.Allocator) void {
        std.debug.assert(g.modules.len == 1);
        std.debug.assert(g.err == null);
        gpa.free(g.modules);
        g.* = undefined;
    }

    pub fn entry(g: *const Graph) *const Module {
        return &g.modules[g.entry_index];
    }
};

/// Discover the module graph from `entry_path`. Reuses the on-disk lex/parse
/// cache (`cache`) so warm rebuilds are free. Returns a `Graph`; the caller owns
/// it and must `deinit`. A structural error is reported via `Graph.err` (NOT a
/// thrown error) so the driver can render it against the owning module's source;
/// only true I/O / OOM failures propagate as Zig errors.
pub fn discover(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    entry_path: []const u8,
) !Graph {
    return discoverDag(gpa, io, cache, target, entry_path, null);
}

/// Same as `discover`, but threads a borrowed `*Dag` so the lex/parse queries run
/// during discovery record their nodes/edges into the per-build graph. The program
/// build (the `Orchestrator`'s DISCOVER stage) threads a non-null dag whenever one
/// exists — a Debug `-o` build, `--dump-dag`, or `--query-stats`; it is null in a
/// release build (verbatim fast path, byte-identical). The `discover` wrapper passes
/// `null` for the single-module / test callers that record no DAG.
pub fn discoverDag(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    entry_path: []const u8,
    dag: ?*Dag,
) !Graph {
    var d: Discoverer = .{
        .gpa = gpa,
        .io = io,
        .cache = cache,
        .target = target,
        .dag = dag,
        .root = dirname(entry_path),
    };
    defer d.deinit();

    // The entry module's canonical name is its file stem (basename minus `.toy`
    // if present). Its file path is the entry path verbatim.
    const entry_name = try gpa.dupe(u8, stem(basename(entry_path)));
    errdefer gpa.free(entry_name);
    const entry_file = try gpa.dupe(u8, entry_path);
    errdefer gpa.free(entry_file);

    const entry_canon = d.canonicalize(entry_file) catch |e| {
        gpa.free(entry_name);
        gpa.free(entry_file);
        return e;
    };
    const root_id = d.intern(entry_name, entry_file, entry_canon) catch |e| {
        gpa.free(entry_name);
        gpa.free(entry_file);
        gpa.free(entry_canon);
        return e;
    };

    // DFS from the entry. `visit` loads+parses on first sight, recurses imports
    // (sorted), and threads a structural error back up. The visiting stack drives
    // cycle detection.
    d.visit(root_id) catch |e| switch (e) {
        error.Structural => {}, // recorded in d.err; fall through to build graph
        else => |err| return err,
    };

    return d.toGraph(root_id);
}

/// Build the trivial one-module graph from an ALREADY lexed+parsed single source.
/// A lone source file is a module graph with exactly one module and no imports —
/// the in-memory analogue of `discover` for the case where the front-end already
/// ran lex/parse (the per-file pipeline + the inline test helpers). This is the
/// ONE entry every single-source resolve/typecheck now flows through, so the graph
/// path (`resolveGraph`/`checkGraph`) is the only front-end implementation.
///
/// `name` is the entry module's canonical name (the file stem, e.g. `main`), so
/// the entry `main` stays bare and every other fn qualifies as `<name>.<fn>`; for
/// the single-source id space `name` is otherwise inert. `file` is the on-disk
/// path verbatim (or `""` for in-memory tests).
///
/// All slices are BORROWED, NOT owned: the caller (the per-file `FileResult`, or a
/// test's parse arena) owns them and OUTLIVES every result derived from this graph.
/// This is LOAD-BEARING for correctness, not just an optimization: the typecheck
/// `EnumLayout` carries its variant `name`s as BORROWED slices into `module.source`
/// (see `layout/Engine.zig`), so the `Typecheck.GraphResult` aliases this graph's
/// `source`. If `single` duped the source and the graph were freed before the
/// result is consumed (e.g. `Driver.pipeline`'s `defer graph.deinit` vs the later
/// `lowerGraphProgram`), every variant name would dangle. Borrowing the caller's
/// (longer-lived) source keeps those names valid. Tear down with `deinitSingle`,
/// which frees ONLY the one-element `modules` spine and never the borrowed fields.
///
/// Deterministic by construction: one allocation, zero map iteration, zero sort,
/// no I/O — byte-identical at `-j1` and `-jN` ([C11]). `entry_index = 0` and
/// `imports = &.{}` are LOAD-BEARING: they pin the global id space to pure decl
/// order and suppress every cross-module diagnostic (the import-walk loops in
/// `resolveGraph`/`checkGraph` are no-ops on zero imports).
pub fn single(
    gpa: std.mem.Allocator,
    name: []const u8,
    file: []const u8,
    source: []const u8,
    tokens: []const Token,
    nodes: []const Ast.Node,
    extra: []const u32,
    pub_bits: []const u32,
) !Graph {
    const modules = try gpa.alloc(Module, 1);
    // `Module` holds mutable slices (the discovery path owns them); the graph path
    // only ever READS them, so borrowing the caller's const inputs via @constCast is
    // safe. `deinitSingle` frees none of these — only the `modules` spine.
    modules[0] = .{
        .path = name,
        .file = file,
        .source = @constCast(source),
        .tokens = @constCast(tokens),
        .nodes = @constCast(nodes),
        .extra = @constCast(extra),
        .pub_bits = pub_bits,
        .imports = &.{},
    };
    return .{ .modules = modules, .entry_index = 0, .err = null };
}

// ---- discovery internals ----------------------------------------------------

const DiscoverError = error{ Structural, OutOfMemory };

/// Per-module discovery state during the DFS.
const Color = enum { white, gray, black };

/// A module slot being assembled during discovery. Mirrors `Module` but holds
/// the DFS bookkeeping (color, parent edge for cycle paths) too.
const Slot = struct {
    name: []const u8, // owned (import-text qualifier)
    file: []const u8, // owned (path actually read)
    canon: []const u8 = &.{}, // owned (physical-identity dedup key)
    source: []u8 = &.{},
    tokens: []Token = &.{},
    nodes: []Ast.Node = &.{},
    extra: []u32 = &.{},
    pub_bits: []const u32 = &.{},
    imports: std.ArrayList(u32) = .empty,
    color: Color = .white,
    loaded: bool = false,

    fn deinit(s: *Slot, gpa: std.mem.Allocator) void {
        gpa.free(s.name);
        gpa.free(s.file);
        if (s.canon.len != 0) gpa.free(s.canon);
        gpa.free(s.source);
        gpa.free(s.tokens);
        gpa.free(s.nodes);
        gpa.free(s.extra);
        if (s.pub_bits.len != 0) gpa.free(@constCast(s.pub_bits));
        s.imports.deinit(gpa);
    }
};

const Discoverer = struct {
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    target: []const u8,
    /// M16: per-build dependency sink (null on default builds; set by `--dump-dag`).
    dag: ?*Dag = null,
    /// The root directory (entry file's directory). Borrowed from `entry_path`.
    root: []const u8,
    /// Module slots, indexed by id, in interning order. Entry is id 0.
    slots: std.ArrayList(Slot) = .empty,
    /// Map PHYSICAL FILE IDENTITY (canonicalized resolved path) → slot id. This is
    /// the dedup key, NOT the import-text name: on a case-insensitive volume
    /// (default macOS APFS / Windows) `import util` and `import Util` both open the
    /// SAME file `util.toy` but carry distinct import-text names. Interning by
    /// physical identity collapses them to one module — one set of symbols, one set
    /// of nominal type ids — preserving the M14 "shared module interned ONCE" and
    /// "one physical file = one set of type ids" invariants. The slot owns its
    /// canonical key; this map borrows it.
    by_canon: std.StringHashMapUnmanaged(u32) = .empty,
    err: ?Error = null,

    fn deinit(d: *Discoverer) void {
        for (d.slots.items) |*s| s.deinit(d.gpa);
        d.slots.deinit(d.gpa);
        d.by_canon.deinit(d.gpa);
        // d.err is moved into the Graph by toGraph; if it stayed it's freed there.
    }

    /// Intern a module by PHYSICAL FILE IDENTITY (`canon`), taking ownership of
    /// `name` (import-text qualifier), `file` (path actually read), and `canon`
    /// (the dedup key). Returns the existing id (freeing all three dupes) if the
    /// same physical file was already interned under any spelling.
    fn intern(d: *Discoverer, name: []const u8, file: []const u8, canon: []const u8) !u32 {
        if (d.by_canon.get(canon)) |id| {
            d.gpa.free(name);
            d.gpa.free(file);
            d.gpa.free(canon);
            return id;
        }
        const id: u32 = @intCast(d.slots.items.len);
        try d.slots.append(d.gpa, .{ .name = name, .file = file, .canon = canon });
        try d.by_canon.put(d.gpa, canon, id);
        return id;
    }

    /// Canonicalize a resolved file path to its physical identity: resolves symlinks
    /// and (on a case-insensitive volume) returns the on-disk case-correct path. On
    /// any failure (e.g. the file vanished between the existence check and here) the
    /// import-text-relative path is duped verbatim as a best-effort key — discovery
    /// will then surface the read failure as `missing` at load time.
    fn canonicalize(d: *Discoverer, file: []const u8) ![]u8 {
        if (Io.Dir.cwd().realPathFileAlloc(d.io, file, d.gpa) catch null) |rp| {
            // `realPathFileAlloc` returns a sentinel `[:0]u8` (allocated len+1); to
            // keep `canon` a plain `[]u8` that frees cleanly, copy and release it.
            defer d.gpa.free(rp);
            return d.gpa.dupe(u8, rp);
        }
        return d.gpa.dupe(u8, file);
    }

    /// Record a structural error (first one wins) and signal the DFS to unwind.
    fn fail(d: *Discoverer, e: Error) DiscoverError {
        if (d.err == null) {
            d.err = e;
        } else {
            var tmp = e;
            tmp.deinit(d.gpa);
        }
        return error.Structural;
    }

    /// Visit one module: load+parse on first sight, then recurse its imports in
    /// stable sorted order. Cycle detection via the gray/black coloring.
    fn visit(d: *Discoverer, id: u32) DiscoverError!void {
        {
            const s = &d.slots.items[id];
            if (s.color == .black) return; // already fully processed
            std.debug.assert(s.color == .white); // a gray re-entry is caught below
            s.color = .gray;
        }

        try d.load(id);

        // Collect this module's imports (resolved to slot ids), in sorted order.
        // Re-borrow the slot each loop because `intern`/`append` may realloc the
        // backing slots array, invalidating a held pointer.
        const edges = try d.collectImports(id);
        defer d.gpa.free(edges);

        for (edges) |child| {
            // A back-edge to a gray ancestor is a cycle.
            if (d.slots.items[child].color == .gray) {
                return d.fail(try d.cycleError(id, child));
            }
            if (d.slots.items[child].color == .white) {
                try d.visit(child);
            }
            try d.slots.items[id].imports.append(d.gpa, child);
        }

        d.slots.items[id].color = .black;
    }

    /// Read + lex + parse module `id` (cached). On a parse error, fail.
    fn load(d: *Discoverer, id: u32) DiscoverError!void {
        const file = d.slots.items[id].file;
        const source = Io.Dir.cwd().readFileAlloc(d.io, file, d.gpa, .unlimited) catch {
            // The entry file failing to read is reported by the driver up front;
            // an imported file that the resolver thought existed but cannot be
            // read is treated as missing.
            return d.fail(.{
                .kind = .missing,
                .message = try std.fmt.allocPrint(d.gpa, "cannot read module file '{s}'", .{file}),
                .module = if (id == 0) null else id,
            });
        };

        // Module discovery routes lex/parse through the same query engine as the
        // per-file pipeline, but with discovery's SWALLOW read policy (a failed
        // cache read is a plain miss, `tmp_tag` = module id). Front-end queries
        // ignore the engine's force/verify mode.
        const engine = if (d.dag) |dp| Engine.initDag(d.cache, .normal, dp) else Engine.init(d.cache, .normal);

        // --- lex (cached) ---
        const lexed = try engine.lex(d.gpa, d.io, d.target, source, id, true);
        const tokens = lexed.value;

        // --- parse (cached) ---
        const parsed = try engine.parse(d.gpa, d.io, d.target, source, tokens, id, true);
        if (parsed.tree == null) {
            const s = &d.slots.items[id];
            s.source = source;
            s.tokens = tokens;
            s.loaded = true;
            return d.fail(.{
                .kind = .parse,
                .message = if (parsed.diag) |dg| try d.gpa.dupe(u8, dg.message) else try d.gpa.dupe(u8, "parse error"),
                .module = id,
                .byte_offset = if (parsed.diag) |dg| dg.byte_offset else null,
            });
        }
        const tree: ?Ast.Tree = parsed.tree;

        const s = &d.slots.items[id];
        s.source = source;
        s.tokens = tokens;
        s.nodes = tree.?.nodes;
        s.extra = tree.?.extra;
        s.pub_bits = tree.?.pub_bits;
        s.loaded = true;
    }

    /// Extract this module's imports, resolve each to a slot id, and return them
    /// SORTED by canonical module name (deterministic edge order). A duplicate
    /// import of the same module collapses to one edge.
    fn collectImports(d: *Discoverer, id: u32) DiscoverError![]u32 {
        // Snapshot the tree view; the slot pointer is only safe until intern().
        const s_nodes = d.slots.items[id].nodes;
        const s_extra = d.slots.items[id].extra;
        const s_tokens = d.slots.items[id].tokens;
        const s_source = d.slots.items[id].source;
        if (s_nodes.len == 0) return d.gpa.alloc(u32, 0);

        const tree: Ast.Tree = .{ .nodes = s_nodes, .extra = s_extra };
        const prog = s_nodes[Ast.root(s_nodes)];
        if (prog.tag != .program) return d.gpa.alloc(u32, 0);

        var edges: std.ArrayList(u32) = .empty;
        errdefer edges.deinit(d.gpa);

        for (Ast.rangeSlice(tree, prog.lhs)) |decl_idx| {
            const decl = s_nodes[decl_idx];
            if (decl.tag != .import_decl) continue;

            // Rebuild the `/`-joined import path from the segment TOKEN indices.
            const seg_tokens = Ast.rangeSlice(tree, decl.lhs);
            const import_off = s_tokens[decl.main_token].start; // last-seg token, for diag
            const path = try joinPath(d.gpa, s_tokens, s_source, seg_tokens);
            defer d.gpa.free(path);

            // Validate + resolve to an on-disk file under the root.
            const file = resolveFile(d.gpa, d.root, path) catch |e| switch (e) {
                error.PathEscape => return d.fail(.{
                    .kind = .escape,
                    .message = try std.fmt.allocPrint(d.gpa, "import path '{s}' escapes the module root", .{path}),
                    .module = id,
                    .byte_offset = import_off,
                }),
                else => |err| return err,
            };
            errdefer d.gpa.free(file);

            // Existence check before interning so a missing module is reported
            // with the offending import's location.
            if (!fileExists(d.io, file)) {
                return d.fail(.{
                    .kind = .missing,
                    .message = try std.fmt.allocPrint(d.gpa, "imported module '{s}' not found (looked for '{s}')", .{ path, file }),
                    .module = id,
                    .byte_offset = import_off,
                });
            }

            // Canonicalize to physical file identity so two imports that open the
            // SAME file (e.g. `util` and `Util` on a case-insensitive volume)
            // intern to ONE module — one symbol set, one set of nominal type ids.
            const canon = try d.canonicalize(file);

            // Guardrail: on a normalizing FS, `import Util` against `util.toy`
            // silently opens the file but requests the wrong case. The canonical
            // path's last segment is the on-disk (case-correct) filename; if it
            // differs from the requested `<lastSeg>.toy`, the import spelling is
            // wrong. Reject with a clear diagnostic rather than (now) aliasing onto
            // the canonical module under a mismatched qualifier.
            {
                const last_seg = basename(path); // last `/`-segment of the import
                const want = try std.fmt.allocPrint(d.gpa, "{s}{s}", .{ last_seg, ext });
                defer d.gpa.free(want);
                const got = basename(canon); // on-disk filename
                if (!std.mem.eql(u8, want, got)) {
                    // Build the message BEFORE freeing canon (`got` borrows it).
                    const msg = try std.fmt.allocPrint(d.gpa, "imported module '{s}' not found (on-disk file '{s}' differs in case from the import path; looked for '{s}')", .{ path, got, file });
                    d.gpa.free(canon);
                    return d.fail(.{
                        .kind = .missing,
                        .message = msg,
                        .module = id,
                        .byte_offset = import_off,
                    });
                }
            }

            const name = try d.gpa.dupe(u8, path);
            const child = d.intern(name, file, canon) catch |e| {
                d.gpa.free(name);
                d.gpa.free(file);
                d.gpa.free(canon);
                return e;
            };
            // Deduplicate edges (two `import a/b` in one file → one edge).
            var seen = false;
            for (edges.items) |existing| {
                if (existing == child) {
                    seen = true;
                    break;
                }
            }
            if (!seen) try edges.append(d.gpa, child);
        }

        const out = try edges.toOwnedSlice(d.gpa);
        // Stable sort by canonical module name for deterministic recursion order.
        std.mem.sort(u32, out, d, struct {
            fn lt(ctx: *Discoverer, a: u32, b: u32) bool {
                return std.mem.lessThan(u8, ctx.slots.items[a].name, ctx.slots.items[b].name);
            }
        }.lt);
        return out;
    }

    /// Build the `/`-arrow cycle path from the gray ancestor `target` to `from`,
    /// then back to `target`, by walking the gray frontier. We do not keep an
    /// explicit DFS stack, so reconstruct the cycle from the gray coloring: every
    /// gray module is on the current stack. The message lists them in name order
    /// joined by ` -> ` with the closing edge back to `target`.
    fn cycleError(d: *Discoverer, from: u32, target: u32) DiscoverError!Error {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(d.gpa);
        try buf.appendSlice(d.gpa, d.slots.items[target].name);
        // List the other gray modules (excluding target) to show the cycle body,
        // then close back on target. The exact interior order is the gray set in
        // id order — deterministic and sufficient to name the cycle.
        for (d.slots.items, 0..) |s, i| {
            if (s.color == .gray and i != target) {
                try buf.appendSlice(d.gpa, " -> ");
                try buf.appendSlice(d.gpa, s.name);
            }
        }
        try buf.appendSlice(d.gpa, " -> ");
        try buf.appendSlice(d.gpa, d.slots.items[target].name);
        const detail = try buf.toOwnedSlice(d.gpa);
        errdefer d.gpa.free(detail);
        const msg = try std.fmt.allocPrint(d.gpa, "import cycle detected: {s}", .{detail});
        // Report at the import in `from` that closed the cycle.
        return Error{
            .kind = .cycle,
            .message = msg,
            .module = from,
            .byte_offset = importTokenOffset(d, from, target),
            .detail = detail,
        };
    }

    /// Move the assembled slots into a `Graph`. Consumes the discoverer's slot
    /// arrays (their owned fields transfer to the Module list).
    fn toGraph(d: *Discoverer, root_id: u32) !Graph {
        const mods = try d.gpa.alloc(Module, d.slots.items.len);
        for (d.slots.items, 0..) |*s, i| {
            mods[i] = .{
                .path = s.name,
                .file = s.file,
                .source = s.source,
                .tokens = s.tokens,
                .nodes = s.nodes,
                .extra = s.extra,
                .pub_bits = s.pub_bits,
                .imports = try s.imports.toOwnedSlice(d.gpa),
            };
            // The canonical dedup key is internal to discovery; it is NOT carried
            // into the Module, so free it here before nulling the slot.
            if (s.canon.len != 0) d.gpa.free(s.canon);
            // Null out the moved fields so Discoverer.deinit doesn't double-free.
            s.* = .{ .name = &.{}, .file = &.{} };
        }
        const e = d.err;
        d.err = null; // ownership moves into the Graph
        return .{ .modules = mods, .entry_index = root_id, .err = e };
    }
};

/// The byte offset of the import statement in module `from` that targets module
/// `target` (for the cycle diagnostic location). Falls back to 0.
fn importTokenOffset(d: *Discoverer, from: u32, target: u32) ?u32 {
    const s = &d.slots.items[from];
    if (s.nodes.len == 0) return null;
    const tree: Ast.Tree = .{ .nodes = s.nodes, .extra = s.extra };
    const prog = s.nodes[Ast.root(s.nodes)];
    if (prog.tag != .program) return null;
    for (Ast.rangeSlice(tree, prog.lhs)) |decl_idx| {
        const decl = s.nodes[decl_idx];
        if (decl.tag != .import_decl) continue;
        const seg_tokens = Ast.rangeSlice(tree, decl.lhs);
        const path = joinPath(d.gpa, s.tokens, s.source, seg_tokens) catch return null;
        defer d.gpa.free(path);
        if (std.mem.eql(u8, path, d.slots.items[target].name)) return s.tokens[decl.main_token].start;
    }
    return s.tokens[0].start;
}

// ---- path helpers ------------------------------------------------------------

/// Join the import path segments with `/`.
fn joinPath(gpa: std.mem.Allocator, tokens: []const Token, source: []const u8, segs: []const u32) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    for (segs, 0..) |seg, i| {
        if (i != 0) try buf.append(gpa, '/');
        try buf.appendSlice(gpa, tokens[seg].text(source));
    }
    return buf.toOwnedSlice(gpa);
}

const PathError = error{ PathEscape, OutOfMemory };

/// Resolve a `/`-joined import path to an on-disk file under `root`:
/// `<root>/<path>.toy`. Rejects any path that escapes the root: an absolute
/// path, an empty segment, a `.`/`..` segment, or a backslash. The grammar makes
/// each segment an identifier, so escapes shouldn't tokenize — but this is the
/// defense-in-depth check the design mandates at the driver boundary.
fn resolveFile(gpa: std.mem.Allocator, root: []const u8, path: []const u8) PathError![]u8 {
    if (path.len == 0) return error.PathEscape;
    if (path[0] == '/') return error.PathEscape; // absolute
    var it = std.mem.splitScalar(u8, path, '/');
    while (it.next()) |seg| {
        if (seg.len == 0) return error.PathEscape; // empty segment (`a//b`, leading/trailing `/`)
        if (std.mem.eql(u8, seg, ".") or std.mem.eql(u8, seg, "..")) return error.PathEscape;
        if (std.mem.indexOfScalar(u8, seg, '\\') != null) return error.PathEscape;
        if (std.mem.indexOfScalar(u8, seg, '/') != null) return error.PathEscape; // (split removes these; defensive)
    }
    if (root.len == 0) {
        return std.fmt.allocPrint(gpa, "{s}{s}", .{ path, ext });
    }
    return std.fmt.allocPrint(gpa, "{s}/{s}{s}", .{ root, path, ext });
}

/// Whether `path` names an existing readable regular file.
fn fileExists(io: Io, path: []const u8) bool {
    var f = Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    f.close(io);
    return true;
}

/// Directory portion of `path` (everything before the last `/`), or `""` when
/// there is no separator (a bare filename → the current directory).
fn dirname(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| return path[0..i];
    return "";
}

/// Final path component (after the last `/`).
fn basename(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| return path[i + 1 ..];
    return path;
}

/// Strip a trailing `.toy` extension from a filename, if present.
fn stem(name: []const u8) []const u8 {
    if (std.mem.endsWith(u8, name, ext)) return name[0 .. name.len - ext.len];
    return name;
}

// ---- tests -------------------------------------------------------------------

const testing = std.testing;

/// Spin up a threaded Io + a cache rooted in a unique temp dir, write the given
/// `{path, source}` files, run discovery from `entry`, and hand the graph to
/// `check`. Cleans up the temp tree afterwards.
const FixtureFile = struct { path: []const u8, source: []const u8 };

fn withFixture(
    comptime dir_name: []const u8,
    files: []const FixtureFile,
    entry: []const u8,
    check: *const fn (gpa: std.mem.Allocator, g: *Graph) anyerror!void,
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
        if (std.mem.lastIndexOfScalar(u8, full, '/')) |i| {
            try Io.Dir.cwd().createDirPath(io, full[0..i]);
        }
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = full, .data = f.source });
    }

    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cache_dir = try std.fmt.bufPrint(&dir_buf, "{s}/.cache", .{dir_name});
    const cache = try Cache.init(io, cache_dir);

    var entry_buf: [std.fs.max_path_bytes]u8 = undefined;
    const entry_path = try std.fmt.bufPrint(&entry_buf, "{s}/{s}", .{ dir_name, entry });

    var g = try discover(gpa, io, cache, "native", entry_path);
    defer g.deinit(gpa);
    try check(gpa, &g);
}

test "discover: 3-module graph (entry -> two imports, one shared)" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import geometry/rect
        \\import util
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "geometry/rect.toy", .source =
        \\import util
        \\pub fn area() -> int { return 1 }
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub fn helper() -> int { return 2 }
        \\
        },
    };
    const Check = struct {
        fn run(_: std.mem.Allocator, g: *Graph) anyerror!void {
            try testing.expect(g.err == null);
            // entry + geometry/rect + util = 3 modules.
            try testing.expectEqual(@as(usize, 3), g.modules.len);
            try testing.expectEqualStrings("main", g.entry().path);
            // Entry imports both, sorted: geometry/rect < util.
            try testing.expectEqual(@as(usize, 2), g.entry().imports.len);
            const imp0 = g.modules[g.entry().imports[0]];
            const imp1 = g.modules[g.entry().imports[1]];
            try testing.expectEqualStrings("geometry/rect", imp0.path);
            try testing.expectEqualStrings("util", imp1.path);
            // The shared `util` is interned ONCE (3 modules total, not 4).
            var util_count: usize = 0;
            for (g.modules) |m| {
                if (std.mem.eql(u8, m.path, "util")) util_count += 1;
            }
            try testing.expectEqual(@as(usize, 1), util_count);
        }
    };
    try withFixture(".toy-test-graph-3mod", files, "main.toy", Check.run);
}

test "discover: import cycle is rejected with a cycle path" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "a.toy", .source =
        \\import b
        \\pub fn fa() -> int { return 1 }
        \\
        },
        .{ .path = "b.toy", .source =
        \\import a
        \\pub fn fb() -> int { return 2 }
        \\
        },
    };
    const Check = struct {
        fn run(_: std.mem.Allocator, g: *Graph) anyerror!void {
            try testing.expect(g.err != null);
            try testing.expectEqual(Error.Kind.cycle, g.err.?.kind);
            // The cycle path names both `a` and `b` and closes on itself.
            try testing.expect(std.mem.indexOf(u8, g.err.?.detail, "a") != null);
            try testing.expect(std.mem.indexOf(u8, g.err.?.detail, "b") != null);
            try testing.expect(std.mem.indexOf(u8, g.err.?.detail, "->") != null);
        }
    };
    try withFixture(".toy-test-graph-cycle", files, "main.toy", Check.run);
}

test "discover: missing import errors cleanly with location" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import nope/missing
        \\fn main() -> int { return 0 }
        \\
        },
    };
    const Check = struct {
        fn run(_: std.mem.Allocator, g: *Graph) anyerror!void {
            try testing.expect(g.err != null);
            try testing.expectEqual(Error.Kind.missing, g.err.?.kind);
            try testing.expectEqual(@as(?u32, 0), g.err.?.module); // reported in entry
            try testing.expect(g.err.?.byte_offset != null);
            try testing.expect(std.mem.indexOf(u8, g.err.?.message, "nope/missing") != null);
        }
    };
    try withFixture(".toy-test-graph-missing", files, "main.toy", Check.run);
}

test "discover: self-import is a cycle" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import main
        \\fn main() -> int { return 0 }
        \\
        },
    };
    const Check = struct {
        fn run(_: std.mem.Allocator, g: *Graph) anyerror!void {
            try testing.expect(g.err != null);
            try testing.expectEqual(Error.Kind.cycle, g.err.?.kind);
        }
    };
    try withFixture(".toy-test-graph-self", files, "main.toy", Check.run);
}

test "discover: single entry with no imports" {
    const files = &[_]FixtureFile{
        .{ .path = "solo.toy", .source =
        \\fn main() -> int { return 0 }
        \\
        },
    };
    const Check = struct {
        fn run(_: std.mem.Allocator, g: *Graph) anyerror!void {
            try testing.expect(g.err == null);
            try testing.expectEqual(@as(usize, 1), g.modules.len);
            try testing.expectEqualStrings("solo", g.entry().path);
            try testing.expectEqual(@as(usize, 0), g.entry().imports.len);
        }
    };
    try withFixture(".toy-test-graph-solo", files, "solo.toy", Check.run);
}

test "single: trivial one-module graph from a parsed source" {
    const gpa = testing.allocator;
    const Lexer = @import("../lex.zig");
    const Parser = @import("../parse.zig");
    const src = "fn main() -> int { return 0 }\n";
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    var diag: ?Parser.Diagnostic = null;
    const tree = (try Parser.parse(gpa, tokens, src, &diag)).?;
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
        if (tree.pub_bits.len != 0) gpa.free(@constCast(tree.pub_bits));
    }

    var g = try single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinitSingle(gpa);
    try testing.expect(g.err == null);
    try testing.expectEqual(@as(usize, 1), g.modules.len);
    try testing.expectEqual(@as(u32, 0), g.entry_index);
    try testing.expectEqual(@as(usize, 0), g.entry().imports.len);
    try testing.expectEqualStrings("main", g.entry().path);
    // The graph BORROWS the caller's slices (same backing memory), so the variant
    // names a downstream EnumLayout aliases stay valid for the caller's lifetime.
    try testing.expectEqual(src.ptr, g.entry().source.ptr);
    try testing.expectEqual(tokens.ptr, g.entry().tokens.ptr);
}

test "resolveFile rejects path escapes" {
    const gpa = testing.allocator;
    try testing.expectError(error.PathEscape, resolveFile(gpa, "root", "/abs/path"));
    try testing.expectError(error.PathEscape, resolveFile(gpa, "root", ".."));
    try testing.expectError(error.PathEscape, resolveFile(gpa, "root", "a/../b"));
    try testing.expectError(error.PathEscape, resolveFile(gpa, "root", "a//b"));
    const ok = try resolveFile(gpa, "root", "a/b");
    defer gpa.free(ok);
    try testing.expectEqualStrings("root/a/b.toy", ok);
}
