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
    /// Who owns the module FIELDS (source/tokens/nodes/extra/pub_bits): `discover`
    /// allocates and OWNS them; `single` BORROWS the caller's const inputs via
    /// `@constCast`. Non-defaulted so every constructor must state it and a partial
    /// `Graph{}` fails to compile — an omission here would be a double-free or leak.
    ownership: Ownership,
    modules: []Module = &.{},
    entry_index: u32 = 0,
    err: ?Error = null,

    pub const Ownership = enum { owned, borrowed };

    /// Tear down per `ownership`: the right teardown is impossible to pick wrong
    /// because the value records who built it.
    pub fn deinit(g: *Graph, gpa: std.mem.Allocator) void {
        switch (g.ownership) {
            // `discover` owns every module field (and any error message).
            .owned => {
                for (g.modules) |*m| m.deinit(gpa);
                if (g.err) |*e| e.deinit(gpa);
            },
            // `single` borrows the caller's const inputs (see `single`), so freeing a
            // module field would double-free — free ONLY the one-element spine below.
            .borrowed => {},
        }
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
    /// `--timings` probe for the discover stage's lex+parse queries (file-read+lex+parse
    /// COMPUTE vs served from the lex/parse content cache). BORROWED; null on a plain
    /// build => the front-end queries read no clock. PERF P4.
    probe: ?*Engine.StageProbe,
) !Graph {
    var d: Discoverer = .{
        .gpa = gpa,
        .io = io,
        .cache = cache,
        .target = target,
        .probe = probe,
        .root = dirname(entry_path),
    };
    defer d.deinit();

    // The entry module's canonical name is its file stem (basename minus `.toy`
    // if present). Its file path is the entry path verbatim.
    const entry_name = try gpa.dupe(u8, stem(basename(entry_path)));
    errdefer gpa.free(entry_name);
    const entry_file = try gpa.dupe(u8, entry_path);
    errdefer gpa.free(entry_file);

    // A null canon means the entry file is unopenable; dupe the path verbatim as a
    // best-effort key so interning still pins the entry to id 0 and `load`'s
    // `readFileAlloc` surfaces the read failure (the driver reports a bad entry).
    const entry_canon = (d.canonicalize(entry_file) catch |e| {
        gpa.free(entry_name);
        gpa.free(entry_file);
        return e;
    }) orelse (gpa.dupe(u8, entry_file) catch |e| {
        gpa.free(entry_name);
        gpa.free(entry_file);
        return e;
    });
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
/// (longer-lived) source keeps those names valid. Built with `ownership = .borrowed`,
/// so `deinit` frees ONLY the one-element `modules` spine, never the borrowed fields.
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
    // safe. The `.borrowed` ownership tag frees none of these — only the `modules` spine.
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
    return .{ .ownership = .borrowed, .modules = modules, .entry_index = 0, .err = null };
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
    /// PERF P4: `--timings` probe for this stage's lex+parse queries (compute vs
    /// cache). Null on a plain build => no clock reads.
    probe: ?*Engine.StageProbe = null,
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
    /// Memoize `realpath` BY RAW RESOLVED PATH (`<root>/<path>.toy`). A module shared
    /// by N importers is resolved to the same `file` string N times (once per edge,
    /// BEFORE `intern` dedups by physical identity), so without this each shared file
    /// pays N realpath(2) syscalls — the discover-time dominator on a graph with reuse.
    /// Same input path → same realpath output on a stable FS (the build contract; the
    /// tree is not mutated mid-build), so this is a pure memoization: it changes which
    /// modules are discovered, their order, ids, and the case/symlink guardrails not at
    /// all ([C11] byte-identical).
    /// Owns one master copy of each key (raw path) and value (canon); callers get a
    /// fresh dupe so the existing intern/free ownership is unchanged.
    realpath_cache: std.StringHashMapUnmanaged([]const u8) = .empty,
    err: ?Error = null,

    fn deinit(d: *Discoverer) void {
        for (d.slots.items) |*s| s.deinit(d.gpa);
        d.slots.deinit(d.gpa);
        d.by_canon.deinit(d.gpa);
        var it = d.realpath_cache.iterator();
        while (it.next()) |e| {
            d.gpa.free(e.key_ptr.*);
            d.gpa.free(e.value_ptr.*);
        }
        d.realpath_cache.deinit(d.gpa);
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
    /// and (on a case-insensitive volume) returns the on-disk case-correct path.
    /// Returns `null` when the file does not exist (or is otherwise unopenable) —
    /// `realPathFileAlloc` opens the file to read its real path, so a null here is
    /// the same presence gate the now-removed separate open+close existence check
    /// gave, folding two syscalls per import into one. A non-null result is a freshly OWNED canonical
    /// key the caller takes (intern adopts it, or frees it on a physical-identity dedup).
    /// Memoized by raw path (`realpath_cache`): a file reached by multiple importers
    /// realpaths once; later hits return a dupe of the cached canon.
    fn canonicalize(d: *Discoverer, file: []const u8) !?[]u8 {
        if (d.realpath_cache.get(file)) |canon| return try d.gpa.dupe(u8, canon);
        const rp = Io.Dir.cwd().realPathFileAlloc(d.io, file, d.gpa) catch return null;
        // `realPathFileAlloc` returns a sentinel `[:0]u8` (allocated len+1); to keep
        // `canon` a plain `[]u8` that frees cleanly, copy and release it.
        defer d.gpa.free(rp);
        const master = try d.gpa.dupe(u8, rp);
        errdefer d.gpa.free(master);
        const key = try d.gpa.dupe(u8, file);
        errdefer d.gpa.free(key);
        // Dupe the caller's copy BEFORE handing `key`/`master` to the cache: after a
        // successful `put` the cache OWNS both, so no fallible op may follow that an
        // errdefer would unwind into a free of cache-owned (and later deinit-freed)
        // memory. `put` is therefore the last fallible step.
        const ret = try d.gpa.dupe(u8, master);
        errdefer d.gpa.free(ret);
        try d.realpath_cache.put(d.gpa, key, master);
        return ret;
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

    /// Obtain module `id`'s source + tokens + AST, then fill its slot. On a parse error,
    /// fail.
    ///
    /// WARM-DISCOVER FAST PATH (PERF): discover unconditionally read + lex + parsed every
    /// module every build. On a warm rebuild lex/parse are content-cache HITS (memory
    /// lookups, no compute), so the residual cost is the per-file READ syscalls — and that
    /// is what dominated warm discover. The fix: a per-file MANIFEST (persisted alongside
    /// the content cache) records each file's last-build `{mtime, size, ctime, content_fp}`.
    /// When a `stat()` whose (mtime, size, ctime) all MATCH the entry says the file is
    /// unchanged, we serve its source + tokens + AST straight from the bulk-loaded pack
    /// (keyed by the recorded `content_fp`) and SKIP the read entirely — discover for that
    /// file collapses to one stat + three in-memory cache-gets. The full read + lex + parse
    /// path also caches the SOURCE bytes (not just tokens/AST) precisely so the warm path can
    /// reconstitute a module without touching the disk.
    ///
    /// SOUNDNESS — the (mtime, size, ctime) all-match is the UNCHANGED-PREDICATE (the central
    /// risk: a wrong oracle = a silent MISCOMPILE). It rests on the BUILD CONTRACT: the source
    /// tree is not mutated mid-build and a write updates mtime — the SAME assumption discover's
    /// `realpath_cache` already relies on (see `Cache.Pack.ManifestEntry` for the full
    /// rationale). ctime (the inode status-change time, read from the SAME stat) additionally
    /// catches the same-size-mtime-restore window (`touch -r`/`cp -p`) at no extra I/O. Every
    /// uncertain case FALLS BACK to the full read + lex + parse: no manifest entry, a failed
    /// stat, a different size, a moved mtime, a moved ctime, or any of the three cache blobs
    /// (source/lex/parse) missing. "When uncertain, READ."
    ///
    /// The serve changes ONLY how a file's source/tokens/AST are obtained — never WHICH
    /// files are discovered, in what order, or the import edges (those come from the AST,
    /// which is byte-identical to a fresh parse because the served source is byte-identical
    /// to the bytes the priming build read) — so the graph + emitted bytes are unchanged
    /// ([C11]).
    fn load(d: *Discoverer, id: u32) DiscoverError!void {
        const file = d.slots.items[id].file;
        // Snapshot (mtime, size, ctime) FIRST — one cheap stat, BEFORE any read. This is the
        // warm-discover unchanged-predicate (see `warmServe`): a match against the
        // manifest collapses this file's discovery to stat + cache-gets, with NO read.
        // `st` is optional: a stat failure just falls through to the full read path (and
        // records no manifest entry, so next build re-reads too).
        const pre_stat: ?Io.Dir.Stat = Io.Dir.cwd().statFile(d.io, file, .{}) catch null;

        // WARM SERVE (read + lex + parse skip): when the manifest's (mtime, size, ctime) all
        // match AND the file's cached source + tokens + AST are all present in the bulk-loaded
        // pack, serve them straight from memory — no file read. This is the warm-discover
        // fast path the whole manifest exists for.
        if (pre_stat) |st| {
            if (d.warmServe(id, st)) return;
        }

        // FULL PATH: the warm serve missed (new file / changed mtime or size / a cache
        // blob absent), so READ the file and lex + parse it. Charge the read to the
        // probe's COMPUTE bucket (it is the never-cached part of discover's work).
        const read_t0: i128 = if (d.probe != null) Engine.StageProbe.now(d.io) else 0;
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
        // No `errdefer free(source)`: `source` is adopted into the slot on EVERY non-read
        // exit below (parse error and success), and the slot/Module owns it thereafter
        // (freed once by `deinit`/`toGraph`). An errdefer here would double-free it on the
        // parse-error path (the slot already holds it when `fail` unwinds).
        if (d.probe) |p| p.lapCompute(d.io, read_t0);

        // The lex/parse/source cache key is `Wyhash(0, source)` of the bytes JUST read.
        const content_fp = std.hash.Wyhash.hash(0, source);

        // Routed through the same query engine as the per-file pipeline, but with
        // discovery's SWALLOW read policy (a failed cache read is a plain miss, `tmp_tag`
        // = module id). The lex/parse queries are themselves cache-first.
        const engine = Engine.initProbe(d.cache, .normal, d.probe);

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

        // Cache the SOURCE bytes (keyed by content_fp) so next build's warm path can serve
        // them from the pack without re-reading the file. Swallowed like every other
        // discover put: a failed store just means next build re-reads this one file.
        d.cache.put(u8, d.io, Cache.Key.fromSource(.source, d.target, source), id, source) catch {};

        const s = &d.slots.items[id];
        s.source = source;
        s.tokens = tokens;
        s.nodes = tree.?.nodes;
        s.extra = tree.?.extra;
        s.pub_bits = tree.?.pub_bits;
        s.loaded = true;

        // Record this file's snapshot for next build's warm-discover unchanged-predicate.
        // Only on a SUCCESSFUL parse + a successful pre-read stat: lex/parse/source all
        // cached their blobs above on this path, so a recorded entry's content_fp points at
        // a complete {source, tokens, AST} triple. Keyed by the canonical path (identity).
        recordManifest(d, id, pre_stat, content_fp);
    }

    /// Try to serve module `id`'s SOURCE + tokens + AST from the bulk-loaded content
    /// cache WITHOUT reading the file. Returns true (slot filled) on a serve, false (the
    /// caller falls through to the full read + lex + parse path) on ANY doubt. This is the
    /// warm-discover fast path: a hit collapses the file's discovery to one stat + three
    /// in-memory cache-gets (no open/read/close syscalls).
    ///
    /// The chain of guards, each falling back to false:
    ///   1. a manifest entry exists for this file's canonical path,
    ///   2. the stat's (mtime, size, ctime) all MATCH the entry — the UNCHANGED-PREDICATE,
    ///   3. the source cache holds the bytes for that content_fp,
    ///   4. the lex cache holds tokens for it,
    ///   5. the parse cache holds a blob for it that `Ast.unpack` validates.
    /// Only when all five hold do we adopt the cached {source, tokens, AST}.
    ///
    /// SOUNDNESS — why the (mtime, size, ctime) all-match is the unchanged-predicate here
    /// (the central risk: a wrong oracle = a silent MISCOMPILE). The match decides the file is
    /// UNCHANGED and serves the bytes the last build recorded under `content_fp`. It rests
    /// on the BUILD
    /// CONTRACT: the source tree is not mutated mid-build, and a write updates the file's
    /// mtime (POSIX writes do; editors/`cp`/compilers do). That is the SAME assumption
    /// `realpath_cache` already relies on (see `discover`'s realpath memo) — we do not
    /// take on a new, stronger assumption. ctime (the inode status-change time) rides the
    /// SAME stat and closes the same-size-mtime-restore window for free. When ANYTHING is
    /// uncertain — no manifest entry, the stat failed (`pre_stat == null`, handled by the
    /// caller), a different size, a moved mtime, a moved ctime, or any of the three cache
    /// blobs missing — we FALL BACK to the full read + lex + parse. "When uncertain, READ."
    ///
    /// The serve changes ONLY HOW a file's source/tokens/AST are obtained, never WHICH
    /// files are discovered or in what order: the served source is byte-identical to the
    /// bytes the priming build read (same content_fp), so its tokens (offsets into source)
    /// resolve identically, the cached AST yields the SAME import edges, and the DFS /
    /// global-id assignment are unchanged. Output is byte-identical ([C11]).
    ///
    /// Adding ctime closes the same-size-mtime-restore windows (`touch -r` / `cp -p` /
    /// a coarse-granularity FS tick): each moves ctime, so they no longer serve stale. The
    /// ONLY residual hole is a tool faking ALL of mtime + size + ctime + content at once —
    /// not a shape a normal build produces. The alternative (re-read + rehash every file
    /// every warm build) defeats the whole purpose of the manifest and is the bottleneck
    /// this fixes.
    fn warmServe(d: *Discoverer, id: u32, st: Io.Dir.Stat) bool {
        const canon = d.slots.items[id].canon;
        const entry = d.cache.manifestGet(canon) orelse return false;

        // The unchanged-predicate: same size AND same mtime AND same ctime => the file is
        // unchanged since the build that recorded this entry. Any mismatch falls back to a
        // full re-read. ctime (the inode STATUS-CHANGE time) rides this SAME stat, so it
        // catches the same-size-mtime-restore window (`touch -r`/`cp -p`, where the metadata
        // write moves ctime even though mtime was pinned) at zero extra I/O — see
        // `Cache.Pack.ManifestEntry`'s soundness contract.
        const mtime: i64 = std.math.cast(i64, st.mtime.nanoseconds) orelse return false;
        const ctime: i64 = std.math.cast(i64, st.ctime.nanoseconds) orelse return false;
        if (mtime != entry.mtime or st.size != entry.size or ctime != entry.ctime) return false;

        // OWNERSHIP: every cache-get is owned by a `defer free()` guarded by `served`,
        // which is set ONLY on the all-hit exit. A `return false` past any get frees what
        // it fetched and leaves the slot untouched (the caller then takes the full path).
        // PERF P4: charge the warm serve's three cache-gets + unpack to the discover GET
        // bucket so `--timings` attributes the warm path (it bypasses `Engine.query`).
        const get_t0: i128 = if (d.probe != null) Engine.StageProbe.now(d.io) else 0;
        var served = false;
        const source = (d.cache.get(u8, d.gpa, d.io, Cache.Key.fromFingerprint(.source, d.target, entry.content_fp)) catch null) orelse return false;
        defer if (!served) d.gpa.free(source);

        const lex_key = Cache.Key.fromFingerprint(.lex, d.target, entry.content_fp);
        const tokens = (d.cache.get(Token, d.gpa, d.io, lex_key) catch null) orelse return false;
        defer if (!served) d.gpa.free(tokens);

        const parse_key = Cache.Key.fromFingerprint(.parse, d.target, entry.content_fp);
        const blob = (d.cache.get(u8, d.gpa, d.io, parse_key) catch null) orelse return false;
        defer d.gpa.free(blob);
        const tree = (Ast.unpack(d.gpa, blob) catch null) orelse return false;
        if (d.probe) |p| p.lapGet(d.io, get_t0);
        // The unpacked tree's arrays are caller-owned; the slot adopts them below. On a
        // fallback past this point there is none (the next steps are infallible), so no
        // tree-free guard is needed.

        served = true;
        const s = &d.slots.items[id];
        s.source = source;
        s.tokens = tokens;
        s.nodes = tree.nodes;
        s.extra = tree.extra;
        s.pub_bits = tree.pub_bits;
        s.loaded = true;

        // Carry the (still-valid) entry forward into THIS build's manifest so it survives
        // the flush (the manifest reflects the current graph, not a union).
        d.cache.manifestPut(canon, entry);
        return true;
    }

    /// Record module `id`'s `{mtime, size, ctime, content_fp}` snapshot for next build's
    /// warm-discover unchanged-predicate, keyed by its canonical path. A no-op when the
    /// stat failed (=> a safe full re-read next build) or either timestamp is unrepresentable.
    fn recordManifest(d: *Discoverer, id: u32, pre_stat: ?Io.Dir.Stat, content_fp: u64) void {
        const st = pre_stat orelse return;
        const mtime = std.math.cast(i64, st.mtime.nanoseconds) orelse return;
        const ctime = std.math.cast(i64, st.ctime.nanoseconds) orelse return;
        d.cache.manifestPut(d.slots.items[id].canon, .{ .mtime = mtime, .size = st.size, .ctime = ctime, .content_fp = content_fp });
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

            // Canonicalize to physical file identity (resolving symlinks/case) so two
            // imports that open the SAME file (e.g. `util` and `Util` on a
            // case-insensitive volume) intern to ONE module — one symbol set, one set
            // of nominal type ids. `canonicalize` ALSO serves as the presence gate:
            // `realPathFileAlloc` opens the file, so a null result means the import
            // names a file that does not exist. Reporting the miss here (at the
            // offending import's location) folds the old separate `openFile`+`close`
            // existence check into this one realpath syscall.
            const canon = (try d.canonicalize(file)) orelse {
                return d.fail(.{
                    .kind = .missing,
                    .message = try std.fmt.allocPrint(d.gpa, "imported module '{s}' not found (looked for '{s}')", .{ path, file }),
                    .module = id,
                    .byte_offset = import_off,
                });
            };

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
        return .{ .ownership = .owned, .modules = mods, .entry_index = root_id, .err = e };
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

    var g = try discover(gpa, io, cache, "native", entry_path, null);
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

test "discover: a parse error is reported and owns its source (no leak/double-free)" {
    // Exercises load()'s parse-error path, where the slot ADOPTS `source` before `fail`
    // unwinds. The test allocator turns any double-free (the bug a stray errdefer on
    // `source` would cause) or leak into a failure.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\fn main( -> int { return 0 }
        \\
        },
    };
    const Check = struct {
        fn run(_: std.mem.Allocator, g: *Graph) anyerror!void {
            try testing.expect(g.err != null);
            try testing.expectEqual(Error.Kind.parse, g.err.?.kind);
            try testing.expectEqual(@as(?u32, 0), g.err.?.module);
        }
    };
    try withFixture(".toy-test-graph-parseerr", files, "main.toy", Check.run);
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
    defer g.deinit(gpa);
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

/// The text of the FIRST `literal_number` token in module `m`'s tree, read against the
/// module's served source. Used by the warm-discover tests to assert which version of the
/// program the served AST reflects (re-read vs cache-served).
fn firstNumberLiteral(m: *const Module) ?[]const u8 {
    for (m.nodes) |n| {
        if (n.tag == .literal_number) return m.tokens[n.main_token].text(m.source);
    }
    return null;
}

test "discover: a same-size edit that MOVES the mtime is NOT served stale" {
    // The central soundness guarantee under the build contract: a real write updates the
    // file's mtime, so even a SIZE-PRESERVING edit (the size alone cannot tell v1 from v2)
    // is caught by the moved mtime and re-read. The warm path serves the cache ONLY when
    // (mtime, size) BOTH match the recorded entry; a moved mtime falls back to a full read.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-stale-serve";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const file = dir ++ "/prog.toy";
    const cache_dir = dir ++ "/.cache";

    // v1 and v2 are the SAME byte length, so the size cannot distinguish them — only the
    // mtime can, which is exactly what the build contract guarantees a write moves.
    const v1 = "fn main() -> int { return 1 }\n";
    const v2 = "fn main() -> int { return 2 }\n";
    try testing.expectEqual(v1.len, v2.len);

    // Build 1 (cold): primes the source/lex/parse cache + the manifest for prog.toy.
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = v1 });
    const mtime_v1 = blk: {
        const st = try Io.Dir.cwd().statFile(io, file, .{});
        break :blk st.mtime;
    };
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expect(g.err == null);
        try testing.expectEqualStrings("1", firstNumberLiteral(g.entry()).?);
        pack.flush(io, cache_dir);
    }

    // EDIT to v2 (same size). A real write moves the mtime; bump it explicitly to a value
    // strictly later than v1's so the test does not depend on the FS's mtime granularity
    // (two writes in the same coarse tick is the documented out-of-contract case, not this).
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = v2 });
    try Io.Dir.cwd().setTimestamps(io, file, .{ .modify_timestamp = .{ .new = .{ .nanoseconds = mtime_v1.nanoseconds + 1_000_000_000 } } });
    {
        const st = try Io.Dir.cwd().statFile(io, file, .{});
        try testing.expect(st.mtime.nanoseconds != mtime_v1.nanoseconds);
    }

    // Build 2 (warm): the mtime moved, so the unchanged-predicate misses and discover
    // re-reads + re-parses, serving v2 (literal "2"), NOT the stale cached "1".
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expect(g.err == null);
        try testing.expectEqualStrings(v2, g.entry().source);
        // The load-bearing assertion: the AST reflects v2 (literal "2"), NOT a stale "1".
        try testing.expectEqualStrings("2", firstNumberLiteral(g.entry()).?);
    }
}

test "discover: a touch -r style edit (mtime restored, ctime moved) is NOT served stale" {
    // The HOLE ctime closes: a same-size edit that ALSO restores the prior mtime
    // (`touch -r ref f`, `cp -p`, two edits in one coarse mtime tick) leaves (mtime, size)
    // matching the recorded entry, so the OLD predicate served the cached v1 STALE. But ANY
    // such metadata write moves ctime, so requiring ctime to match too catches it WITHOUT a
    // read. We model that here by writing v2 (same size as v1), then forcing the file's
    // (mtime, size) to equal the manifest's recorded values while the manifest's ctime is a
    // sentinel that the file's REAL (moved) ctime can never equal — so warmServe rejects on
    // the ctime field alone and re-reads, serving v2.
    //
    // The manifest ctime is forged (not produced by a real `touch -r`) on purpose: a real
    // restore moves ctime to "now", which a coarse-granularity FS clock could round to the
    // cold build's tick, making the test flaky. Pinning the recorded ctime to a value the
    // current inode cannot hold tests the ctime guard deterministically. The end-to-end
    // `touch -r` PROBE (the real binary) covers the genuine operation.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-stale-touchr";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const file = dir ++ "/prog.toy";
    const cache_dir = dir ++ "/.cache";
    // Same byte length, so only mtime/ctime (not size) can tell v1 from v2.
    const v1 = "fn main() -> int { return 1 }\n";
    const v2 = "fn main() -> int { return 2 }\n";
    try testing.expectEqual(v1.len, v2.len);

    // Cold build on v1: caches {source, tokens, AST} keyed by hash(v1). We do NOT keep its
    // manifest entry — we forge our own below so the ctime mismatch is exact and stable.
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = v1 });
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expectEqualStrings("1", firstNumberLiteral(g.entry()).?);
        pack.flush(io, cache_dir);
    }

    // The touch -r outcome on disk: v2's bytes, but the prior mtime. (A real `touch -r`
    // restores mtime; we restore it explicitly so the test does not depend on whether v2's
    // own write happened to land in the same coarse mtime tick as v1's.)
    const canon = try Io.Dir.cwd().realPathFileAlloc(io, file, gpa);
    defer gpa.free(canon);
    const v1_mtime = (try Io.Dir.cwd().statFile(io, file, .{})).mtime;
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = v2 });
    try Io.Dir.cwd().setTimestamps(io, file, .{ .modify_timestamp = .{ .new = v1_mtime } });
    const now = try Io.Dir.cwd().statFile(io, file, .{});
    const now_ctime: i64 = @intCast(now.ctime.nanoseconds);

    // Forge the manifest: (mtime, size) = the file's CURRENT values (so those two match), but
    // ctime = a sentinel the live inode cannot hold (`now_ctime - 1`), and content_fp =
    // hash(v1) (so a — wrongly — served entry would yield the STALE "1"). The only thing that
    // stops the stale serve is the ctime guard.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        cache.manifestPut(canon, .{
            .mtime = @intCast(now.mtime.nanoseconds),
            .size = now.size,
            .ctime = now_ctime - 1, // the moved ctime can never match this
            .content_fp = std.hash.Wyhash.hash(0, v1),
        });
        pack.flush(io, cache_dir);
    }

    // Warm build: (mtime, size) match the forged entry but ctime does not, so warmServe
    // rejects and discover re-reads v2 ("2"), NOT the stale cached "1".
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expect(g.err == null);
        try testing.expectEqualStrings(v2, g.entry().source);
        try testing.expectEqualStrings("2", firstNumberLiteral(g.entry()).?);
    }
}

test "discover: a changed file whose mtime moves is re-read (size-changing edit)" {
    // Belt-and-suspenders for the common edit shape: adding bytes changes BOTH size and
    // mtime, so the warm-serve predicate misses on two independent fields and re-reads.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-stale-serve-grow";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const file = dir ++ "/prog.toy";
    const cache_dir = dir ++ "/.cache";
    const v1 = "fn main() -> int { return 1 }\n";
    const v2 = "fn main() -> int { return 22 }\n"; // one byte longer

    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = v1 });
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expectEqualStrings("1", firstNumberLiteral(g.entry()).?);
        pack.flush(io, cache_dir);
    }
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = v2 });
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expectEqualStrings(v2, g.entry().source);
        try testing.expectEqualStrings("22", firstNumberLiteral(g.entry()).?);
    }
}

test "discover: an unchanged file is served warm (literal preserved across a clean rebuild)" {
    // The happy path the manifest exists for: same bytes + same (mtime, size) => the cached
    // source + tokens + AST are served WITHOUT reading the file, and the result is identical.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-warm-serve";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const file = dir ++ "/prog.toy";
    const cache_dir = dir ++ "/.cache";
    const src = "fn main() -> int { return 7 }\n";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = src });

    inline for (.{ "cold", "warm" }) |_| {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expect(g.err == null);
        try testing.expectEqualStrings("7", firstNumberLiteral(g.entry()).?);
        pack.flush(io, cache_dir);
    }
}

test "discover: the warm path serves from cache WITHOUT reading the file" {
    // PROVES the read is actually skipped (the whole point of the manifest). A cold build on
    // GOOD bytes caches {source, tokens, AST} keyed by content_fp = hash(good). We then put
    // GARBAGE on disk and, in a priming build, OVERWRITE the manifest entry so its
    // (mtime, size, ctime) equal the GARBAGE file's CURRENT stat but its content_fp still
    // points at the GOOD blobs. The warm build's stat then MATCHES the manifest, so discover
    // serves the cached GOOD tree off content_fp WITHOUT reading the garbage on disk — a
    // clean "5" despite unparseable bytes is the direct read-skip assertion.
    //
    // Why not overwrite-then-restore (mtime, size)? With the ctime hardening that no longer
    // serves stale: the overwrite + any `setTimestamps` BOTH move ctime, so the predicate
    // would (correctly) miss and re-read. Forging the manifest to the garbage file's own
    // current (mtime, size, ctime) is the only way to make all three match while the on-disk
    // CONTENT differs — exactly the residual hole the contract documents, used here to prove
    // the serve bypasses the disk read.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-read-skip";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const file = dir ++ "/prog.toy";
    const cache_dir = dir ++ "/.cache";
    const good = "fn main() -> int { return 5 }\n";
    // Same byte length as `good` but pure garbage that would NOT parse — if discover read
    // it, the build would fail (or serve "garbage"), so a clean "5" proves the read-skip.
    const garbage = "@" ** (good.len - 1) ++ "\n";
    comptime std.debug.assert(good.len == garbage.len);

    // Cold build on GOOD bytes: primes the source/lex/parse blobs under hash(good).
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = good });
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expect(g.err == null);
        try testing.expectEqualStrings("5", firstNumberLiteral(g.entry()).?);
        pack.flush(io, cache_dir);
    }

    // Put GARBAGE on disk, then snapshot ITS current (mtime, size, ctime).
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = garbage });
    const canon = try Io.Dir.cwd().realPathFileAlloc(io, file, gpa);
    defer gpa.free(canon);
    const gst = try Io.Dir.cwd().statFile(io, file, .{});
    const gmtime: i64 = @intCast(gst.mtime.nanoseconds);
    const gctime: i64 = @intCast(gst.ctime.nanoseconds);

    // Priming build: force the manifest entry to the GARBAGE file's stat (so the warm build's
    // stat matches all three fields) but keep content_fp = hash(good) (so the serve resolves
    // to the still-cached GOOD blobs). The GOOD blobs are carried forward by `load`.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        cache.manifestPut(canon, .{ .mtime = gmtime, .size = gst.size, .ctime = gctime, .content_fp = std.hash.Wyhash.hash(0, good) });
        pack.flush(io, cache_dir);
    }

    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        // Served from cache: no parse error despite the garbage on disk, and the GOOD tree.
        try testing.expect(g.err == null);
        try testing.expectEqualStrings(good, g.entry().source);
        try testing.expectEqualStrings("5", firstNumberLiteral(g.entry()).?);
    }
}

test "discover: warm path FALLS BACK to a read when the cache blobs are missing" {
    // Soundness of the fallback chain: even when a manifest entry EXISTS and its
    // (mtime, size, ctime) all match the file, a MISSING source/lex/parse blob (evicted/pruned
    // cache) must NOT serve — warmServe's `cache.get -> null` guards fall back to a full read.
    // We construct exactly this: persist a manifest entry keyed to the file's real
    // (mtime, size, ctime) but NEVER put the blobs, then discover.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir = ".toy-test-warm-fallback";
    Io.Dir.cwd().deleteTree(io, dir) catch {};
    try Io.Dir.cwd().createDirPath(io, dir);
    defer Io.Dir.cwd().deleteTree(io, dir) catch {};

    const file = dir ++ "/prog.toy";
    const cache_dir = dir ++ "/.cache";
    const src = "fn main() -> int { return 9 }\n";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = file, .data = src });

    // The canonical path discover interns by (so the manifest key matches its lookup).
    const canon = try Io.Dir.cwd().realPathFileAlloc(io, file, gpa);
    defer gpa.free(canon);
    const st = try Io.Dir.cwd().statFile(io, file, .{});
    const mtime: i64 = @intCast(st.mtime.nanoseconds);
    const ctime: i64 = @intCast(st.ctime.nanoseconds);

    // Build 1: persist ONLY a manifest entry (matching mtime/size/ctime, with the true
    // content_fp) — NO source/lex/parse blobs. This is the evicted-blob state.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        cache.manifestPut(canon, .{ .mtime = mtime, .size = st.size, .ctime = ctime, .content_fp = std.hash.Wyhash.hash(0, src) });
        pack.flush(io, cache_dir);
    }

    // Build 2: warmServe's manifestGet hits and (mtime, size, ctime) all match, but every
    // blob get misses, so it falls back to a full read + parse — yielding the correct tree.
    {
        var pack = Cache.Pack.init(gpa);
        defer pack.deinit();
        pack.load(gpa, io, cache_dir);
        const cache = try Cache.initPack(io, cache_dir, &pack);
        var g = try discover(gpa, io, cache, "native", file, null);
        defer g.deinit(gpa);
        try testing.expect(g.err == null);
        try testing.expectEqualStrings(src, g.entry().source);
        try testing.expectEqualStrings("9", firstNumberLiteral(g.entry()).?);
    }
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
