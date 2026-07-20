//! Whole-graph name resolution.
//!
//! Generalises the single-file `resolve.zig` across a discovered module `Graph`.
//! Two phases, mirroring the per-file resolver but at graph scope:
//!
//!   * GLOBAL COLLECT — walk every module's top-level decls once. Assign each
//!     user fn a GRAPH-GLOBAL id (deterministic: module order, then decl order)
//!     and a MODULE-QUALIFIED symbol name (`<module-path>.<fn>`), except the entry
//!     module's `main` (stays bare `main`). The seeded builtins (`panic` plus the
//!     core-only intrinsics) are appended after all user fns. Each module records
//!     its own fns (pub +
//!     private, visible only inside the module) plus its `pub` fns/structs/enums
//!     (visible to importers under the import namespace). Each module's `import`
//!     decls bind a namespace name → imported module id, rejecting a same-last-
//!     segment collision (unless `as`).
//!
//!   * PER-MODULE BODY RESOLVE — resolve each fn body in its own lexical state,
//!     exactly like the single-file resolver, with two cross-module additions:
//!       - `lookupName`: locals → this module's fns (incl. seeded builtins) →
//!         import namespaces (innermost lexical binding still shadows a namespace).
//!       - `mod.member` (field_access whose receiver resolves to `.module`): the
//!         receiver gets `.module`, and the member is resolved against the owning
//!         module's PUB tables: a pub fn → `.func` (global id) written ON THE
//!         FIELD_ACCESS node; a pub type → quiet (Typecheck binds it later); a
//!         non-pub or missing member → an error rendered against THIS module.
//!
//! Output: one `[]Resolution` per module (parallel to that module's node array,
//! `.func` carrying GRAPH-GLOBAL fn ids), the global fn table (qualified names +
//! decl sites + owning module), and per-module diagnostics carrying their owning
//! module id so the driver renders each against the right source.
//!
//! This is the ONE name resolver. A lone source file is resolved as the trivial
//! one-module graph (`Graph.single` → `resolveGraph`); there is no separate
//! single-file resolver, and the `GraphResult` is consumed WHOLE everywhere (the
//! driver `FileResult`, `lower`). `resolve.zig` holds only the shared peer types
//! (`Resolution`/`Diagnostic`) those readers spell as `Resolve.*`.

const std = @import("std");
const Token = @import("ast/Token.zig").Token;
const Ast = @import("ast/Ast.zig");
const Graph = @import("driver/Graph.zig");
const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;
const DiagnosticSink = @import("diagnostics/Sink.zig");
const nearmiss = @import("diagnostics/nearmiss.zig");
const codes = @import("diagnostics/codes.zig");

pub const Resolution = @import("symbols/Resolution.zig").Resolution;
const symbols_res = @import("symbols/Resolution.zig");
const SymKind = @import("symbols/Sym.zig").SymKind;
const Intrinsic = @import("symbols/Intrinsic.zig");
const StdNames = @import("symbols/StdNames.zig");

/// A graph-global function symbol.
pub const GlobalFn = struct {
    /// Module-qualified symbol name (`geometry/rect.area`), or bare for the
    /// entry `main` / a seeded builtin (`panic`, intrinsics). Owned by `GraphResult`.
    name: []const u8,
    /// Owning module id (index into `Graph.modules`). For a seeded builtin this is
    /// the entry module (it has no real home; only the bare name matters).
    module: u32,
    /// The `fn_decl` node in that module's tree (`Ast.none` for a bodyless seeded
    /// builtin, identified by `kind` — never test `decl_node` for builtin-ness).
    decl_node: Ast.Index,
    /// `.user_fn` for a real decl, `.builtin` for a seeded builtin (`panic`/intrinsics).
    /// The single source of "is this a builtin", carried across the resolve→types boundary.
    kind: SymKind,
    /// Whether the decl is `pub` (exported). `main` and the seeded builtins are not pub.
    is_pub: bool,
    /// For an inherent method: the receiver type-ref node (an `identifier`) in
    /// this module's tree — Typecheck resolves it to the receiver `Type` and keys the
    /// program-wide method table off it. `Ast.none` for an ordinary fn / builtin.
    recv_type: Ast.Index = Ast.none,
};

/// The whole-graph resolve output. Caller owns it; free with `deinit`.
pub const GraphResult = struct {
    /// One `[]Resolution` per module, parallel to `Graph.modules` (indexed by
    /// module id). Each inner slice is parallel to that module's node array.
    resolutions: [][]Resolution,
    /// The merged program-wide function table; `.func` resolutions index it.
    fns: []GlobalFn,
    /// Diagnostics, each tagged with its owning module via `Diagnostic.scope`.
    diags: []Diagnostic,
    owned_msgs: [][]u8,

    pub fn deinit(self: *GraphResult, gpa: std.mem.Allocator) void {
        for (self.resolutions) |r| gpa.free(r);
        gpa.free(self.resolutions);
        for (self.fns) |f| gpa.free(@constCast(f.name));
        gpa.free(self.fns);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        self.* = undefined;
    }
};

/// Symbol tables collected for one module during the global-collect phase.
const ModuleTables = struct {
    /// This module's fns (pub + private), name → global fn id. Visible only
    /// within the module body (bare calls).
    fns: std.StringHashMapUnmanaged(u32) = .empty,
    /// This module's PUB fns, name → global fn id. Visible to importers.
    pub_fns: std.StringHashMapUnmanaged(u32) = .empty,
    /// This module's struct type names (pub flag as value).
    structs: std.StringHashMapUnmanaged(bool) = .empty,
    /// This module's enum type names (pub flag as value).
    enums: std.StringHashMapUnmanaged(bool) = .empty,
    /// Import namespaces: bound name → imported module id.
    namespaces: std.StringHashMapUnmanaged(u32) = .empty,

    fn deinit(self: *ModuleTables, gpa: std.mem.Allocator) void {
        self.fns.deinit(gpa);
        self.pub_fns.deinit(gpa);
        self.structs.deinit(gpa);
        self.enums.deinit(gpa);
        self.namespaces.deinit(gpa);
    }
};

/// The prelude enums nameable in every module with no import (see `collectGlobals`).
/// A bare reference to one of these credits NO import, so `warnUnusedImports` must
/// exclude them when crediting an import via a bare pub-enum name — otherwise a bare
/// `Option` would spuriously mark EVERY import used (they are injected pub into every
/// module's enum table).
const prelude_enum_names = [_][]const u8{ "Ordering", "Option", "Result" };

const Local = struct { name_tok: u32, slot: u32, used: bool = false, warn_code: codes.Code = .none };
const Scope = struct { names: std.StringHashMapUnmanaged(u32) = .empty };
const LabelEntry = struct { name: []const u8, construct_node: Ast.Index };

const GraphResolve = struct {
    gpa: std.mem.Allocator,
    graph: *const Graph.Graph,

    /// Per-module collected tables (parallel to graph.modules).
    tables: []ModuleTables,
    /// Per-module resolution arrays (parallel to graph.modules).
    resolutions: [][]Resolution,
    /// The growing global fn table.
    fns: std.ArrayList(GlobalFn),

    sink: DiagnosticSink,

    cur_mod: u32 = 0,
    scopes: std.ArrayList(Scope) = .empty,
    locals: std.ArrayList(Local) = .empty,
    slot_next: u32 = 0,
    label_stack: std.ArrayList(LabelEntry) = .empty,

    fn tree(g: *GraphResolve, mod: u32) Ast.Tree {
        const m = &g.graph.modules[mod];
        return .{ .nodes = m.nodes, .extra = m.extra, .pub_bits = m.pub_bits };
    }
    fn nodes(g: *GraphResolve) []const Ast.Node {
        return g.graph.modules[g.cur_mod].nodes;
    }
    fn tokens(g: *GraphResolve) []const Token {
        return g.graph.modules[g.cur_mod].tokens;
    }
    fn source(g: *GraphResolve) []const u8 {
        return g.graph.modules[g.cur_mod].source;
    }
    fn nameOf(g: *GraphResolve, mod: u32, tok: u32) []const u8 {
        const m = &g.graph.modules[mod];
        return m.tokens[tok].text(m.source);
    }
    fn nameText(g: *GraphResolve, tok: u32) []const u8 {
        return g.nameOf(g.cur_mod, tok);
    }

/// A synthetic builtin the resolver seeds after the user fns. `core_only` gates
/// registration into per-module fn tables: `false` = global prelude (every module),
/// `true` = only bundled `core/` modules.
const SeededBuiltin = struct { name: []const u8, core_only: bool };

/// The single ordered authority for the seeded builtins. Order is load-bearing: it
/// fixes each builtin's global fn id, which feeds the content fingerprint / -jN
/// identity — `panic` leads so later appends never shift an existing id.
/// Append, never reorder. The intrinsic names are pulled from `Intrinsic.Kind` (their
/// own id-order authority) so the two lists stay in one derivation, not two spellings.
const seeded_builtins = blk: {
    const kinds = std.enums.values(Intrinsic.Kind);
    var list: [1 + kinds.len]SeededBuiltin = undefined;
    list[0] = .{ .name = "panic", .core_only = false };
    for (kinds, 0..) |k, i| list[1 + i] = .{ .name = Intrinsic.name(k), .core_only = true };
    break :blk list;
};

/// Append one homeless `.builtin` fn for `b` and register its name into the in-scope
/// modules' fn tables (never over a module's own same-name decl — user-first-wins).
fn seedBuiltin(g: *GraphResolve, entry: u32, b: SeededBuiltin) !void {
    const id: u32 = @intCast(g.fns.items.len);
    try g.fns.append(g.gpa, .{
        .name = try g.gpa.dupe(u8, b.name),
        .module = entry,
        .decl_node = Ast.none,
        .kind = .builtin,
        .is_pub = false,
    });
    for (g.tables, 0..) |*t, i| {
        const in_scope = if (b.core_only) g.graph.modules[i].isCore() else true;
        if (in_scope and !t.fns.contains(b.name)) try t.fns.put(g.gpa, b.name, id);
    }
}

test "seeded_builtins: panic leads as global prelude, then the intrinsics core-only" {
    try std.testing.expectEqualStrings("panic", seeded_builtins[0].name);
    try std.testing.expect(!seeded_builtins[0].core_only);
    const kinds = std.enums.values(Intrinsic.Kind);
    try std.testing.expectEqual(1 + kinds.len, seeded_builtins.len);
    for (kinds, 0..) |k, i| {
        try std.testing.expectEqualStrings(Intrinsic.name(k), seeded_builtins[1 + i].name);
        try std.testing.expect(seeded_builtins[1 + i].core_only);
    }
}

/// Assign every user fn a global id + qualified name, and register each module's
/// struct/enum type names. Deterministic: modules in id order, decls in source
/// order. The synthetic builtins are seeded last (see `seeded_builtins`).
fn collectGlobals(g: *GraphResolve) !void {
    const entry = g.graph.entry_index;
    // Scratch set of minted method symbol names, to reject a duplicate `(Receiver,
    // method)` (two `impl P { fn m }`) — which would otherwise be a linker duplicate
    // symbol. Keyed by the mangled qname (owned by `g.fns` once appended); the map's
    // own storage is freed here, the borrowed keys are not.
    var method_names: std.StringHashMapUnmanaged(void) = .empty;
    defer method_names.deinit(g.gpa);
    for (g.graph.modules, 0..) |m, mi| {
        const mod: u32 = @intCast(mi);
        if (m.nodes.len == 0) continue;
        const t = g.tree(mod);
        const prog = m.nodes[Ast.root(m.nodes).int()];
        if (prog.tag != .program) continue;

        for (Ast.rangeSlice(t, prog.lhs.int())) |decl_idx| {
            const decl = m.nodes[decl_idx.int()];
            const is_pub = t.isPub(decl_idx);
            switch (decl.tag) {
                .struct_decl, .tuple_struct_decl => {
                    const name = g.nameOf(mod, decl.main_token);
                    try g.tables[mod].structs.put(g.gpa, name, is_pub);
                },
                .enum_decl => {
                    const name = g.nameOf(mod, decl.main_token);
                    try g.tables[mod].enums.put(g.gpa, name, is_pub);
                },
                .fn_decl => {
                    const name = g.nameOf(mod, decl.main_token);
                    const gop = try g.tables[mod].fns.getOrPut(g.gpa, name);
                    if (gop.found_existing) {
                        const dup_off = m.tokens[decl.main_token].start;
                        // Point a secondary "previously defined here" at the first
                        // definition: the stored map value is its global fn id, whose
                        // decl node lives in THIS module (same scope).
                        if (g.fns.items[gop.value_ptr.*].decl_node.unwrap()) |first_dn| {
                            const first_off = m.tokens[m.nodes[first_dn.int()].main_token].start;
                            try g.emitRelated(.R0002, mod, dup_off, first_off, "duplicate function '{s}'", .{name});
                        } else {
                            try g.emit(.R0002, mod, dup_off, "duplicate function '{s}'", .{name});
                        }
                        continue;
                    }
                    const id: u32 = @intCast(g.fns.items.len);
                    gop.value_ptr.* = id;
                    // Entry `main` stays bare (→ dyld _main); everything else is
                    // qualified by its module's import path.
                    const is_entry_main = mod == entry and std.mem.eql(u8, name, "main");
                    const qname = if (is_entry_main)
                        try g.gpa.dupe(u8, "main")
                    else
                        try std.fmt.allocPrint(g.gpa, "{s}.{s}", .{ m.path, name });
                    try g.fns.append(g.gpa, .{
                        .name = qname,
                        .module = mod,
                        .decl_node = decl_idx,
                        .kind = .user_fn,
                        .is_pub = is_pub,
                    });
                    if (is_pub) try g.tables[mod].pub_fns.put(g.gpa, name, id);
                },
                .extern_fn_decl => {
                    const name = g.nameOf(mod, decl.main_token);
                    // The DECL gate is core-only: an `extern fn` is legal solely in a
                    // bundled `core/` module. Anywhere else it is never registered, so
                    // any use of the name is undeclared (R0001).
                    if (!m.isCore()) {
                        try g.emit(.R0010, mod, m.tokens[decl.main_token].start, "'extern' functions are only allowed in 'core/' modules", .{});
                        continue;
                    }
                    const gop = try g.tables[mod].fns.getOrPut(g.gpa, name);
                    if (gop.found_existing) {
                        try g.emit(.R0002, mod, m.tokens[decl.main_token].start, "duplicate function '{s}'", .{name});
                        continue;
                    }
                    const id: u32 = @intCast(g.fns.items.len);
                    gop.value_ptr.* = id;
                    // The symbol name is the BARE C symbol (never module-qualified):
                    // it is what dyld binds through the `__got`, so `labs` stays `labs`.
                    try g.fns.append(g.gpa, .{
                        .name = try g.gpa.dupe(u8, name),
                        .module = mod,
                        .decl_node = decl_idx,
                        .kind = .import,
                        .is_pub = is_pub,
                    });
                    if (is_pub) try g.tables[mod].pub_fns.put(g.gpa, name, id);
                },
                .impl_decl, .impl_has_decl => {
                    // Each method is an ordinary global fn with a MANGLED name
                    // `<module.path>.<Receiver>.<method>` (so it gets a global id +
                    // codegen unit) but is NOT bare-callable (never inserted into
                    // `tables[mod].fns`) — a method is reached only via `x.m(..)`
                    // dispatch through the program-wide method table (built in Pass A).
                    // A conformance method (`impl_has_decl`) DISPATCHES identically — its
                    // protocol-method dispatch is indistinguishable from an inherent one
                    // — but its symbol carries a protocol suffix (see below).
                    // `Ast.implMethods` yields the method run for either shape.
                    const recv_name = g.nameOf(mod, decl.main_token);
                    // A conformance method (`impl P has Q`) mangles its symbol +
                    // duplicate-check key with the protocol identity (its name, plus any
                    // type-args), so two conformances sharing a method name on one receiver
                    // don't collide on the `<recv>.<method>` symbol (a linker duplicate) nor
                    // trip a spurious R0002 that would pre-empt the T0025 multi-conformance
                    // resolver. Generic protocols additionally distinguish `Into[int]` from
                    // `Into[bool]` by their type-args. An inherent impl (`impl P`) keeps the
                    // bare `<recv>.<method>` name.
                    var proto_suffix: []const u8 = "";
                    defer if (proto_suffix.len > 0) g.gpa.free(proto_suffix);
                    if (decl.tag == .impl_has_decl) {
                        if (Ast.implProtocol(t, decl)) |proto_ref| {
                            var sb: std.ArrayList(u8) = .empty;
                            errdefer sb.deinit(g.gpa);
                            try sb.append(g.gpa, '$');
                            try sb.appendSlice(g.gpa, g.nameOf(mod, m.nodes[Ast.protocolRefBase(t, proto_ref).int()].main_token));
                            for (Ast.protocolRefArgs(t, proto_ref)) |an| {
                                try sb.append(g.gpa, '$');
                                try sb.appendSlice(g.gpa, g.nameOf(mod, m.nodes[an.int()].main_token));
                            }
                            proto_suffix = try sb.toOwnedSlice(g.gpa);
                        }
                    }
                    for (Ast.implMethods(t, decl)) |method_idx| {
                        const method = m.nodes[method_idx.int()];
                        if (method.tag != .fn_decl) continue;
                        const mname = g.nameOf(mod, method.main_token);
                        const qname = try std.fmt.allocPrint(g.gpa, "{s}.{s}.{s}{s}", .{ m.path, recv_name, mname, proto_suffix });
                        const gop = try method_names.getOrPut(g.gpa, qname);
                        if (gop.found_existing) {
                            try g.emit(.R0002, mod, m.tokens[method.main_token].start, "duplicate method '{s}.{s}'", .{ recv_name, mname });
                            g.gpa.free(qname);
                            continue;
                        }
                        try g.fns.append(g.gpa, .{
                            .name = qname,
                            .module = mod,
                            .decl_node = method_idx,
                            .kind = .user_fn,
                            .is_pub = false,
                            // `decl.lhs` is the receiver type-ref: a bare `identifier`,
                            // a generic `impl Box[T]` `type_app`, or a
                            // qualified `impl mod.T` `field_access` —
                            // `decodeFnSig` decodes any of them to the self type.
                            .recv_type = decl.lhs,
                        });
                    }
                },
                // A `protocol_decl`'s bodyless method sigs must NEVER enter the fn table
                // (no global id / codegen / bare-callability): the `else` ignores it.
                else => {},
            }
        }
    }

    // Seed every synthetic builtin off one ordered list (`seeded_builtins`). Each is a
    // homeless `.builtin` (bodyless; the ones that need machine code are hand-emitted at
    // link time). `panic` is global prelude (registered into every module's fn table
    // unless shadowed); the `Intrinsic` names are core-only (registered only into a
    // bundled `core/` module — naming them elsewhere leaves the name unresolved, R0001).
    // The list's order fixes each id, so `panic` leads and no later append shifts an
    // existing id.
    for (seeded_builtins) |b| try seedBuiltin(g, entry, b);
    // The prelude enums (`Ordering`; generic value enums `Option`/`Result`) are nameable
    // in every module with no import (the `panic` precedent). Register each into a module's enum
    // table UNLESS the module declares its own (user-first-wins), so their construction/match
    // resolve quietly (left for Typecheck, which native-registers the enums). The typecheck-time
    // `enum_ids` injection in `registerPrelude` mirrors this on its own tables.
    for (g.tables) |*t| {
        for (prelude_enum_names) |name| {
            if (!t.enums.contains(name)) try t.enums.put(g.gpa, name, true);
        }
    }
}

/// Bind each module's import namespaces (last path segment, or the `as` alias) to
/// the imported module id. A same-name collision (two imports sharing a last
/// segment, or an alias colliding with another namespace) is an error unless
/// disambiguated by `as`.
fn collectNamespaces(g: *GraphResolve) !void {
    for (g.graph.modules, 0..) |m, mi| {
        const mod: u32 = @intCast(mi);
        if (m.nodes.len == 0) continue;
        const t = g.tree(mod);
        const prog = m.nodes[Ast.root(m.nodes).int()];
        if (prog.tag != .program) continue;

        for (Ast.rangeSlice(t, prog.lhs.int())) |decl_idx| {
            const decl = m.nodes[decl_idx.int()];
            if (decl.tag != .import_decl) continue;

            // Resolve which graph module this import points at.
            const target = g.importTarget(mod, decl) orelse {
                // Discovery already proved every import resolves; a miss here is
                // defensive (e.g. a stale graph). Report against this module.
                try g.emit(.R0003, mod, m.tokens[decl.main_token].start, "unknown imported module", .{});
                continue;
            };

            // Namespace name = alias token if present, else the last path segment.
            const ns_tok: u32 = if (Ast.importAliasTok(decl).unwrap()) |alias| alias.int() else decl.main_token;
            const ns_name = g.nameOf(mod, ns_tok);

            // A namespace must not collide with a top-level declaration in the
            // SAME module (design point 3: innermost binding wins, but a top-level
            // fn/type and an import sharing a name is an ambiguity, not a shadow —
            // there is no inner scope to disambiguate them). Diagnose it here at
            // collect time, against the import token, rather than letting the fn
            // silently win and leaving the qualified call `.unresolved` → a codegen
            // crash with no user diagnostic.
            if (g.tables[mod].fns.contains(ns_name) or
                g.tables[mod].structs.contains(ns_name) or
                g.tables[mod].enums.contains(ns_name))
            {
                try g.emit(.R0004, mod, m.tokens[ns_tok].start, "import namespace '{s}' collides with a top-level declaration named '{s}' (use `as` to disambiguate)", .{ ns_name, ns_name });
                continue;
            }

            const gop = try g.tables[mod].namespaces.getOrPut(g.gpa, ns_name);
            if (gop.found_existing) {
                try g.emit(.R0004, mod, m.tokens[ns_tok].start, "import namespace '{s}' collides with an earlier import (use `as` to disambiguate)", .{ns_name});
                continue;
            }
            gop.value_ptr.* = target;
        }
    }
}

/// Make a `pub` struct/enum in a DIRECTLY imported module nameable UNQUALIFIED in the
/// importer (the `std/vec` `Vec` surface), mirroring the prelude Option/Result injection
/// above. Copies each import target's OWN pub type names into the importer's bare
/// struct/enum tables if-absent, marked NON-pub so the injected name is not transitively
/// re-exported to a module importing this one. Runs after namespaces are bound and every
/// module's own decls are registered, before body resolution. A local/prelude name always
/// wins (if-absent), so the corpus — which references imports qualified — is unperturbed.
fn injectImportedTypes(g: *GraphResolve) !void {
    for (g.tables) |*t| {
        var nit = t.namespaces.valueIterator();
        while (nit.next()) |target_ptr| {
            const target = target_ptr.*;
            var sit = g.tables[target].structs.iterator();
            while (sit.next()) |se| {
                if (!se.value_ptr.*) continue; // pub only (own decls)
                if (!t.structs.contains(se.key_ptr.*))
                    try t.structs.put(g.gpa, se.key_ptr.*, false);
            }
            var eit = g.tables[target].enums.iterator();
            while (eit.next()) |ee| {
                if (!ee.value_ptr.*) continue;
                if (!t.enums.contains(ee.key_ptr.*))
                    try t.enums.put(g.gpa, ee.key_ptr.*, false);
            }
        }
    }
}

/// Find the graph module id an `import_decl` in module `mod` refers to, by
/// rebuilding its `/`-joined path and matching a module's canonical name.
fn importTarget(g: *GraphResolve, mod: u32, decl: Ast.Node) ?u32 {
    const m = &g.graph.modules[mod];
    const t = g.tree(mod);
    const segs = Ast.importPathToks(t, decl);
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
    for (g.graph.modules, 0..) |om, oi| {
        if (std.mem.eql(u8, om.path, path)) return @intCast(oi);
    }
    return null;
}

fn resolveModule(g: *GraphResolve, mod: u32) !void {
    g.cur_mod = mod;
    const m = &g.graph.modules[mod];
    if (m.nodes.len == 0) return;
    const t = g.tree(mod);
    const prog = m.nodes[Ast.root(m.nodes).int()];
    if (prog.tag != .program) return;

    for (Ast.rangeSlice(t, prog.lhs.int())) |decl_idx| {
        switch (m.nodes[decl_idx.int()].tag) {
            .fn_decl => try g.resolveFn(decl_idx),
            // A struct's field type-refs may name a qualified cross-module type
            // (`box: rect.Rect`). Resolve those for namespace binding + pub-type
            // visibility (mirrors the param/return type-ref check below).
            .struct_decl => {
                const decl = m.nodes[decl_idx.int()];
                for (Ast.rangeSlice(t, decl.lhs.int())) |field_idx| {
                    const field = m.nodes[field_idx.int()];
                    try g.resolveTypeRef(field.lhs);
                }
            },
            .tuple_struct_decl => {
                const decl = m.nodes[decl_idx.int()];
                for (Ast.rangeSlice(t, decl.lhs.int())) |type_idx| {
                    try g.resolveTypeRef(type_idx);
                }
            },
            // An alias target may name a qualified cross-module type (`type S = geom.Rect`);
            // resolve it here so the R0005 pub-visibility check fires, exactly as for a
            // struct/tuple-struct field type-ref (else an alias launders a non-`pub` type).
            .type_alias_decl => try g.resolveTypeRef(m.nodes[decl_idx.int()].lhs),
            // Resolve each method body like any fn: its synthesized `self` param
            // binds as a local (slot 0) via the ordinary param loop in `resolveFn`.
            // A conformance impl (`impl_has_decl`) resolves identically; its qualified
            // self-param type-ref binds the module namespace via `resolveTypeRef`.
            .impl_decl, .impl_has_decl => {
                const decl = m.nodes[decl_idx.int()];
                for (Ast.implMethods(t, decl)) |method_idx| {
                    if (m.nodes[method_idx.int()].tag == .fn_decl) try g.resolveFn(method_idx);
                }
            },
            // A `protocol_decl`'s bodyless sigs have no bodies to resolve.
            else => {},
        }
    }
}

fn resolveFn(g: *GraphResolve, fn_idx: Ast.Index) error{OutOfMemory}!void {
    for (g.scopes.items) |*s| s.names.deinit(g.gpa);
    g.scopes.clearRetainingCapacity();
    g.locals.clearRetainingCapacity();
    g.slot_next = 0;
    g.label_stack.clearRetainingCapacity();

    const decl = g.nodes()[fn_idx.int()];
    const proto = Ast.protoAt(g.tree(g.cur_mod), decl.lhs.int());

    try g.pushScope();
    for (proto.params) |param_idx| {
        const param = g.nodes()[param_idx.int()];
        _ = try g.declare(param.main_token, "duplicate parameter '{s}'", .W0002);
        // Resolve the param's type-ref for a qualified cross-module type.
        try g.resolveTypeRef(param.lhs);
    }
    try g.resolveTypeRef(proto.ret_type);
    try g.resolveBlock(decl.rhs);
    g.popScope();

    // `g.locals` is per-fn (cleared at entry) and appended once at each `declare`, so it
    // is in source-declaration order here — deterministic without touching the
    // nondeterministic scope name maps. Each binder carries the code to emit if unused
    // (`.none` opts out); a leading `_` silences. `self` is the conventional method
    // receiver: never warn on it (an unusable `_self` is not an acceptable fix), but
    // gate that skip to W0002 so an ordinary `self := ..` local still warns as before.
    for (g.locals.items) |loc| {
        if (loc.warn_code == .none or loc.used) continue;
        const nm = g.nameText(loc.name_tok);
        if (nm.len != 0 and nm[0] == '_') continue;
        if (loc.warn_code == .W0002 and std.mem.eql(u8, nm, "self")) continue;
        const off = g.tokens()[loc.name_tok].start;
        switch (loc.warn_code) {
            .W0001 => try g.emit(.W0001, g.cur_mod, off, "unused variable '{s}'; prefix with '_' (as '_{s}') to silence", .{ nm, nm }),
            .W0002 => try g.emit(.W0002, g.cur_mod, off, "unused parameter '{s}'; prefix with '_' (as '_{s}') to silence", .{ nm, nm }),
            else => unreachable,
        }
    }
}

/// Resolve a TYPE-REF node (a param/return/field type annotation). A bare
/// `identifier` type-name (`int`, `Rect`) is left for Typecheck. A qualified
/// `mod.Type` parses as a `field_access` with a module-namespace receiver: bind
/// the receiver to `.module` and enforce pub-type visibility (a non-pub or
/// unknown type member is an error rendered against THIS module). The owning
/// module's actual layout id is bound later by Typecheck/`typeRefToType`.
fn resolveTypeRef(g: *GraphResolve, type_idx: Ast.Index) error{OutOfMemory}!void {
    if (type_idx == Ast.none) return;
    const n = g.nodes()[type_idx.int()];
    if (n.tag != .field_access) return; // bare name / unit: nothing to bind here
    const recv = g.nodes()[n.lhs.int()];
    if (recv.tag != .identifier) return;
    const recv_name = g.nameText(recv.main_token);
    // A type-ref receiver is never a local (no value scope around a type), but a
    // module-named local could still shadow per the scope rules; honour them.
    if (g.lookupLocalOrFn(recv.main_token) != null) return;
    const target_mod = g.tables[g.cur_mod].namespaces.get(recv_name) orelse return;
    g.res(n.lhs, .{ .module = target_mod });
    // The member is a TYPE name; check it is an exported struct/enum.
    const member = g.nameText(n.main_token);
    const tt = &g.tables[target_mod];
    if (tt.structs.get(member)) |is_pub| {
        if (!is_pub) try g.emit(.R0005, g.cur_mod, g.tokens()[n.main_token].start, "struct '{s}' is not exported by module '{s}'", .{ member, g.graph.modules[target_mod].path });
        return;
    }
    if (tt.enums.get(member)) |is_pub| {
        if (!is_pub) try g.emit(.R0005, g.cur_mod, g.tokens()[n.main_token].start, "enum '{s}' is not exported by module '{s}'", .{ member, g.graph.modules[target_mod].path });
        return;
    }
    try g.emit(.R0006, g.cur_mod, g.tokens()[n.main_token].start, "module '{s}' has no type '{s}'", .{ g.graph.modules[target_mod].path, member });
}

fn resolveBlock(g: *GraphResolve, block_idx: Ast.Index) error{OutOfMemory}!void {
    const block = g.nodes()[block_idx.int()];
    try g.pushScope();
    for (Ast.rangeSlice(g.tree(g.cur_mod), block.lhs.int())) |stmt_idx| {
        try g.resolveStmt(stmt_idx);
    }
    g.popScope();
}

fn resolveStmt(g: *GraphResolve, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = g.nodes()[stmt_idx.int()];
    switch (stmt.tag) {
        .var_decl => {
            try g.resolveExpr(stmt.lhs);
            const slot = try g.declare(stmt.main_token, "redeclaration of '{s}'", .W0001);
            if (slot) |s| g.res(stmt_idx, .{ .local = s });
        },
        .assign => {
            const target = g.nodes()[stmt.lhs.int()];
            // A `.`-rooted place (`p.x`), or a `*r` deref place, resolves as an
            // expression (its root name is checked there), not as a bare assignable name.
            if (target.tag == .field_access or target.tag == .tuple_field or target.tag == .unary) {
                try g.resolveExpr(stmt.lhs);
            } else {
                const resn = g.lookupName(target.main_token);
                g.res(stmt.lhs, resn);
                switch (resn) {
                    .unresolved => try g.emit(.R0007, g.cur_mod, g.tokens()[target.main_token].start, "assignment to undeclared name '{s}'; declare it first with ':='", .{g.nameText(target.main_token)}),
                    .func => try g.emit(.R0007, g.cur_mod, g.tokens()[target.main_token].start, "cannot assign to function '{s}'", .{g.nameText(target.main_token)}),
                    .module => try g.emit(.R0007, g.cur_mod, g.tokens()[target.main_token].start, "cannot assign to module '{s}'", .{g.nameText(target.main_token)}),
                    .local, .label => {},
                }
            }
            try g.resolveExpr(stmt.rhs);
        },
        .return_stmt => if (stmt.lhs != Ast.none) try g.resolveExpr(stmt.lhs),
        .expr_stmt => try g.resolveExpr(stmt.lhs),
        .block => try g.resolveBlock(stmt_idx),
        .if_stmt => {
            try g.resolveExpr(stmt.lhs);
            const h = Ast.ifHeaderAt(g.tree(g.cur_mod), stmt.rhs.int());
            try g.resolveBlock(h.then_block);
            if (h.else_node != Ast.none) {
                if (g.nodes()[h.else_node.int()].tag == .if_stmt)
                    try g.resolveStmt(h.else_node)
                else
                    try g.resolveBlock(h.else_node);
            }
        },
        .while_stmt => {
            try g.resolveExpr(stmt.lhs);
            try g.resolveBlock(stmt.rhs);
        },
        .for_stmt => {
            const h = Ast.forHeaderAt(g.tree(g.cur_mod), stmt.rhs.int());
            try g.resolveExpr(h.lo);
            try g.resolveExpr(h.hi);
            try g.pushScope();
            const slot = try g.declare(stmt.main_token, "redeclaration of '{s}'", .W0001);
            if (slot) |s| g.res(stmt_idx, .{ .local = s });
            try g.resolveBlock(stmt.lhs);
            g.popScope();
        },
        .for_in_stmt => {
            try g.resolveExpr(stmt.rhs);
            try g.pushScope();
            const slot = try g.declare(stmt.main_token, "redeclaration of '{s}'", .W0001);
            if (slot) |s| g.res(stmt_idx, .{ .local = s });
            try g.resolveBlock(stmt.lhs);
            g.popScope();
        },
        .for_in2_stmt => {
            const h = Ast.forIn2HeaderAt(g.tree(g.cur_mod), stmt.lhs.int());
            try g.resolveExpr(stmt.rhs);
            try g.pushScope();
            const ks = try g.declare(stmt.main_token, "redeclaration of '{s}'", .W0001);
            if (ks) |s| g.res(stmt_idx, .{ .local = s });
            const vtok = g.nodes()[h.val_leaf.int()].main_token;
            const vs = try g.declare(vtok, "redeclaration of '{s}'", .W0001);
            if (vs) |s| g.res(h.val_leaf, .{ .local = s });
            try g.resolveBlock(h.body);
            g.popScope();
        },
        .break_stmt => {
            if (Ast.labelTok(stmt).unwrap()) |label| try g.resolveLabelTarget(stmt_idx, label.int(), "break");
            if (stmt.lhs != Ast.none) try g.resolveExpr(stmt.lhs);
        },
        .continue_stmt => {
            if (Ast.labelTok(stmt).unwrap()) |label| try g.resolveLabelTarget(stmt_idx, label.int(), "continue");
        },
        .labeled => try g.resolveLabeled(stmt_idx),
        // A poison leaf as a statement: already-diagnosed, resolve nothing.
        .error_node => {},
        else => try g.resolveExpr(stmt_idx),
    }
}

fn resolveExpr(g: *GraphResolve, node_idx: Ast.Index) error{OutOfMemory}!void {
    if (node_idx == Ast.none) return;
    const n = g.nodes()[node_idx.int()];
    switch (n.tag) {
        .identifier => {
            const resn = g.lookupName(n.main_token);
            g.res(node_idx, resn);
            if (resn == .unresolved) {
                if (g.tables[g.cur_mod].structs.contains(g.nameText(n.main_token))) return;
                if (g.tables[g.cur_mod].enums.contains(g.nameText(n.main_token))) return;
                const name = g.nameText(n.main_token);
                const off = g.tokens()[n.main_token].start;
                // An EXACT stdlib name (checked before the fuzzy suggester) means the
                // import is missing, not that the name is a typo — name the module to
                // import. Only an exact match hints, so a genuine typo still falls
                // through to near-miss below; a user struct/enum of the same name has
                // already returned above, so it is never overridden here.
                if (StdNames.importHintFor(name)) |hint| {
                    if (hint.spelling) |sp|
                        try g.emit(.R0001, g.cur_mod, off, "undeclared identifier '{s}'; use '{s}' and add 'import std/{s}'", .{ name, sp, hint.module })
                    else
                        try g.emit(.R0001, g.cur_mod, off, "undeclared identifier '{s}'; add 'import std/{s}'", .{ name, hint.module });
                }
                // If an in-scope name is a close typo of the undeclared one, append a
                // "did you mean" hint (message-embedded — no note channel yet). The
                // suggester is conservative (short names / distant names / ties → no
                // hint), so this stays byte-identical for the existing no-hint cases.
                else if (nearmiss.suggest(name, g.candidateIter())) |cand|
                    try g.emit(.R0001, g.cur_mod, off, "undeclared identifier '{s}'; did you mean '{s}'?", .{ name, cand })
                else
                    try g.emit(.R0001, g.cur_mod, off, "undeclared identifier '{s}'", .{name});
            }
        },
        .literal_number, .literal_float, .literal_string, .literal_bool, .literal_char => {},
        // A poison leaf is already-diagnosed: no name lookup, no diagnostic.
        .error_node => {},
        .unary => try g.resolveExpr(n.lhs),
        .binary => {
            try g.resolveExpr(n.lhs);
            try g.resolveExpr(n.rhs);
        },
        .call => {
            // A type_app DIRECTLY in callee position is a turbofish call — `id[int](..)`
            // or a method turbofish `v.into[int](..)` — whose bracket holds TYPE args, not
            // a value subscript. Resolve only the base callee; never descend the type-args
            // (resolving `int` would fire R0001). Going through the generic `.type_app` arm
            // would misread `v.into[int]` as a value index (leftmost `v` is `.local`) and
            // descend `int`. The language has no first-class function values, so a value
            // index in callee position (`xs[i]()`) is never a valid call anyway — leaving
            // its subscript's names unbound here is harmless (a later stage rejects the
            // call), so this base-only resolution is safe for both readings.
            const callee = g.nodes()[n.lhs.int()];
            if (callee.tag == .type_app)
                try g.resolveExpr(callee.lhs)
            else
                try g.resolveExpr(n.lhs);
            for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |arg| try g.resolveExpr(arg);
        },
        .struct_init => {
            for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |fi| try g.resolveExpr(g.nodes()[fi.int()].lhs);
        },
        .field_access => try g.resolveFieldAccess(node_idx, n),
        .tuple_field => try g.resolveExpr(n.lhs),
        // A type-application callee `id[int](..)`: resolve ONLY the base callee
        // (`n.lhs`). The type-arg Range is NOT descended — resolving a type name
        // like `int` would fire R0001 and stop the pipeline at resolve, pre-empting
        // the T0013 generics gate at typecheck. Forward-safety.
        //
        // EXCEPT when the head chain's LEFTMOST identifier resolves to a `.local`:
        // then this is a value index `b.items[i]` the parser could not tell from a
        // turbofish, and its bracket contents are an ordinary expression (`i`) whose
        // names must bind. A real type-app's leftmost is a struct/enum name
        // (.unresolved), a generic fn (.func), or a module (.module) — never `.local`
        // — so its type-arg range stays undescended and R0001 on `int` never fires.
        // `resolveExpr(n.lhs)` runs FIRST so the leftmost ident is bound before the
        // predicate reads it.
        .type_app => {
            try g.resolveExpr(n.lhs);
            if (symbols_res.leftmostHeadIsValue(g.tree(g.cur_mod), g.resolutions[g.cur_mod], n.lhs))
                for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |a| try g.resolveExpr(a);
        },
        // Postfix `?`: resolve the operand's names; the `?` itself binds nothing.
        .try_expr => try g.resolveExpr(n.lhs),
        // A list literal's elements and a value index's receiver + index are ordinary
        // expressions whose names must resolve (the silent `else` would skip them).
        .list_literal => for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |el| try g.resolveExpr(el),
        .index => {
            try g.resolveExpr(n.lhs);
            try g.resolveExpr(n.rhs);
        },
        .enum_init_unit => {},
        .enum_init_tuple => for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |a| try g.resolveExpr(a),
        .enum_init_struct => for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |fi| try g.resolveExpr(g.nodes()[fi.int()].lhs),
        .match_expr => try g.resolveMatch(node_idx),
        .literal_unit => {},
        .block => try g.resolveBlock(node_idx),
        // `unsafe { .. }`: resolve the names in its inner block (the silent `else`
        // would otherwise skip them). The unsafe context itself binds nothing here.
        .unsafe_block => try g.resolveBlock(n.lhs),
        .labeled => try g.resolveLabeled(node_idx),
        .loop_expr => try g.resolveBlock(n.lhs),
        .if_stmt => {
            try g.resolveExpr(n.lhs);
            const h = Ast.ifHeaderAt(g.tree(g.cur_mod), n.rhs.int());
            try g.resolveBlock(h.then_block);
            if (h.else_node != Ast.none) {
                if (g.nodes()[h.else_node.int()].tag == .if_stmt)
                    try g.resolveExpr(h.else_node)
                else
                    try g.resolveBlock(h.else_node);
            }
        },
        else => {},
    }
}

/// Resolve a `field_access` node. The `.` 4th meaning: if the receiver is a bare
/// identifier that binds to an import namespace, this is `mod.member` — bind the
/// receiver to `.module` and resolve the member against the owning module's PUB
/// tables (a pub fn → `.func` global id ON THIS field_access node; a pub type →
/// quiet for Typecheck; a non-pub/missing member → error). Otherwise it is an
/// ordinary value field access (or a same-module qualified enum `N.V`) and only
/// the receiver chain is resolved, exactly as the single-file resolver does.
fn resolveFieldAccess(g: *GraphResolve, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!void {
    const recv = g.nodes()[n.lhs.int()];
    if (recv.tag == .identifier) {
        // Innermost lexical binding shadows a namespace: only treat the receiver
        // as a module if the name is NOT a local/fn in scope and IS a namespace.
        const recv_name = g.nameText(recv.main_token);
        const shadowed = g.lookupLocalOrFn(recv.main_token) != null;
        if (!shadowed) {
            if (g.tables[g.cur_mod].namespaces.get(recv_name)) |target_mod| {
                g.res(n.lhs, .{ .module = target_mod });
                try g.resolveModuleMember(node_idx, n, target_mod);
                return;
            }
        }
    }
    // Ordinary receiver chain (value field access, or same-module `N.V`).
    try g.resolveExpr(n.lhs);
}

/// Resolve the member tail of a `mod.member` access against module `target`'s pub
/// tables. A pub fn binds `.func` (global id) onto the field_access node; a pub
/// struct/enum stays quiet (Typecheck resolves the qualified type / 3-level
/// `mod.Enum.Variant` against that module's tables); anything non-pub or unknown
/// is a visibility/undeclared error rendered against THIS module.
fn resolveModuleMember(g: *GraphResolve, node_idx: Ast.Index, n: Ast.Node, target: u32) error{OutOfMemory}!void {
    const member = g.nameText(n.main_token);
    const tt = &g.tables[target];
    if (tt.pub_fns.get(member)) |gid| {
        // Quarantine: a pub `.import` (extern) member is served only to a `core/*`
        // module (so `core/sys.print` reaches `write`, but `std/math` never names an
        // extern — it calls a safe `core/ffi.abs` wrapper). A std/user module doing
        // `import core/ffi; ffi.labs(..)` falls through to the R0005 not-exported branch
        // below, closing the bypass.
        if (!(g.fns.items[gid].kind == .import and !g.refCore())) {
            g.res(node_idx, .{ .func = gid });
            return;
        }
    }
    // A pub type member is quiet (the qualified type / enum-variant tail is bound
    // by Typecheck). For a 3-level `mod.Enum.Variant`, this node is the inner
    // `mod.Enum` field_access — leave it unresolved; the outer field_access keeps
    // its receiver chain resolved.
    if (tt.structs.get(member)) |is_pub| {
        if (is_pub) return;
        try g.emit(.R0005, g.cur_mod, g.tokens()[n.main_token].start, "struct '{s}' is not exported by module '{s}'", .{ member, g.graph.modules[target].path });
        return;
    }
    if (tt.enums.get(member)) |is_pub| {
        if (is_pub) return;
        try g.emit(.R0005, g.cur_mod, g.tokens()[n.main_token].start, "enum '{s}' is not exported by module '{s}'", .{ member, g.graph.modules[target].path });
        return;
    }
    // A private fn referenced cross-module: distinguish "exists but not pub" from
    // "no such member" for a clearer message.
    if (tt.fns.contains(member)) {
        try g.emit(.R0005, g.cur_mod, g.tokens()[n.main_token].start, "function '{s}' is not exported by module '{s}'", .{ member, g.graph.modules[target].path });
    } else {
        try g.emit(.R0006, g.cur_mod, g.tokens()[n.main_token].start, "module '{s}' has no member '{s}'", .{ g.graph.modules[target].path, member });
    }
}

fn resolveLabeled(g: *GraphResolve, idx: Ast.Index) error{OutOfMemory}!void {
    const n = g.nodes()[idx.int()];
    const name = g.nameText(n.main_token);
    for (g.label_stack.items) |e| {
        if (std.mem.eql(u8, e.name, name)) {
            try g.emit(.R0008, g.cur_mod, g.tokens()[n.main_token].start, "duplicate label '{s}'", .{name});
            break;
        }
    }
    try g.label_stack.append(g.gpa, .{ .name = name, .construct_node = n.lhs });
    const inner = g.nodes()[n.lhs.int()];
    switch (inner.tag) {
        .block => try g.resolveBlock(n.lhs),
        .loop_expr, .while_stmt, .for_stmt, .for_in_stmt, .for_in2_stmt => try g.resolveStmt(n.lhs),
        else => {},
    }
    _ = g.label_stack.pop();
}

fn resolveMatch(g: *GraphResolve, node_idx: Ast.Index) error{OutOfMemory}!void {
    const n = g.nodes()[node_idx.int()];
    try g.resolveExpr(n.lhs);
    for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |arm_idx| {
        const arm = g.nodes()[arm_idx.int()];
        try g.pushScope();
        try g.declarePattern(arm.lhs);
        const h = Ast.armHeaderAt(g.tree(g.cur_mod), arm.rhs.int());
        if (h.guard != Ast.none) try g.resolveExpr(h.guard);
        try g.resolveExpr(h.body);
        g.popScope();
    }
}

fn declarePattern(g: *GraphResolve, pat_idx: Ast.Index) error{OutOfMemory}!void {
    if (pat_idx == Ast.none) return;
    const pat = g.nodes()[pat_idx.int()];
    switch (pat.tag) {
        .pattern_wildcard, .pattern_literal => {},
        .pattern_binding => {
            const slot = try g.declare(pat.main_token, "redeclaration of '{s}'", .none);
            if (slot) |s| g.res(pat_idx, .{ .local = s });
            if (pat.rhs != Ast.none) try g.declarePattern(pat.rhs);
        },
        .pattern_variant => {
            if (pat.rhs != Ast.none)
                for (Ast.rangeSlice(g.tree(g.cur_mod), pat.rhs.int())) |c| try g.declarePattern(c);
        },
        .pattern_or => {
            const alts = Ast.rangeSlice(g.tree(g.cur_mod), pat.lhs.int());
            if (alts.len == 0) return;
            try g.declarePattern(alts[0]);
            for (alts[1..]) |alt| try g.bindOrAltToFirst(alt);
        },
        else => {},
    }
}

fn bindOrAltToFirst(g: *GraphResolve, pat_idx: Ast.Index) error{OutOfMemory}!void {
    if (pat_idx == Ast.none) return;
    const pat = g.nodes()[pat_idx.int()];
    switch (pat.tag) {
        .pattern_wildcard, .pattern_literal => {},
        .pattern_binding => {
            const existing = g.lookupName(pat.main_token);
            if (existing == .local) {
                g.res(pat_idx, existing);
            } else {
                const slot = try g.declare(pat.main_token, "redeclaration of '{s}'", .none);
                if (slot) |s| g.res(pat_idx, .{ .local = s });
            }
            if (pat.rhs != Ast.none) try g.bindOrAltToFirst(pat.rhs);
        },
        .pattern_variant => {
            if (pat.rhs != Ast.none)
                for (Ast.rangeSlice(g.tree(g.cur_mod), pat.rhs.int())) |c| try g.bindOrAltToFirst(c);
        },
        .pattern_or => {
            for (Ast.rangeSlice(g.tree(g.cur_mod), pat.lhs.int())) |alt| try g.bindOrAltToFirst(alt);
        },
        else => {},
    }
}

fn resolveLabelTarget(g: *GraphResolve, node_idx: Ast.Index, label_tok: u32, comptime verb: []const u8) error{OutOfMemory}!void {
    const name = g.nameText(label_tok);
    var i = g.label_stack.items.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, g.label_stack.items[i].name, name)) {
            g.res(node_idx, .{ .label = g.label_stack.items[i].construct_node });
            return;
        }
    }
    try g.emit(.R0009, g.cur_mod, g.tokens()[label_tok].start, verb ++ " to undefined label '{s}'", .{name});
}

fn res(g: *GraphResolve, node_idx: Ast.Index, r: Resolution) void {
    g.resolutions[g.cur_mod][node_idx.int()] = r;
}

fn pushScope(g: *GraphResolve) !void {
    try g.scopes.append(g.gpa, .{});
}

fn popScope(g: *GraphResolve) void {
    var s = g.scopes.pop().?;
    s.names.deinit(g.gpa);
}

fn declare(g: *GraphResolve, name_tok: u32, comptime dup_fmt: []const u8, comptime warn_code: codes.Code) !?u32 {
    const name = g.nameText(name_tok);
    const scope = &g.scopes.items[g.scopes.items.len - 1];
    const gop = try scope.names.getOrPut(g.gpa, name);
    if (gop.found_existing) {
        try g.emit(.none, g.cur_mod, g.tokens()[name_tok].start, dup_fmt, .{name});
        return null;
    }
    const slot = g.slot_next;
    g.slot_next += 1;
    const local_idx: u32 = @intCast(g.locals.items.len);
    try g.locals.append(g.gpa, .{ .name_tok = name_tok, .slot = slot, .warn_code = warn_code });
    gop.value_ptr.* = local_idx;
    return slot;
}

/// Whether a name resolves to a local or this module's fn (used to decide if a
/// receiver identifier shadows an import namespace).
fn lookupLocalOrFn(g: *GraphResolve, name_tok: u32) ?Resolution {
    const name = g.nameText(name_tok);
    var i = g.scopes.items.len;
    while (i > 0) {
        i -= 1;
        if (g.scopes.items[i].names.get(name)) |local_idx| {
            g.locals.items[local_idx].used = true;
            return .{ .local = g.locals.items[local_idx].slot };
        }
    }
    if (g.tables[g.cur_mod].fns.get(name)) |gid| {
        // An `.import` (extern) symbol resolves only in a `core/*` module; anywhere
        // else it falls through to the undeclared diagnostic. In practice only a
        // `core/` module holds an import symbol in its own `fns` (the DECL gate is
        // core-only), but this keeps the quarantine invariant explicit at the bare-name
        // site too.
        if (g.fns.items[gid].kind == .import and !g.refCore()) return null;
        return .{ .func = gid };
    }
    return null;
}

/// Whether the CURRENTLY-RESOLVING module is under `core/`. An `.import` (extern)
/// name resolves only when the referring module is `core/`; `std` reaches C only
/// through a safe core wrapper, so a `std`/user module naming an extern gets the
/// standard not-exported/unresolved diagnostic.
fn refCore(g: *GraphResolve) bool {
    return g.graph.modules[g.cur_mod].isCore();
}

/// An iterator over exactly the names `lookupName` searches, for near-miss
/// suggestions: every lexical scope's local names (innermost scope first), then this
/// module's fn names (incl. seeded builtins), then this module's import namespace
/// names. Fixed traversal order; the suggester's strict-unique-winner rule makes the
/// emitted string independent of the per-map hash iteration order.
const CandidateIter = struct {
    g: *GraphResolve,
    /// Phase: 0 = scopes (walking from innermost outward), 1 = fns, 2 = namespaces.
    phase: u8 = 0,
    scope_i: usize, // index into g.scopes, walked downward from the end
    map_it: ?std.StringHashMapUnmanaged(u32).KeyIterator = null,

    pub fn next(self: *CandidateIter) ?[]const u8 {
        while (true) {
            if (self.map_it) |*it| {
                if (it.next()) |k| return k.*;
                self.map_it = null;
            }
            switch (self.phase) {
                0 => {
                    if (self.scope_i == 0) {
                        self.phase = 1;
                        self.map_it = self.g.tables[self.g.cur_mod].fns.keyIterator();
                        continue;
                    }
                    self.scope_i -= 1;
                    self.map_it = self.g.scopes.items[self.scope_i].names.keyIterator();
                },
                1 => {
                    self.phase = 2;
                    self.map_it = self.g.tables[self.g.cur_mod].namespaces.keyIterator();
                },
                else => return null,
            }
        }
    }
};

fn candidateIter(g: *GraphResolve) CandidateIter {
    return .{ .g = g, .scope_i = g.scopes.items.len };
}

/// Look a name up: locals (innermost first) → this module's fns (incl. seeded
/// builtins) → import namespaces. A lexical binding shadows a namespace.
fn lookupName(g: *GraphResolve, name_tok: u32) Resolution {
    if (g.lookupLocalOrFn(name_tok)) |r| return r;
    const name = g.nameText(name_tok);
    if (g.tables[g.cur_mod].namespaces.get(name)) |mod| return .{ .module = mod };
    return .unresolved;
}

/// Warn (W0003) on every non-`pub` top-level user fn that no `.func` resolution
/// anywhere in the graph names. The sweep is write-path-agnostic: it reads every
/// module's resolution slots, so a bare call, a qualified `mod.fn`, and a turbofish
/// callee all mark their target used. Over-marking (marking a fn called when it isn't)
/// only ever MISSES a warning, never produces a false positive.
///
/// Methods (`recv_type` set) are dispatched through the program-wide method table,
/// never via a `.func` resolution, so they are excluded structurally — otherwise every
/// uncalled method would false-positive. A recursive-only fn writes a `.func` at its
/// self-call, so it is (correctly) not flagged. Emission walks `fns` in id order for a
/// deterministic sink (resolveGraph is sequential; `-j1` == `-jN`).
fn warnUnusedFns(g: *GraphResolve) !void {
    const called = try g.gpa.alloc(bool, g.fns.items.len);
    defer g.gpa.free(called);
    @memset(called, false);
    for (g.resolutions) |modres| for (modres) |r| switch (r) {
        .func => |gid| called[gid] = true,
        else => {},
    };
    for (g.fns.items, 0..) |f, gid| {
        if (f.kind != .user_fn or f.is_pub or f.recv_type != Ast.none or called[gid]) continue;
        const m = &g.graph.modules[f.module];
        const name_tok = m.nodes[f.decl_node.int()].main_token;
        const name = g.nameOf(f.module, name_tok);
        if (name.len != 0 and name[0] == '_') continue;
        if (f.module == g.graph.entry_index and std.mem.eql(u8, name, "main")) continue;
        try g.emit(.W0003, f.module, m.tokens[name_tok].start, "unused function '{s}'; remove it, make it 'pub', or prefix with '_' to silence", .{name});
    }
}

/// Warn (W0004) on every `import a/b` whose bound namespace is never used in the
/// importing module — neither QUALIFIED (`b.member`, whose receiver is a bare
/// `identifier` node spelling the namespace) nor via a BARE imported type (`Vec` from
/// `import std/vec`, a bare `identifier` node spelling a pub struct/enum the import
/// provides). An imported fn is always called qualified, so only the namespace name and
/// the imported pub TYPES can carry a bare use.
///
/// The bare-type crediting is complete only because rules 3 and 4 enumerate the ONLY
/// two library-type needs the language expresses with no textual type token: a
/// `[..]`/`[]` list literal constructs a `Vec`, and a `for k, v` binds an `Entry`.
/// Every other construct either spells its type (caught by the identifier scan) or needs
/// no import (single-var `for-in`, range-for, `&`/`*`, `?`, str methods, tuples). If a
/// future desugar introduces a new implicit library-type need, a rule MUST be added
/// here or that import silently becomes a false positive.
///
/// A module that declares any `impl` is credited wholesale: its conformances and
/// inherent methods join whole-program coherence, and an importer can depend on one via
/// method-call or operator dispatch that resolves only in typecheck (no textual token
/// this resolver-side pass can key on). Over-approximating an impl-bearing import as used
/// is the safe direction — it only ever misses a warning, never a false positive.
///
/// Diagnostics-only: reads nodes/tokens/`tables`, writes only `g.sink`. Emission walks
/// each module's import decls in SOURCE order; the membership sets are order-independent
/// and `resolveGraph` is sequential, so `-j1` == `-jN`. Over-crediting only ever MISSES
/// a warning; it can never fabricate a false positive.
fn warnUnusedImports(g: *GraphResolve) !void {
    var ref_idents: std.StringHashMapUnmanaged(void) = .empty;
    var own_types: std.StringHashMapUnmanaged(void) = .empty;
    var seen_ns: std.StringHashMapUnmanaged(void) = .empty;
    defer ref_idents.deinit(g.gpa);
    defer own_types.deinit(g.gpa);
    defer seen_ns.deinit(g.gpa);

    // A target's impl decls credit any importer of it (see `importIsUsed`), so the bit
    // is needed for every module as a potential target before the emit loop runs.
    const has_impl = try g.gpa.alloc(bool, g.graph.modules.len);
    defer g.gpa.free(has_impl);
    @memset(has_impl, false);
    for (g.graph.modules, 0..) |m, mi| {
        if (m.nodes.len == 0) continue;
        const t = g.tree(@intCast(mi));
        const prog = m.nodes[Ast.root(m.nodes).int()];
        if (prog.tag != .program) continue;
        for (Ast.rangeSlice(t, prog.lhs.int())) |decl_idx| switch (m.nodes[decl_idx.int()].tag) {
            .impl_decl, .impl_has_decl => {
                has_impl[mi] = true;
                break;
            },
            else => {},
        };
    }

    for (g.graph.modules, 0..) |m, mi| {
        const mod: u32 = @intCast(mi);
        if (m.nodes.len == 0) continue;
        const t = g.tree(mod);
        const prog = m.nodes[Ast.root(m.nodes).int()];
        if (prog.tag != .program) continue;

        ref_idents.clearRetainingCapacity();
        own_types.clearRetainingCapacity();
        seen_ns.clearRetainingCapacity();

        // This module's OWN struct/enum/alias names. A bare reference to one binds the
        // module's own type (own wins the if-absent bare injection), so it must NOT credit
        // an import that happens to provide a same-named pub type.
        for (Ast.rangeSlice(t, prog.lhs.int())) |decl_idx| {
            const decl = m.nodes[decl_idx.int()];
            switch (decl.tag) {
                .struct_decl, .tuple_struct_decl, .enum_decl, .type_alias_decl => try own_types.put(g.gpa, g.nameOf(mod, decl.main_token), {}),
                else => {},
            }
        }

        // One flat, order-independent pass: every referenced bare identifier name, plus
        // whether a list-literal / `for k,v` desugar is present (their implicit
        // `Vec`/`Entry` needs). Import path/alias are TOKENS, not nodes, so an import
        // never self-credits its own namespace name here.
        var has_list = false;
        var has_forkv = false;
        for (m.nodes) |node| switch (node.tag) {
            .identifier => try ref_idents.put(g.gpa, g.nameOf(mod, node.main_token), {}),
            .list_literal, .empty_list => has_list = true,
            .for_in2_stmt => has_forkv = true,
            else => {},
        };

        // Emit in source order. `seen_ns` gives the first import binding a name the win
        // (matching `collectNamespaces`, which binds the first and rejects later
        // collisions); a loser was never bound to this target and already carries R0003/R0004.
        for (Ast.rangeSlice(t, prog.lhs.int())) |decl_idx| {
            const decl = m.nodes[decl_idx.int()];
            if (decl.tag != .import_decl) continue;
            const target = g.importTarget(mod, decl) orelse continue;
            const ns_tok: u32 = if (Ast.importAliasTok(decl).unwrap()) |a| a.int() else decl.main_token;
            const ns_name = g.nameOf(mod, ns_tok);
            const gop = try seen_ns.getOrPut(g.gpa, ns_name);
            if (gop.found_existing) continue;
            const bound = g.tables[mod].namespaces.get(ns_name) orelse continue;
            if (bound != target) continue;
            if (ns_name.len != 0 and ns_name[0] == '_') continue; // deliberate side-effect-only import
            if (g.importIsUsed(target, ns_name, &ref_idents, &own_types, has_list, has_forkv, has_impl[target])) continue;
            try g.emit(.W0004, mod, m.tokens[decl.main_token].start, "unused import '{s}'; remove it", .{g.graph.modules[target].path});
        }
    }
}

/// Whether import `target` (bound as `ns_name`) is referenced in the current module.
/// True iff any of: (1) the namespace name appears bare (every qualified `ns.member`
/// — call, type-ref, turbofish, or an `impl .. has ns.P` protocol ref — has `ns` as a
/// bare `identifier` node); (2) a pub struct/enum NAME the target OWNS appears bare,
/// excluding a name the module declares itself (own wins) and the prelude enums
/// (injected pub into every table, so not specific to any import); (3) a list literal
/// is present and the target owns pub `Vec`; (4) a `for k,v` is present and the target
/// owns pub `Entry`; (5) the target declares any `impl` (`target_has_impl`), whose
/// conformances / inherent methods an importer can reach through method-call or operator
/// dispatch resolved only in typecheck. Every branch only ever CREDITS use, so it can
/// never false-positive.
fn importIsUsed(
    g: *GraphResolve,
    target: u32,
    ns_name: []const u8,
    ref_idents: *const std.StringHashMapUnmanaged(void),
    own_types: *const std.StringHashMapUnmanaged(void),
    has_list: bool,
    has_forkv: bool,
    target_has_impl: bool,
) bool {
    if (ref_idents.contains(ns_name)) return true;
    if (target_has_impl) return true;
    const tt = &g.tables[target];
    var sit = tt.structs.iterator();
    while (sit.next()) |se| {
        if (!se.value_ptr.*) continue; // pub only: the target's OWN exported types
        const name = se.key_ptr.*;
        if (own_types.contains(name)) continue;
        if (ref_idents.contains(name)) return true;
    }
    var eit = tt.enums.iterator();
    while (eit.next()) |ee| {
        if (!ee.value_ptr.*) continue;
        const name = ee.key_ptr.*;
        if (own_types.contains(name)) continue;
        var is_prelude = false;
        for (prelude_enum_names) |p| if (std.mem.eql(u8, name, p)) {
            is_prelude = true;
        };
        if (is_prelude) continue;
        if (ref_idents.contains(name)) return true;
    }
    if (has_list and (tt.structs.get(StdNames.vec_struct) orelse false)) return true;
    if (has_forkv and (tt.structs.get(StdNames.entry_struct) orelse false)) return true;
    return false;
}

/// Stamp the owning module onto the diagnostic (this resolver emits with an
/// explicit `mod` per call rather than a single per-walk scope), attach the stable
/// `code`, and record it. Message text is unchanged from the old `emitFmt`.
fn emit(g: *GraphResolve, code: codes.Code, mod: u32, byte_offset: u32, comptime fmt: []const u8, args: anytype) !void {
    g.sink.setScope(mod);
    try g.sink.emitFmtCode(code, byte_offset, fmt, args);
}

/// Like `emit`, but records a RELATED prior location: `related` is a byte offset in
/// module `mod` (e.g. a duplicate's first definition), rendered as a secondary
/// "previously defined here" label.
fn emitRelated(g: *GraphResolve, code: codes.Code, mod: u32, byte_offset: u32, related: u32, comptime fmt: []const u8, args: anytype) !void {
    g.sink.setScope(mod);
    try g.sink.emitFmtCodeRelated(code, byte_offset, related, fmt, args);
}
};

/// Resolve names across the whole module `graph`. Caller owns the returned
/// `GraphResult`. Never stops at the first error — collects diagnostics so one
/// run surfaces as many problems as possible (mirroring the single-file pass).
pub fn resolveGraph(gpa: std.mem.Allocator, graph: *const Graph.Graph) !GraphResult {
    const n = graph.modules.len;

    const tables = try gpa.alloc(ModuleTables, n);
    for (tables) |*t| t.* = .{};
    const resolutions = try gpa.alloc([]Resolution, n);
    @memset(resolutions, &.{});

    var g: GraphResolve = .{
        .gpa = gpa,
        .graph = graph,
        .tables = tables,
        .resolutions = resolutions,
        .fns = .empty,
        .sink = DiagnosticSink.init(gpa),
    };
    defer {
        for (tables) |*t| t.deinit(gpa);
        gpa.free(tables);
        for (g.scopes.items) |*s| s.names.deinit(gpa);
        g.scopes.deinit(gpa);
        g.locals.deinit(gpa);
        g.label_stack.deinit(gpa);
    }
    errdefer {
        for (resolutions) |r| if (r.len != 0) gpa.free(r);
        gpa.free(resolutions);
        for (g.fns.items) |f| gpa.free(@constCast(f.name));
        g.fns.deinit(gpa);
        g.sink.deinit();
    }

    // Each module's resolution array is parallel to its node count.
    for (graph.modules, 0..) |m, i| {
        const rr = try gpa.alloc(Resolution, m.nodes.len);
        @memset(rr, .unresolved);
        resolutions[i] = rr;
    }

    try g.collectGlobals();
    try g.collectNamespaces();
    try g.injectImportedTypes();
    for (0..n) |i| try g.resolveModule(@intCast(i));

    try g.warnUnusedFns();
    try g.warnUnusedImports();

    g.sink.sort();
    const owned = try g.sink.toOwned();
    return GraphResult{
        .resolutions = resolutions,
        .fns = try g.fns.toOwnedSlice(gpa),
        .diags = owned.diags,
        .owned_msgs = owned.owned,
    };
}

const testing = std.testing;
const Lexer = @import("lex.zig");
const Parser = @import("parse.zig");
const Cache = @import("query/Cache.zig");
const Io = std.Io;

const FixtureFile = struct { path: []const u8, source: []const u8 };

/// Write the fixture files, discover the graph, resolve it, and hand the result
/// to `check`. The `check` fn must not retain pointers past return.
fn withResolvedGraph(
    comptime dir_name: []const u8,
    files: []const FixtureFile,
    entry: []const u8,
    check: *const fn (g: *const Graph.Graph, r: *GraphResult) anyerror!void,
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

    var r = try resolveGraph(gpa, &graph);
    defer r.deinit(gpa);
    try check(&graph, &r);
}

/// Find module id by canonical path.
fn modId(g: *const Graph.Graph, path: []const u8) u32 {
    for (g.modules, 0..) |m, i| if (std.mem.eql(u8, m.path, path)) return @intCast(i);
    unreachable;
}

/// Find the (first) node of a given tag in a module.
fn nodeOfTag(g: *const Graph.Graph, mod: u32, tag: Ast.Node.Tag) Ast.Index {
    for (g.modules[mod].nodes, 0..) |n, i| if (n.tag == tag) return Ast.Index.from(@intCast(i));
    unreachable;
}

/// Count the diagnostics in a resolved result carrying code `c` (for the unused-variable
/// tests, which assert exactly N W0001s).
fn countCode(r: *GraphResult, c: codes.Code) usize {
    var n: usize = 0;
    for (r.diags) |d| if (d.code == c) {
        n += 1;
    };
    return n;
}

/// Count only ERROR-severity diagnostics — the visibility/undeclared tests assert on
/// resolution errors and must stay independent of the W-band lint warnings.
fn errorCount(r: *GraphResult) usize {
    var n: usize = 0;
    for (r.diags) |d| if (d.severity != .warning) {
        n += 1;
    };
    return n;
}

/// The first ERROR-severity diagnostic (mirrors `errorCount`; callers assert its message).
fn firstError(r: *GraphResult) Diagnostic {
    for (r.diags) |d| if (d.severity != .warning) return d;
    unreachable;
}

test "unused `:=` local warns W0001" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn main() -> int {
    \\ x := 5
    \\ return 0
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0001));
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "unused variable 'x'") != null);
        }
    };
    try withResolvedGraph(".toy-test-unused-let", files, "main.toy", Check.run);
}

test "unused `for` range loop var warns W0001" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn main() -> int {
    \\ for i in 0..3 {}
    \\ return 0
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0001));
        }
    };
    try withResolvedGraph(".toy-test-unused-for", files, "main.toy", Check.run);
}

test "unused `for-in` iterator var warns W0001" {
    // Resolve does no type-checking, so the iterable's type is irrelevant here; only the
    // NAME binding of the loop var matters. `xs` is referenced (the iterable) so never warns.
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn f(xs: int) -> int {
    \\ for x in xs {}
    \\ return 0
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0001));
        }
    };
    try withResolvedGraph(".toy-test-unused-forin", files, "main.toy", Check.run);
}

test "`_`-prefixed and bare `_` bindings do not warn" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn main() -> int {
    \\ _x := 5
    \\ for _ in 0..3 {}
    \\ return 0
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0001));
        }
    };
    try withResolvedGraph(".toy-test-unused-underscore", files, "main.toy", Check.run);
}

test "used local does not warn" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn main() -> int {
    \\ x := 5
    \\ return x
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0001));
        }
    };
    try withResolvedGraph(".toy-test-used-local", files, "main.toy", Check.run);
}

test "a local used only as an assignment target does not warn" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn main() -> int {
    \\ x := 5
    \\ x = 6
    \\ return x
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0001));
        }
    };
    try withResolvedGraph(".toy-test-used-assign", files, "main.toy", Check.run);
}

test "an unused parameter warns W0002" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn f(p: int) -> int { return 0 }
    \\fn main() -> int { return f(1) }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0002));
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "'p'") != null);
        }
    };
    try withResolvedGraph(".toy-test-unused-param", files, "main.toy", Check.run);
}

test "an `_`-prefixed parameter does not warn W0002" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn f(_p: int) -> int { return 0 }
    \\fn main() -> int { return f(1) }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0002));
        }
    };
    try withResolvedGraph(".toy-test-unused-param-underscore", files, "main.toy", Check.run);
}

test "a used parameter does not warn W0002" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn f(p: int) -> int { return p }
    \\fn main() -> int { return f(1) }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0002));
        }
    };
    try withResolvedGraph(".toy-test-used-param", files, "main.toy", Check.run);
}

test "an unused `self` receiver does not warn W0002" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\struct S { x: int }
    \\impl S { fn m(self) -> int { return 0 } }
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0002));
        }
    };
    try withResolvedGraph(".toy-test-unused-self", files, "main.toy", Check.run);
}

test "a parameter used only in a nested block is considered used" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn f(p: int) -> int {
    \\ if true {
    \\  return p
    \\ }
    \\ return 0
    \\}
    \\fn main() -> int { return f(1) }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0002));
        }
    };
    try withResolvedGraph(".toy-test-param-nested-use", files, "main.toy", Check.run);
}

test "a protocol-signature parameter never warns W0002" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\protocol P { fn m(self, x: int) -> int }
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0002));
        }
    };
    try withResolvedGraph(".toy-test-proto-sig-param", files, "main.toy", Check.run);
}

test "an unused match binding does not warn (out of scope)" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\enum E { A(int) }
    \\fn main() -> int {
    \\ e := E.A(1)
    \\ return match e {
    \\  E.A(v) -> 0
    \\ }
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0001));
        }
    };
    try withResolvedGraph(".toy-test-unused-match", files, "main.toy", Check.run);
}

test "two unused locals in one scope warn in source order" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn main() -> int {
    \\ a := 1
    \\ b := 2
    \\ return 0
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 2), countCode(r, .W0001));
            // `a` is declared before `b`, so it must render first (ascending offset).
            try testing.expect(r.diags[0].byte_offset < r.diags[1].byte_offset);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "'a'") != null);
            try testing.expect(std.mem.indexOf(u8, r.diags[1].message, "'b'") != null);
        }
    };
    try withResolvedGraph(".toy-test-unused-order", files, "main.toy", Check.run);
}

test "`for k,v in m` both unused warn twice" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn f(m: int) -> int {
    \\ for k, v in m {}
    \\ return 0
    \\}
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 2), countCode(r, .W0001));
        }
    };
    try withResolvedGraph(".toy-test-unused-kv", files, "main.toy", Check.run);
}

test "an uncalled private fn warns W0003 with its bare source name" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn helper() -> int { return 7 }
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0003));
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "unused function 'helper'") != null);
        }
    };
    try withResolvedGraph(".toy-test-unused-fn", files, "main.toy", Check.run);
}

test "a called private fn does not warn W0003" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn helper() -> int { return 7 }
    \\fn main() -> int { return helper() }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-called-fn", files, "main.toy", Check.run);
}

test "a pub uncalled fn does not warn W0003" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\pub fn api() -> int { return 7 }
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-pub-fn", files, "main.toy", Check.run);
}

test "the entry main never warns W0003 even though nothing calls it" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-main-fn", files, "main.toy", Check.run);
}

test "an uncalled impl method never warns W0003 (dispatched via the method table)" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\struct P { x: int }
    \\impl P { fn get(self) -> int { return self.x } }
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-method-fn", files, "main.toy", Check.run);
}

test "an `_`-prefixed uncalled fn does not warn W0003" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn _stub() -> int { return 7 }
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-underscore-fn", files, "main.toy", Check.run);
}

test "a recursive-only private fn does not warn W0003 (the self-call is a call site)" {
    const files = &[_]FixtureFile{.{ .path = "main.toy", .source =
    \\fn recur(n: int) -> int { return recur(n) }
    \\fn main() -> int { return 0 }
    \\
    }};
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-recursive-fn", files, "main.toy", Check.run);
}

test "a pub fn used only from another module does not warn W0003" {
    // Proves the sweep reads EVERY module's resolution array (util.used is named
    // only from main) and that the `!is_pub` exclusion holds cross-module.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int { return util.used() }
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub fn used() -> int { return 7 }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-xmod-pub-fn", files, "main.toy", Check.run);
}

test "a private fn named cross-module is R0005 and W0003 in its own module" {
    // A non-pub fn cannot be referenced across modules: the importer gets R0005 (no
    // `.func` is ever written), so the fn is genuinely uncalled and correctly W0003
    // warned in its home module.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int { return util.hidden() }
        \\
        },
        .{ .path = "util.toy", .source =
        \\fn hidden() -> int { return 7 }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .R0005));
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0003));
        }
    };
    try withResolvedGraph(".toy-test-xmod-private-fn", files, "main.toy", Check.run);
}

test "cross-module call resolves to a global fn with a qualified name" {
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
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            const entry = modId(g, "main");
            // The call's callee is a field_access `util.helper`; resolve binds it
            // to a global .func, and its receiver `util` to a .module.
            const fa = nodeOfTag(g, entry, .field_access);
            const fa_res = r.resolutions[entry][fa.int()];
            try testing.expect(fa_res == .func);
            const gf = r.fns[fa_res.func];
            try testing.expectEqualStrings("util.helper", gf.name);
            try testing.expect(gf.is_pub);
            // The receiver identifier resolves to the util module.
            const recv = g.modules[entry].nodes[g.modules[entry].nodes[fa.int()].lhs.int()];
            _ = recv;
            const recv_idx = g.modules[entry].nodes[fa.int()].lhs;
            try testing.expect(r.resolutions[entry][recv_idx.int()] == .module);
            try testing.expectEqual(modId(g, "util"), r.resolutions[entry][recv_idx.int()].module);
            // Entry main stays a bare global fn.
            var saw_main = false;
            for (r.fns) |f| if (std.mem.eql(u8, f.name, "main")) {
                saw_main = true;
            };
            try testing.expect(saw_main);
        }
    };
    try withResolvedGraph(".toy-test-res-xmod", files, "main.toy", Check.run);
}

test "two modules with same-named fn get distinct qualified global ids" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a
        \\import b
        \\fn main() -> int { return a.area() + b.area() }
        \\
        },
        .{ .path = "a.toy", .source =
        \\pub fn area() -> int { return 1 }
        \\
        },
        .{ .path = "b.toy", .source =
        \\pub fn area() -> int { return 2 }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            var saw_a = false;
            var saw_b = false;
            for (r.fns) |f| {
                if (std.mem.eql(u8, f.name, "a.area")) saw_a = true;
                if (std.mem.eql(u8, f.name, "b.area")) saw_b = true;
            }
            try testing.expect(saw_a and saw_b);
            _ = g;
        }
    };
    try withResolvedGraph(".toy-test-res-2area", files, "main.toy", Check.run);
}

test "referencing a non-pub fn cross-module is a visibility error" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int { return util.secret() }
        \\
        },
        .{ .path = "util.toy", .source =
        \\fn secret() -> int { return 9 }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), errorCount(r));
            try testing.expect(std.mem.indexOf(u8, firstError(r).message, "not exported") != null);
            try testing.expect(std.mem.indexOf(u8, firstError(r).message, "secret") != null);
        }
    };
    try withResolvedGraph(".toy-test-res-priv", files, "main.toy", Check.run);
}

test "unknown cross-module member errors against the importer" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\fn main() -> int { return util.nope() }
        \\
        },
        .{ .path = "util.toy", .source =
        \\pub fn helper() -> int { return 1 }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            try testing.expectEqual(@as(usize, 1), r.diags.len);
            try testing.expectEqual(modId(g, "main"), r.diags[0].scope);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "no member") != null);
        }
    };
    try withResolvedGraph(".toy-test-res-unknown", files, "main.toy", Check.run);
}

test "namespace collision without `as` is an error; `as` disambiguates" {
    const collide = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a/util
        \\import b/util
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "a/util.toy", .source = "pub fn f() -> int { return 1 }\n" },
        .{ .path = "b/util.toy", .source = "pub fn f() -> int { return 2 }\n" },
    };
    const Collide = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            // The subject is the collision error. The winning import `a/util` (bound
            // first) is genuinely unused and warns W0004; the losing `b/util` gets R0004.
            // Assert on the ERROR channel only so the collision is what's under test.
            try testing.expectEqual(@as(usize, 1), errorCount(r));
            try testing.expect(std.mem.indexOf(u8, firstError(r).message, "collides") != null);
        }
    };
    try withResolvedGraph(".toy-test-res-collide", collide, "main.toy", Collide.run);

    const aliased = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a/util as au
        \\import b/util as bu
        \\fn main() -> int { return au.f() + bu.f() }
        \\
        },
        .{ .path = "a/util.toy", .source = "pub fn f() -> int { return 1 }\n" },
        .{ .path = "b/util.toy", .source = "pub fn f() -> int { return 2 }\n" },
    };
    const Aliased = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), r.diags.len);
        }
    };
    try withResolvedGraph(".toy-test-res-alias", aliased, "main.toy", Aliased.run);
}

test "a local shadows an import namespace (innermost binding wins)" {
    // `util` is both an import namespace AND a local; the local must win, so
    // `util.helper()` (here a value field access on the local) does NOT resolve
    // to the module fn. We assert the receiver is a .local, not a .module.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import util
        \\struct Box { v: int }
        \\fn main() -> int {
        \\ util := Box { v: 5 }
        \\ return util.v
        \\}
        \\
        },
        .{ .path = "util.toy", .source = "pub fn helper() -> int { return 1 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            const entry = modId(g, "main");
            const fa = nodeOfTag(g, entry, .field_access);
            const recv_idx = g.modules[entry].nodes[fa.int()].lhs;
            // Receiver binds to the local Box, NOT the util module.
            try testing.expect(r.resolutions[entry][recv_idx.int()] == .local);
        }
    };
    try withResolvedGraph(".toy-test-res-shadow", files, "main.toy", Check.run);
}

test "qualified pub type in a signature resolves quietly (no error)" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import geometry/rect as r
        \\fn use(_b: r.Rect) -> int { return 0 }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "geometry/rect.toy", .source =
        \\pub struct Rect { w: int, h: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            // No undeclared/visibility error: `r.Rect` is a pub type reference.
            try testing.expectEqual(@as(usize, 0), errorCount(r));
        }
    };
    try withResolvedGraph(".toy-test-res-qtype", files, "main.toy", Check.run);
}

test "referencing a non-pub type cross-module is a visibility error" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import geometry/rect as r
        \\fn use(_b: r.Rect) -> int { return 0 }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "geometry/rect.toy", .source =
        \\struct Rect { w: int, h: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), errorCount(r));
            try testing.expect(std.mem.indexOf(u8, firstError(r).message, "not exported") != null);
        }
    };
    try withResolvedGraph(".toy-test-res-qtype-priv", files, "main.toy", Check.run);
}

test "3-level cross-module mod.Enum.Variant binds the inner receiver to .module, no error" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import palette as p
        \\fn main() -> int {
        \\ _c := p.Color.Red
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
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            // No undeclared/visibility error: `p.Color.Red` references a pub enum.
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            const entry = modId(g, "main");
            // The INNER field_access `p.Color` has receiver `p` bound to .module.
            // (There are two field_access nodes: inner p.Color, outer (p.Color).Red.)
            var inner_recv_is_module = false;
            for (g.modules[entry].nodes, 0..) |n, i| {
                if (n.tag == .field_access) {
                    const recv = g.modules[entry].nodes[n.lhs.int()];
                    if (recv.tag == .identifier) {
                        const cr = r.resolutions[entry][n.lhs.int()];
                        if (cr == .module) {
                            inner_recv_is_module = true;
                            try testing.expectEqual(modId(g, "palette"), cr.module);
                        }
                    }
                    _ = i;
                }
            }
            try testing.expect(inner_recv_is_module);
        }
    };
    try withResolvedGraph(".toy-test-res-3level", files, "main.toy", Check.run);
}

test "single-module graph resolves bodies like the single-file resolver" {
    const files = &[_]FixtureFile{
        .{ .path = "solo.toy", .source =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn main() -> int { return add(1, 2) }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            const entry = modId(g, "solo");
            // The bare call `add(...)` resolves to a global .func.
            const call = nodeOfTag(g, entry, .call);
            const callee = g.modules[entry].nodes[call.int()].lhs;
            try testing.expect(r.resolutions[entry][callee.int()] == .func);
        }
    };
    try withResolvedGraph(".toy-test-res-solo", files, "solo.toy", Check.run);
}

test "a bare `print` with no import is R0001 undeclared" {
    // `print` is ordinary library code in `std/io` now, not a global builtin. A bare
    // `print(..)` with no `import std/io` binds nothing in any module's fn table, so the
    // call callee is `.unresolved` and R0001 fires — in EVERY module, since none seed it.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        "import util\nfn main() -> int { print(\"hi\")\n return util.go() }\n" },
        .{ .path = "util.toy", .source =
        "pub fn go() -> int { print(\"yo\")\n return 0 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            var r0001: usize = 0;
            for (r.diags) |d| if (d.code == .R0001) {
                r0001 += 1;
            };
            try testing.expect(r0001 >= 1);
            // No module registers `print`, so every `print(...)` callee is `.unresolved`.
            for (g.modules, 0..) |m, mi| {
                for (m.nodes) |n| {
                    if (n.tag == .call) {
                        const callee = m.nodes[n.lhs.int()];
                        if (callee.tag == .identifier and
                            std.mem.eql(u8, m.tokens[callee.main_token].text(m.source), "print"))
                        {
                            try testing.expect(r.resolutions[mi][n.lhs.int()] == .unresolved);
                        }
                    }
                }
            }
        }
    };
    try withResolvedGraph(".toy-test-res-print", files, "main.toy", Check.run);
}

test "a non-core module naming a pub extern is a visibility error (extern name served only to core)" {
    // The entry `main.toy` is a disk (non-`core/`) module importing the REAL bundled
    // `core/ffi` and naming its pub extern `labs`. Discovery serves `core/ffi` from the
    // embedded bundle, so `ffi.labs` finds the pub `.import` member but the referring
    // module is not `core/*` → the quarantine falls through to R0005 (not exported),
    // pinning "an extern name resolves only inside `core/`".
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import core/ffi
        \\fn main() -> int { return ffi.labs(-15) }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), r.diags.len);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "not exported") != null);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "labs") != null);
        }
    };
    try withResolvedGraph(".toy-test-res-extern-gate", files, "main.toy", Check.run);
}

test "two non-generic conformances sharing a method name register distinctly (no R0002)" {
    const files = &[_]FixtureFile{
        .{ .path = "solo.toy", .source =
        \\protocol Show { fn render(self) -> int }
        \\protocol Debug { fn render(self) -> int }
        \\struct P { x: int }
        \\impl P has Show { fn render(self) -> int { return 1 } }
        \\impl P has Debug { fn render(self) -> int { return 2 } }
        \\fn main() -> int { return 0 }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            for (r.diags) |d| try testing.expect(d.code != .R0002);
            // Both conformance methods must own a distinct mangled symbol; the protocol
            // identity is what keeps them apart on the shared `P.render` receiver+name.
            var show: ?[]const u8 = null;
            var debug: ?[]const u8 = null;
            for (r.fns) |f| {
                if (std.mem.indexOf(u8, f.name, "P.render") == null) continue;
                if (std.mem.endsWith(u8, f.name, "Show")) show = f.name;
                if (std.mem.endsWith(u8, f.name, "Debug")) debug = f.name;
            }
            try testing.expect(show != null);
            try testing.expect(debug != null);
            try testing.expect(!std.mem.eql(u8, show.?, debug.?));
        }
    };
    try withResolvedGraph(".toy-test-res-multiconf", files, "solo.toy", Check.run);
}

/// The first W0004 diagnostic's message (callers assert its text).
fn firstW0004(r: *GraphResult) []const u8 {
    for (r.diags) |d| if (d.code == .W0004) return d.message;
    unreachable;
}

test "an import used only qualified does not warn W0004" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn main() -> int { return lib.f() }
        \\
        },
        .{ .path = "lib.toy", .source = "pub fn f() -> int { return 1 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-qualified", files, "main.toy", Check.run);
}

test "an import used ONLY via a bare imported type does not warn W0004" {
    // THE critical case: `Point` is nameable unqualified because `shapes` exports it;
    // the resolver never observes a bare type reference, so W0004 must credit it via the
    // provider's pub-struct table, not via a `.module` resolution.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import shapes
        \\fn f(_p: Point) -> int { return 0 }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "shapes.toy", .source = "pub struct Point { x: int, y: int }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-bare-type", files, "main.toy", Check.run);
}

test "a list literal credits its provider's `Vec` (no W0004)" {
    // A `[..]` list literal constructs a `Vec` with no textual `Vec` anywhere — the
    // hidden implicit-need rule 3. Without it this import false-positives.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import buf
        \\fn main() -> int {
        \\ _xs := [1, 2, 3]
        \\ return 0
        \\}
        \\
        },
        .{ .path = "buf.toy", .source =
        \\pub struct Vec { n: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-list", files, "main.toy", Check.run);
}

test "a `for k, v` credits its provider's `Entry` (no W0004)" {
    // `for k, v in m` binds an `Entry` with no textual `Entry` — implicit-need rule 4.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import mapmod
        \\fn f(m: int) -> int {
        \\ for k, v in m { return k + v }
        \\ return 0
        \\}
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "mapmod.toy", .source =
        \\pub struct Entry { key: int, val: int }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-forkv", files, "main.toy", Check.run);
}

test "a qualified protocol ref `impl .. has lib.Show` credits the import (no W0004)" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\struct W { x: int }
        \\impl W has lib.Show { fn render(self) -> int { return self.x } }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source = "pub protocol Show { fn render(self) -> int }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-proto", files, "main.toy", Check.run);
}

test "an import providing only a cross-module conformance is credited (no W0004)" {
    // `import show` binds a namespace never named and no bare type; its sole contribution
    // is the `impl shape.Sq has Show` that lets `q.show()` resolve in typecheck. Removing
    // it breaks the build, so the impl-crediting branch must keep it out of W0004.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import shape
        \\import show
        \\fn main() -> int {
        \\ q := Sq { s: 42 }
        \\ return q.show()
        \\}
        \\
        },
        .{ .path = "shape.toy", .source = "pub struct Sq { s: int }\n" },
        .{ .path = "show.toy", .source =
        \\import shape
        \\pub protocol Show { fn show(self) -> int }
        \\impl shape.Sq has Show { fn show(self) -> int { self.s } }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-conf", files, "main.toy", Check.run);
}

test "an import providing only a cross-module inherent method is credited (no W0004)" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import shape
        \\import ext
        \\fn main() -> int {
        \\ q := Sq { s: 25 }
        \\ return q.area()
        \\}
        \\
        },
        .{ .path = "shape.toy", .source = "pub struct Sq { s: int }\n" },
        .{ .path = "ext.toy", .source =
        \\import shape
        \\impl Sq { fn area(self) -> int { self.s } }
        \\
        },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-inherent", files, "main.toy", Check.run);
}

test "a truly-unused import warns W0004 with a remove hint" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source = "pub fn f() -> int { return 1 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0004));
            try testing.expect(std.mem.indexOf(u8, firstW0004(r), "unused import 'lib'; remove it") != null);
        }
    };
    try withResolvedGraph(".toy-test-w4-unused", files, "main.toy", Check.run);
}

test "a multi-import file warns only the unused import" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import a
        \\import b
        \\fn main() -> int { return a.f() }
        \\
        },
        .{ .path = "a.toy", .source = "pub fn f() -> int { return 1 }\n" },
        .{ .path = "b.toy", .source = "pub fn g() -> int { return 2 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0004));
            try testing.expect(std.mem.indexOf(u8, firstW0004(r), "unused import 'b'") != null);
        }
    };
    try withResolvedGraph(".toy-test-w4-multi", files, "main.toy", Check.run);
}

test "a bare imported type in a struct field credits the import (no W0004)" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import shapes
        \\struct Wrap { p: Point }
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "shapes.toy", .source = "pub struct Point { x: int }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-field", files, "main.toy", Check.run);
}

test "a module's OWN same-named type does not credit an otherwise-unused import" {
    // `Point` is referenced, but it is the module's OWN type (own wins the bare name);
    // the import's same-named pub type is unnameable, so the import still warns.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import shapes
        \\struct Point { x: int }
        \\fn main() -> int {
        \\ _p := Point { x: 0 }
        \\ return 0
        \\}
        \\
        },
        .{ .path = "shapes.toy", .source = "pub struct Point { x: int, y: int }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-own-shadow", files, "main.toy", Check.run);
}

test "a bare prelude `Option` does not credit an unused import" {
    // The prelude enums are injected pub into EVERY module's table; referencing one must
    // not mark an import used, or every import would be spuriously credited.
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib
        \\fn main() -> int {
        \\ _o := Option[int].none
        \\ return 0
        \\}
        \\
        },
        .{ .path = "lib.toy", .source = "pub fn f() -> int { return 1 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 1), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-prelude", files, "main.toy", Check.run);
}

test "an `as _`-aliased unused import is a deliberate silence (no W0004)" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import lib as _lib
        \\fn main() -> int { return 0 }
        \\
        },
        .{ .path = "lib.toy", .source = "pub fn f() -> int { return 1 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            _ = g;
            try testing.expectEqual(@as(usize, 0), countCode(r, .W0004));
        }
    };
    try withResolvedGraph(".toy-test-w4-silence", files, "main.toy", Check.run);
}
