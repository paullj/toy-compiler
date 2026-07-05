//! Whole-graph name resolution.
//!
//! Generalises the single-file `resolve.zig` across a discovered module `Graph`.
//! Two phases, mirroring the per-file resolver but at graph scope:
//!
//!   * GLOBAL COLLECT — walk every module's top-level decls once. Assign each
//!     user fn a GRAPH-GLOBAL id (deterministic: module order, then decl order)
//!     and a MODULE-QUALIFIED symbol name (`<module-path>.<fn>`), except the entry
//!     module's `main` (stays bare `main`) and the synthetic `print` (one shared
//!     `print` at the end of the table). Each module records its own fns (pub +
//!     private, visible only inside the module) plus its `pub` fns/structs/enums
//!     (visible to importers under the import namespace). Each module's `import`
//!     decls bind a namespace name → imported module id, rejecting a same-last-
//!     segment collision (unless `as`).
//!
//!   * PER-MODULE BODY RESOLVE — resolve each fn body in its own lexical state,
//!     exactly like the single-file resolver, with two cross-module additions:
//!       - `lookupName`: locals → this module's fns → the global `print` →
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
const SymKind = @import("symbols/Sym.zig").SymKind;

/// A graph-global function symbol.
pub const GlobalFn = struct {
    /// Module-qualified symbol name (`geometry/rect.area`), or bare for the
    /// entry `main` / the synthetic `print`. Owned by `GraphResult`.
    name: []const u8,
    /// Owning module id (index into `Graph.modules`). For `print` this is the
    /// entry module (it has no real home; only the bare name matters).
    module: u32,
    /// The `fn_decl` node in that module's tree (`Ast.none` for the bodyless
    /// builtin `print`, identified by `kind` — never test `decl_node` for builtin-ness).
    decl_node: Ast.Index,
    /// `.user_fn` for a real decl, `.builtin` for the synthetic `print`. The
    /// single source of "is this a builtin", carried across the resolve→types boundary.
    kind: SymKind,
    /// Whether the decl is `pub` (exported). `main`/`print` are not pub.
    is_pub: bool,
    /// For an inherent method (M8): the receiver type-ref node (an `identifier`) in
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

const Local = struct { name_tok: u32, slot: u32 };
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

/// Assign every user fn a global id + qualified name, and register each module's
/// struct/enum type names. Deterministic: modules in id order, decls in source
/// order. `print` is seeded last (one shared synthetic builtin).
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
                .struct_decl => {
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
                .impl_decl, .impl_has_decl => {
                    // Each method is an ordinary global fn with a MANGLED name
                    // `<module.path>.<Receiver>.<method>` (so it gets a global id +
                    // codegen unit) but is NOT bare-callable (never inserted into
                    // `tables[mod].fns`) — a method is reached only via `x.m(..)`
                    // dispatch through the program-wide method table (built in Pass A).
                    // A conformance method (`impl_has_decl`) registers IDENTICALLY: its
                    // protocol-method dispatch is indistinguishable from an inherent one
                    // (M11). `Ast.implMethods` yields the method run for either shape.
                    const recv_name = g.nameOf(mod, decl.main_token);
                    // M14: a GENERIC-protocol conformance (`impl P has Into[int]`) mangles
                    // the method symbol + duplicate-check key with the protocol identity +
                    // its type-args, so `Into[int]` and `Into[bool]` don't collide on the
                    // shared `P.into` symbol (a linker duplicate) nor trip a spurious R0002.
                    // A non-generic-protocol conformance / inherent impl keeps the bare
                    // `<recv>.<method>` name — byte-identical to M11/M13.
                    var proto_suffix: []const u8 = "";
                    defer if (proto_suffix.len > 0) g.gpa.free(proto_suffix);
                    if (decl.tag == .impl_has_decl) {
                        if (Ast.implProtocol(t, decl)) |proto_ref| {
                            const pargs = Ast.protocolRefArgs(t, proto_ref);
                            if (pargs.len > 0) {
                                var sb: std.ArrayList(u8) = .empty;
                                errdefer sb.deinit(g.gpa);
                                try sb.append(g.gpa, '$');
                                try sb.appendSlice(g.gpa, g.nameOf(mod, m.nodes[Ast.protocolRefBase(t, proto_ref).int()].main_token));
                                for (pargs) |an| {
                                    try sb.append(g.gpa, '$');
                                    try sb.appendSlice(g.gpa, g.nameOf(mod, m.nodes[an.int()].main_token));
                                }
                                proto_suffix = try sb.toOwnedSlice(g.gpa);
                            }
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
                            // `decl.lhs` is the receiver type-ref: a bare `identifier`
                            // (M8), a generic `impl Box[T]` (M10) `type_app`, or a
                            // qualified `impl mod.T` (M11) `field_access` —
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

    // Seed the shared synthetic `print` builtin once, at the end of the table.
    // Every module's bare `print` resolves to this id (global prelude). It is
    // bodyless and not pub; `kind = .builtin` is the homeless entry's marker.
    const print_id: u32 = @intCast(g.fns.items.len);
    try g.fns.append(g.gpa, .{
        .name = try g.gpa.dupe(u8, "print"),
        .module = entry,
        .decl_node = Ast.none,
        .kind = .builtin,
        .is_pub = false,
    });
    // Register `print` into every module's local fn table UNLESS that module
    // declares its own `print` (the user fn keeps its own id, as in single-file).
    for (g.tables) |*t| {
        if (!t.fns.contains("print")) try t.fns.put(g.gpa, "print", print_id);
    }
    // The prelude `enum Ordering { lt, eq, gt }` (M16) is nameable in every module with no
    // import (the `print` precedent). Register it into each module's enum table UNLESS the
    // module declares its own `Ordering` (user-first-wins), so `Ordering.lt` construction and
    // a match on it resolve quietly (left for Typecheck, which native-registers the enum). The
    // typecheck-time `enum_ids` injection in `registerPrelude` mirrors this on its own tables.
    for (g.tables) |*t| {
        if (!t.enums.contains("Ordering")) try t.enums.put(g.gpa, "Ordering", true);
    }
    // The prelude generic value enums `Option`/`Result` (M23) are likewise nameable in every
    // module with no import (`Option.some(..)` / `Option[int].none` / a bare `.none` under a
    // target). Register into each module's enum table UNLESS the module declares its own
    // (user-first-wins), mirroring `Ordering`; Typecheck native-registers the enum itself.
    for (g.tables) |*t| {
        if (!t.enums.contains("Option")) try t.enums.put(g.gpa, "Option", true);
    }
    for (g.tables) |*t| {
        if (!t.enums.contains("Result")) try t.enums.put(g.gpa, "Result", true);
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
        _ = try g.declare(param.main_token, "duplicate parameter '{s}'");
        // Resolve the param's type-ref for a qualified cross-module type.
        try g.resolveTypeRef(param.lhs);
    }
    try g.resolveTypeRef(proto.ret_type);
    try g.resolveBlock(decl.rhs);
    g.popScope();
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
            const slot = try g.declare(stmt.main_token, "redeclaration of '{s}'");
            if (slot) |s| g.res(stmt_idx, .{ .local = s });
        },
        .assign => {
            const target = g.nodes()[stmt.lhs.int()];
            if (target.tag == .field_access) {
                try g.resolveExpr(stmt.lhs);
            } else {
                const resn = g.lookupName(target.main_token);
                g.res(stmt.lhs, resn);
                switch (resn) {
                    .unresolved => try g.emit(.R0007, g.cur_mod, g.tokens()[target.main_token].start, "assignment to undeclared name '{s}'", .{g.nameText(target.main_token)}),
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
            const slot = try g.declare(stmt.main_token, "redeclaration of '{s}'");
            if (slot) |s| g.res(stmt_idx, .{ .local = s });
            try g.resolveBlock(stmt.lhs);
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
                // If an in-scope name is a close typo of the undeclared one, append a
                // "did you mean" hint (message-embedded — no note channel yet). The
                // suggester is conservative (short names / distant names / ties → no
                // hint), so this stays byte-identical for the existing no-hint cases.
                if (nearmiss.suggest(name, g.candidateIter())) |cand|
                    try g.emit(.R0001, g.cur_mod, off, "undeclared identifier '{s}'; did you mean '{s}'?", .{ name, cand })
                else
                    try g.emit(.R0001, g.cur_mod, off, "undeclared identifier '{s}'", .{name});
            }
        },
        .literal_number, .literal_string, .literal_bool => {},
        // A poison leaf is already-diagnosed: no name lookup, no diagnostic.
        .error_node => {},
        .unary => try g.resolveExpr(n.lhs),
        .binary => {
            try g.resolveExpr(n.lhs);
            try g.resolveExpr(n.rhs);
        },
        .call => {
            try g.resolveExpr(n.lhs); // callee (may be a field_access mod.fn)
            for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |arg| try g.resolveExpr(arg);
        },
        .struct_init => {
            for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |fi| try g.resolveExpr(g.nodes()[fi.int()].lhs);
        },
        .field_access => try g.resolveFieldAccess(node_idx, n),
        // A type-application callee `id[int](..)`: resolve ONLY the base callee
        // (`n.lhs`). The type-arg Range is NOT descended — resolving a type name
        // like `int` would fire R0001 and stop the pipeline at resolve, pre-empting
        // the T0013 generics gate at typecheck. Forward-safety for M2.
        .type_app => try g.resolveExpr(n.lhs),
        // Postfix `?` (M24): resolve the operand's names; the `?` itself binds nothing.
        .try_expr => try g.resolveExpr(n.lhs),
        .enum_init_unit => {},
        .enum_init_tuple => for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |a| try g.resolveExpr(a),
        .enum_init_struct => for (Ast.rangeSlice(g.tree(g.cur_mod), n.rhs.int())) |fi| try g.resolveExpr(g.nodes()[fi.int()].lhs),
        .match_expr => try g.resolveMatch(node_idx),
        .literal_unit => {},
        .block => try g.resolveBlock(node_idx),
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
        g.res(node_idx, .{ .func = gid });
        return;
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
        .loop_expr, .while_stmt, .for_stmt => try g.resolveStmt(n.lhs),
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
            const slot = try g.declare(pat.main_token, "redeclaration of '{s}'");
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
                const slot = try g.declare(pat.main_token, "redeclaration of '{s}'");
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

fn declare(g: *GraphResolve, name_tok: u32, comptime dup_fmt: []const u8) !?u32 {
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
    try g.locals.append(g.gpa, .{ .name_tok = name_tok, .slot = slot });
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
            return .{ .local = g.locals.items[local_idx].slot };
        }
    }
    if (g.tables[g.cur_mod].fns.get(name)) |gid| return .{ .func = gid };
    return null;
}

/// An iterator over exactly the names `lookupName` searches, for near-miss
/// suggestions: every lexical scope's local names (innermost scope first), then this
/// module's fn names (incl. the shared `print`), then this module's import namespace
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

/// Look a name up: locals (innermost first) → this module's fns (incl. the
/// shared `print`) → import namespaces. A lexical binding shadows a namespace.
fn lookupName(g: *GraphResolve, name_tok: u32) Resolution {
    if (g.lookupLocalOrFn(name_tok)) |r| return r;
    const name = g.nameText(name_tok);
    if (g.tables[g.cur_mod].namespaces.get(name)) |mod| return .{ .module = mod };
    return .unresolved;
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
    for (0..n) |i| try g.resolveModule(@intCast(i));

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
            try testing.expectEqual(@as(usize, 1), r.diags.len);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "not exported") != null);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "secret") != null);
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
            try testing.expectEqual(@as(usize, 1), r.diags.len);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "collides") != null);
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
        \\fn use(b: r.Rect) -> int { return 0 }
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
            try testing.expectEqual(@as(usize, 0), r.diags.len);
        }
    };
    try withResolvedGraph(".toy-test-res-qtype", files, "main.toy", Check.run);
}

test "referencing a non-pub type cross-module is a visibility error" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        \\import geometry/rect as r
        \\fn use(b: r.Rect) -> int { return 0 }
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
            try testing.expectEqual(@as(usize, 1), r.diags.len);
            try testing.expect(std.mem.indexOf(u8, r.diags[0].message, "not exported") != null);
        }
    };
    try withResolvedGraph(".toy-test-res-qtype-priv", files, "main.toy", Check.run);
}

test "3-level cross-module mod.Enum.Variant binds the inner receiver to .module, no error" {
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

test "print is available unqualified in every module with one shared id" {
    const files = &[_]FixtureFile{
        .{ .path = "main.toy", .source =
        "import util\nfn main() -> int { print(\"hi\")\n return util.go() }\n" },
        .{ .path = "util.toy", .source =
        "pub fn go() -> int { print(\"yo\")\n return 0 }\n" },
    };
    const Check = struct {
        fn run(g: *const Graph.Graph, r: *GraphResult) anyerror!void {
            try testing.expectEqual(@as(usize, 0), r.diags.len);
            // Find the print global fn id (bodyless, name "print").
            var print_id: ?u32 = null;
            for (r.fns, 0..) |f, i| if (std.mem.eql(u8, f.name, "print")) {
                print_id = @intCast(i);
            };
            try testing.expect(print_id != null);
            // Both modules' print(...) call callees resolve to that same id.
            for (g.modules, 0..) |m, mi| {
                for (m.nodes) |n| {
                    if (n.tag == .call) {
                        const callee = m.nodes[n.lhs.int()];
                        if (callee.tag == .identifier and
                            std.mem.eql(u8, m.tokens[callee.main_token].text(m.source), "print"))
                        {
                            const cr = r.resolutions[mi][n.lhs.int()];
                            try testing.expect(cr == .func);
                            try testing.expectEqual(print_id.?, cr.func);
                        }
                    }
                }
            }
        }
    };
    try withResolvedGraph(".toy-test-res-print", files, "main.toy", Check.run);
}
