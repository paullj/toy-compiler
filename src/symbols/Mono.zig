//! Monomorphization instance table — a peer data module.
//!
//! An `Instance` is one reachable `(generic template, concrete-arg-tuple)` pair
//! reified to an ordinary concrete per-fn codegen unit. The type checker's serial
//! monomorphization tail produces the table; codegen enumerates it as extra
//! lowerable units; `lower`/`Fingerprint`(`CallVisitor`) resolve a generic call
//! site to its instance through it. This module is the SINGLE home for the
//! canonical logic (`find`, the sort key, the mangled `SymName`) so the four
//! consumers cannot drift.
//!
//! Determinism is SACRED: `lessThan` is a total order over `(template gid, arg
//! bytes)` and `mangle` is a pure function of the template name + the arg tuple,
//! so the instance list, its ids, and its mangled symbols are a pure function of
//! source (never hashmap/thread order).

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Type = @import("../layout/Type.zig").Type;
const Infer = @import("Infer.zig");

/// One resolved bound `[T has P]` on a monomorphized instance: the witnessing
/// `impl <conform_ty> has P` chosen at the mono worklist by conformance lookup. This
/// is the structural datum folded (ordered, never XOR) into the instance's content
/// fingerprint so toggling a sibling-module conformance invalidates EXACTLY the
/// dependent monomorphizations. NOT the impl body-fp (that would over-invalidate) —
/// only the protocol identity + the conforming type's layout + the witnessing method
/// symbols in protocol-declared order.
///
/// LIFETIME (getting this wrong is a use-after-free in codegen):
///   * `protocol_name` borrows the SOURCE-backed `ProtocolSym.name` (stable; the
///     `t.protocols` array is freed at typecheck teardown but the name BYTES are
///     source slices that outlive codegen).
///   * `witness_syms` element slices borrow `gph_fn_names` (resolve-result-backed,
///     outlives codegen); the OUTER `witness_syms` slice is OWNED by the Instance.
///   * `conform_ty` is a 12-byte POD (the ground conforming `structT`/`enumT`/scalar).
///   * `protocol_args` are the bound's protocol type-args (`[T has Into[int]]` ->
///     `[int]`), substituted through the instance args — 12-byte POD Types. Empty for a
///     non-generic protocol. The OUTER slice is OWNED by the Instance. Folded structurally
///     into the (e) fingerprint component so an `Into[int]` -> `Into[bool]` edit can't
///     serve a stale cached witness.
/// Only the outer `conformances` slice + each `witness_syms`/`protocol_args` outer slice
/// are owned.
pub const ResolvedConformance = struct {
    protocol_name: []const u8,
    conform_ty: Type,
    witness_syms: []const []const u8,
    protocol_args: []const Type = &.{},
};

/// One reified generic instance. All slices are OWNED (freed by the owning
/// `GraphResult`).
pub const Instance = struct {
    /// Global fn id of the generic template this instance specializes.
    template_gid: u32,
    /// The concrete type-args, in generic-param order.
    args: []const Type,
    /// A per-instance `node_types` slice sized to the template module's node count
    /// (written by the per-instance re-check). NEVER the module's shared slice, so
    /// two instances of one template do not alias each other's node slots.
    node_types: []Type,
    /// The template's params/ret with every `type_var` substituted to concrete.
    params: []const Type,
    ret: Type,
    /// The mangled instance symbol name (e.g. `id$int`), minted in canonical order
    /// after the sort. `null` until then, so a read (`.name.?`) traps loudly on
    /// out-of-order use.
    name: ?[]const u8 = null,
    /// Owning module id (the template's module) and the template's decl node.
    mod: u32,
    decl_node: Ast.Index,
    /// The resolved bounds `[T has P]` witnessing conformances for this instance,
    /// one per bounded generic param, in generic-param order. Empty (`&.{}`)
    /// for an unbounded template's instance so its fingerprint fold is byte-identical
    /// (warm cache preserved). The OUTER slice + each entry's `witness_syms` outer
    /// slice are OWNED by the owning `GraphResult`; see `ResolvedConformance`.
    conformances: []const ResolvedConformance = &.{},
};

/// A generic template's inference facts, supplied by the caller's read-only view
/// (`params` is the template's value params; `count` is the generic-param count fed
/// to `Infer.infer` and used as the explicit-turbofish arity). Discovery derives it
/// from the `FnSym` (`generic_params.len`); the fingerprint fold from the `Sig`
/// (`Sig.genericParamCount`) — they agree for any call that survives Pass C.
pub const TemplateRef = struct { params: []const Type, count: u32 };

/// A recovered generic call site: the template `gid` and its concrete type-args
/// (OWNED — the caller frees `args`).
pub const CallRef = struct { gid: u32, args: []Type };

/// Recover `(template gid, concrete type-args)` from a `.call` node's callee — the
/// single arg-extraction both mono discovery (`types.scanCalls`) and the content
/// fingerprint (`AstWalk.CallVisitor`) walk, so they cannot drift on which
/// `Instance` a call selects. Two call-shapes:
///   * explicit turbofish `id[int](..)` — the callee is a `type_app`; the type-args
///     are read straight from `node_types`.
///   * bare inferred `id(7)` — the callee is a plain `identifier`; the type-args are
///     inferred from the value-arg `node_types` via `Infer.infer`.
/// A field-access method callee is NOT handled here (discovery binds impl params;
/// the fingerprint resolves the conformance witness — genuinely different leaves).
///
/// `view` supplies `.tree`, `.resolutions`, `.node_types`, and
/// `genericTemplate(gid) -> ?TemplateRef` (returns null for a non-generic callee,
/// which filters both shapes). Returns null for any non-generic / non-`.func` /
/// out-of-range / uninferable call; on `.ok` the caller owns `.args`.
pub fn callInstanceRef(gpa: std.mem.Allocator, view: anytype, call_node: Ast.Node) error{OutOfMemory}!?CallRef {
    if (call_node.tag != .call or call_node.lhs == Ast.none) return null;
    const tree = view.tree;
    const callee = tree.nodes[call_node.lhs.int()];
    switch (callee.tag) {
        .type_app => {
            const bres = view.resolutions[callee.lhs.int()];
            if (bres != .func) return null;
            const gid = bres.func;
            if (view.genericTemplate(gid) == null) return null;
            const targ_nodes = Ast.rangeSlice(tree, callee.rhs.int());
            const args = try gpa.alloc(Type, targ_nodes.len);
            errdefer gpa.free(args);
            for (targ_nodes, 0..) |tn, k| {
                if (tn.int() >= view.node_types.len) {
                    gpa.free(args);
                    return null; // pre-typecheck view
                }
                args[k] = view.node_types[tn.int()];
            }
            return .{ .gid = gid, .args = args };
        },
        .identifier => {
            const bres = view.resolutions[call_node.lhs.int()];
            if (bres != .func) return null;
            const gid = bres.func;
            const tmpl = view.genericTemplate(gid) orelse return null;
            const value_args = Ast.rangeSlice(tree, call_node.rhs.int());
            const arg_types = try gpa.alloc(Type, value_args.len);
            defer gpa.free(arg_types);
            for (value_args, 0..) |va, k| {
                if (va.int() >= view.node_types.len) return null; // pre-typecheck view
                arg_types[k] = view.node_types[va.int()];
            }
            const targs = (try Infer.infer(gpa, tmpl.count, tmpl.params, arg_types)) orelse return null;
            return .{ .gid = gid, .args = targs };
        },
        else => return null,
    }
}

/// The index of the instance for `(gid, args)`, or null. A deterministic linear
/// scan comparing the template gid + each arg by `Type.eql`.
pub fn find(insts: []const Instance, gid: u32, args: []const Type) ?usize {
    for (insts, 0..) |inst, i| {
        if (inst.template_gid != gid or inst.args.len != args.len) continue;
        var ok = true;
        for (inst.args, args) |a, b| {
            if (!Type.eql(a, b)) {
                ok = false;
                break;
            }
        }
        if (ok) return i;
    }
    return null;
}

/// The canonical total order: by template gid, then lexicographically by the arg
/// tuple's `(kind, struct_id, enum_id)` triples. Drives BOTH the instance
/// enumeration order and the name-minting order, so the whole table is a pure
/// function of source.
pub fn lessThan(_: void, a: Instance, b: Instance) bool {
    if (a.template_gid != b.template_gid) return a.template_gid < b.template_gid;
    const n = @min(a.args.len, b.args.len);
    for (a.args[0..n], b.args[0..n]) |x, y| {
        switch (x.orderLeaf(y)) {
            .lt => return true,
            .gt => return false,
            .eq => {},
        }
    }
    return a.args.len < b.args.len;
}

/// Append the canonical dedup key for `(gid, args)` to `buf`: the gid then each
/// arg's `(kind, struct_id, enum_id)` in a fixed width. Injective — two distinct
/// `(gid, args)` never share a key (the gid fixes the arity, so lengths never
/// collide). Used by the mono tail's worklist dedup.
pub fn writeKey(gpa: std.mem.Allocator, buf: *std.ArrayList(u8), gid: u32, args: []const Type) !void {
    var w: [4]u8 = undefined;
    std.mem.writeInt(u32, &w, gid, .little);
    try buf.appendSlice(gpa, &w);
    for (args) |a| try a.appendKeyBytes(gpa, buf);
}

/// The mangled instance symbol: `<template>$<arg0>$<arg1>...`. `$` is not a valid
/// source identifier char, so a mangled name can never collide with a user fn.
/// Struct/enum args mangle to `s<id>`/`e<id>` (the global id is a deterministic
/// function of source); the id is only a SEPARATION layer — a struct-layout edit
/// that keeps the id still flips the instance's fingerprint via the type-arg
/// layout fold, so the mangled name never needs to encode the layout.
pub fn mangle(gpa: std.mem.Allocator, template_name: []const u8, args: []const Type) ![]u8 {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(gpa);
    try buf.appendSlice(gpa, template_name);
    var nb: [16]u8 = undefined;
    for (args) |a| {
        try buf.append(gpa, '$');
        switch (a.kind) {
            .int => try buf.appendSlice(gpa, a.intName()),
            .bool => try buf.appendSlice(gpa, "bool"),
            .str => try buf.appendSlice(gpa, "str"),
            .unit => try buf.appendSlice(gpa, "unit"),
            .float => try buf.appendSlice(gpa, "float"),
            .@"struct" => try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "s{d}", .{a.nominalId()}) catch unreachable),
            .@"enum" => try buf.appendSlice(gpa, std.fmt.bufPrint(&nb, "e{d}", .{a.nominalId()}) catch unreachable),
            else => try buf.appendSlice(gpa, "x"),
        }
    }
    return buf.toOwnedSlice(gpa);
}

const testing = std.testing;

const Resolution = @import("Resolution.zig").Resolution;

/// A hand-built read-only view over a tiny tree, matching the `anytype` interface
/// `callInstanceRef` reads. `genericTemplate` answers for exactly one gid.
const FakeView = struct {
    tree: Ast.Tree,
    resolutions: []const Resolution,
    node_types: []const Type,
    gen_gid: u32,
    gen_params: []const Type,
    gen_count: u32,
    fn genericTemplate(self: FakeView, gid: u32) ?TemplateRef {
        if (gid != self.gen_gid) return null;
        return .{ .params = self.gen_params, .count = self.gen_count };
    }
};

test "callInstanceRef recovers (gid, args) from an explicit turbofish callee" {
    const gpa = testing.allocator;
    // Tree for `id[int](x)`: n0=id, n1=int(type-arg), n2=type_app(lhs=0,rhs=hdr@0),
    // n3=x(value-arg), n4=call(lhs=2,rhs=hdr@3).
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .type_app, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.Index.from(0) },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(2), .rhs = Ast.Index.from(3) },
    };
    var extra = [_]u32{ 2, 1, 1, 4, 1, 3 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 5;
    resolutions[0] = .{ .func = 7 };
    const node_types = [_]Type{ Type.invalid, Type.int, Type.invalid, Type.invalid, Type.invalid };
    const view = FakeView{ .tree = tree, .resolutions = &resolutions, .node_types = &node_types, .gen_gid = 7, .gen_params = &.{}, .gen_count = 1 };

    const ref = (try callInstanceRef(gpa, view, nodes[4])).?;
    defer gpa.free(ref.args);
    try testing.expectEqual(@as(u32, 7), ref.gid);
    try testing.expectEqual(@as(usize, 1), ref.args.len);
    try testing.expect(Type.eql(ref.args[0], Type.int));
}

test "callInstanceRef infers (gid, args) from a bare generic callee" {
    const gpa = testing.allocator;
    // Tree for `id(x)`: n0=id, n1=x(value-arg), n2=call(lhs=0,rhs=hdr@0).
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.Index.from(0) },
    };
    var extra = [_]u32{ 2, 1, 1 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 3;
    resolutions[0] = .{ .func = 5 };
    const node_types = [_]Type{ Type.invalid, Type.int, Type.invalid };
    const params = [_]Type{Type.typeVar(0)};
    const view = FakeView{ .tree = tree, .resolutions = &resolutions, .node_types = &node_types, .gen_gid = 5, .gen_params = &params, .gen_count = 1 };

    const ref = (try callInstanceRef(gpa, view, nodes[2])).?;
    defer gpa.free(ref.args);
    try testing.expectEqual(@as(u32, 5), ref.gid);
    try testing.expectEqual(@as(usize, 1), ref.args.len);
    try testing.expect(Type.eql(ref.args[0], Type.int));
}

test "callInstanceRef returns null for a non-generic or non-func callee" {
    const gpa = testing.allocator;
    var nodes = [_]Ast.Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none },
        .{ .tag = .call, .main_token = 0, .lhs = Ast.Index.from(0), .rhs = Ast.Index.from(0) },
    };
    var extra = [_]u32{ 2, 1, 1 };
    const tree = Ast.Tree{ .nodes = &nodes, .extra = &extra };
    var resolutions = [_]Resolution{.unresolved} ** 3;
    resolutions[0] = .{ .func = 5 };
    const node_types = [_]Type{ Type.invalid, Type.int, Type.invalid };
    // gen_gid 9 != the callee's gid 5 -> genericTemplate is null -> not a template.
    const view = FakeView{ .tree = tree, .resolutions = &resolutions, .node_types = &node_types, .gen_gid = 9, .gen_params = &.{}, .gen_count = 1 };
    try testing.expectEqual(@as(?CallRef, null), try callInstanceRef(gpa, view, nodes[2]));
}

test "mangle is index-free-ish and distinct per arg tuple" {
    const gpa = testing.allocator;
    const a = try mangle(gpa, "id", &.{Type.int});
    defer gpa.free(a);
    try testing.expectEqualStrings("id$int", a);
    const b = try mangle(gpa, "id", &.{Type.structT(0)});
    defer gpa.free(b);
    try testing.expectEqualStrings("id$s0", b);
    try testing.expect(!std.mem.eql(u8, a, b));
    const c = try mangle(gpa, "pair", &.{ Type.int, Type.bool });
    defer gpa.free(c);
    try testing.expectEqualStrings("pair$int$bool", c);
}

test "mangle encodes integer width/sign distinctly" {
    const gpa = testing.allocator;
    const i8m = try mangle(gpa, "id", &.{Type.int8});
    defer gpa.free(i8m);
    try testing.expectEqualStrings("id$int8", i8m);
    const u32m = try mangle(gpa, "id", &.{Type.uint32});
    defer gpa.free(u32m);
    try testing.expectEqualStrings("id$uint32", u32m);
    const im = try mangle(gpa, "id", &.{Type.int});
    defer gpa.free(im);
    // Plain int is byte-identical to before; the widths never collide with it.
    try testing.expectEqualStrings("id$int", im);
    try testing.expect(!std.mem.eql(u8, i8m, im));
    try testing.expect(!std.mem.eql(u8, u32m, im));
    try testing.expect(!std.mem.eql(u8, i8m, u32m));
}

test "find matches on gid + args by Type.eql" {
    const insts = [_]Instance{
        .{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .params = &.{Type.int}, .ret = Type.int, .name = "id$int", .mod = 0, .decl_node = Ast.none },
        .{ .template_gid = 0, .args = &.{Type.structT(0)}, .node_types = &.{}, .params = &.{Type.structT(0)}, .ret = Type.structT(0), .name = "id$s0", .mod = 0, .decl_node = Ast.none },
    };
    try testing.expectEqual(@as(?usize, 0), find(&insts, 0, &.{Type.int}));
    try testing.expectEqual(@as(?usize, 1), find(&insts, 0, &.{Type.structT(0)}));
    try testing.expectEqual(@as(?usize, null), find(&insts, 0, &.{Type.bool}));
    try testing.expectEqual(@as(?usize, null), find(&insts, 1, &.{Type.int}));
}

test "an instance carries its resolved conformances" {
    const witness = [_][]const u8{"lib.P.into$Into$int"};
    const pargs = [_]Type{Type.int};
    const confs = [_]ResolvedConformance{.{ .protocol_name = "Into", .conform_ty = Type.structT(3), .witness_syms = &witness, .protocol_args = &pargs }};
    const inst = Instance{
        .template_gid = 5,
        .args = &.{Type.structT(3)},
        .node_types = &.{},
        .params = &.{Type.structT(3)},
        .ret = Type.int,
        .name = "use$s3",
        .mod = 0,
        .decl_node = Ast.none,
        .conformances = &confs,
    };
    try testing.expectEqual(@as(usize, 1), inst.conformances.len);
    try testing.expectEqualStrings("Into", inst.conformances[0].protocol_name);
    try testing.expect(Type.eql(Type.structT(3), inst.conformances[0].conform_ty));
    try testing.expectEqualStrings("lib.P.into$Into$int", inst.conformances[0].witness_syms[0]);
    // The bound's protocol type-args ride the conformance (folded into the (e) fp).
    try testing.expectEqual(@as(usize, 1), inst.conformances[0].protocol_args.len);
    try testing.expect(Type.eql(Type.int, inst.conformances[0].protocol_args[0]));
    // An instance built WITHOUT conformances defaults to the empty slice (warm-cache
    // byte-identity for unbounded templates); its conformance's protocol_args default too.
    const plain = Instance{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .params = &.{Type.int}, .ret = Type.int, .name = "id$int", .mod = 0, .decl_node = Ast.none };
    try testing.expectEqual(@as(usize, 0), plain.conformances.len);
}

test "lessThan is a total canonical order (gid then arg triples)" {
    const a = Instance{ .template_gid = 0, .args = &.{Type.int}, .node_types = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    const b = Instance{ .template_gid = 0, .args = &.{Type.structT(0)}, .node_types = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    const c = Instance{ .template_gid = 1, .args = &.{Type.int}, .node_types = &.{}, .params = &.{}, .ret = Type.int, .name = "", .mod = 0, .decl_node = Ast.none };
    // int (kind 2) < struct (kind 6), so a < b; gid 0 < gid 1, so a < c.
    try testing.expect(lessThan({}, a, b));
    try testing.expect(!lessThan({}, b, a));
    try testing.expect(lessThan({}, a, c));
    try testing.expect(lessThan({}, b, c));
}
