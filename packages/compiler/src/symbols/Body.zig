//! One lowered body written in source, and the program's bodies numbered once: every
//! non-generic fn with a body in global fn id order, then every monomorphized instance in
//! instance order. A peer data module like `Closure`: types builds the list when its tables
//! are final, and the closure tables, the escape pass, the spec table, codegen and the LSP
//! all name a body by its index in it, which is also the body's codegen unit number.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Type = @import("../layout/Type.zig").Type;
const Layout = @import("../layout/Engine.zig").Layout;
const Resolution = @import("Resolution.zig").Resolution;
const Callee = @import("Callee.zig").Callee;
const Sig = @import("Sig.zig").Sig;
const Sym = @import("Sym.zig");
const Mono = @import("Mono.zig");
const Method = @import("../types.zig").Method;

const Body = @This();

of: Of,
mod: u32,
decl: Ast.Index,
/// Its own signature: a fn's `sigs[gid]`, or an instance's reified params and return
/// under its mangled name, with its template's modes, since modes are syntactic. So
/// `sig.mut_mask` is the body's `mut` params, `mut self` first.
sig: Sig,
node_types: []const Type,
callees: []const Callee,
/// Its boxed locals: resolver slots, ascending.
boxed: []const u32,

/// What a body lowers: a fn by global id, or an instance by its index in the instance
/// table.
pub const Of = union(enum) { base: u32, instance: u32 };

/// A body's position in `List.items`.
pub const Index = u32;

pub fn view(b: Body, tree: Ast.Tree, resolutions: []const Resolution) BodyView {
    return .{ .tree = tree, .resolutions = resolutions, .node_types = b.node_types, .callees = b.callees };
}

/// The program's bodies, the base fns first.
pub const List = struct {
    items: []const Body = &.{},
    /// How many base fns lead `items`, ascending by global id.
    n_base: u32 = 0,

    /// The body of fn `gid`, or null when it lowers none of its own (a builtin, an extern,
    /// a generic template).
    pub fn ofFn(l: List, gid: u32) ?Index {
        const i = std.sort.binarySearch(Body, l.items[0..l.n_base], gid, struct {
            fn order(key: u32, b: Body) std.math.Order {
                return std.math.order(key, b.of.base);
            }
        }.order) orelse return null;
        return @intCast(i);
    }

    pub fn ofInstance(l: List, ii: u32) Index {
        return l.n_base + ii;
    }
};

/// One body's tables: its module's tree and resolutions, and the types and callees its own
/// check recorded (an instance's, never its template's module slice).
pub const BodyView = struct {
    tree: Ast.Tree,
    resolutions: []const Resolution,
    node_types: []const Type,
    callees: []const Callee,
};

/// The program-wide tables a body's calls and values are read against.
pub const ProgramView = struct {
    sigs: []const Sig,
    methods: []const Method,
    instances: []const Mono.Instance,
    /// One per global fn id, parallel to `sigs`.
    names: []const Sym.SymName = &.{},
    layouts: []const Layout = &.{},
};

test "a fn's body is found by its global id, an instance's by its index" {
    const sig: Sig = .{ .kind = .user_fn, .name = "f", .params = &.{}, .ret = Type.unit };
    const b = [_]Body{
        .{ .of = .{ .base = 2 }, .mod = 0, .decl = Ast.none, .sig = sig, .node_types = &.{}, .callees = &.{}, .boxed = &.{} },
        .{ .of = .{ .base = 5 }, .mod = 0, .decl = Ast.none, .sig = sig, .node_types = &.{}, .callees = &.{}, .boxed = &.{} },
        .{ .of = .{ .instance = 0 }, .mod = 0, .decl = Ast.none, .sig = sig, .node_types = &.{}, .callees = &.{}, .boxed = &.{} },
    };
    const l: List = .{ .items = &b, .n_base = 2 };
    try std.testing.expectEqual(@as(?Index, 0), l.ofFn(2));
    try std.testing.expectEqual(@as(?Index, 1), l.ofFn(5));
    try std.testing.expectEqual(@as(?Index, null), l.ofFn(3));
    try std.testing.expectEqual(@as(Index, 2), l.ofInstance(0));
}
