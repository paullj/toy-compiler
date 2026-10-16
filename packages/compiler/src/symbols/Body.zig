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
nodes: Nodes,
/// Its boxed locals: resolver slots, ascending.
boxed: []const u32,

/// A body's types and call selections for a run of its module's nodes, indexed by module
/// node index. A fn's run is its whole module; an instance's is its template's decl range,
/// since a module-sized pair per instance made N instances of a template declared late in
/// a large module cost O(N x module) (#41). A node before `base` reads as unchecked, as it
/// would in a module-sized table; one past the end has no slot.
pub const Nodes = struct {
    base: u32 = 0,
    types: []const Type = &.{},
    /// Parallel to `types`.
    callees: []const Callee = &.{},

    pub fn typeOf(n: Nodes, idx: Ast.Index) Type {
        const i = idx.int();
        return if (i < n.base) .invalid else n.types[i - n.base];
    }

    /// Whether node `idx` has a slot.
    pub fn has(n: Nodes, idx: Ast.Index) bool {
        const i = idx.int();
        return i >= n.base and i - n.base < n.types.len;
    }

    /// Node `idx`'s call selection, `.none` when it has no slot.
    pub fn callee(n: Nodes, idx: Ast.Index) Callee {
        const i = idx.int();
        return if (i < n.base or i - n.base >= n.callees.len) .{} else n.callees[i - n.base];
    }
};

/// What a body lowers: a fn by global id, or an instance by its index in the instance
/// table.
pub const Of = union(enum) { base: u32, instance: u32 };

/// A body's position in `List.items`.
pub const Index = u32;

pub fn view(b: Body, tree: Ast.Tree, resolutions: []const Resolution) BodyView {
    return .{ .tree = tree, .resolutions = resolutions, .nodes = b.nodes };
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
    nodes: Nodes,
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
        .{ .of = .{ .base = 2 }, .mod = 0, .decl = Ast.none, .sig = sig, .nodes = .{}, .boxed = &.{} },
        .{ .of = .{ .base = 5 }, .mod = 0, .decl = Ast.none, .sig = sig, .nodes = .{}, .boxed = &.{} },
        .{ .of = .{ .instance = 0 }, .mod = 0, .decl = Ast.none, .sig = sig, .nodes = .{}, .boxed = &.{} },
    };
    const l: List = .{ .items = &b, .n_base = 2 };
    try std.testing.expectEqual(@as(?Index, 0), l.ofFn(2));
    try std.testing.expectEqual(@as(?Index, 1), l.ofFn(5));
    try std.testing.expectEqual(@as(?Index, null), l.ofFn(3));
    try std.testing.expectEqual(@as(Index, 2), l.ofInstance(0));
}

test "a body's nodes read unchecked before their base and by offset after it" {
    const types = [_]Type{ Type.int, Type.bool };
    const callees = [_]Callee{ .{}, Callee.direct(4) };
    const n: Nodes = .{ .base = 3, .types = &types, .callees = &callees };
    try std.testing.expect(!n.has(Ast.Index.from(2)));
    try std.testing.expect(n.has(Ast.Index.from(3)) and n.has(Ast.Index.from(4)));
    try std.testing.expect(!n.has(Ast.Index.from(5)));
    try std.testing.expect(n.typeOf(Ast.Index.from(2)).kind == .invalid);
    try std.testing.expect(Type.eql(Type.bool, n.typeOf(Ast.Index.from(4))));
    try std.testing.expectEqual(Callee.direct(4), n.callee(Ast.Index.from(4)));
    try std.testing.expectEqual(@as(Callee, .{}), n.callee(Ast.Index.from(5)));
}
