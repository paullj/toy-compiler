//! The type algebra + layout deep module.
//!
//! Owns the pure `Type` algebra (kinds, constants, `eql`/`assignable`) AND the
//! struct/enum layout engine: the `LayoutState` cycle-detection state machine,
//! `layoutReferent` recursion, and poison propagation. The recursion + the state
//! machine are HIDDEN; the public surface is the type/layout queries plus the two
//! lazy entry points (`layoutStruct`/`layoutEnum`) the checker drives in id order.
//!
//! The engine mutates the checker's struct/enum tables in place through a narrow
//! `Env` callback (the laid DATA — `.field_types`/`.offsets`/`.size`/`.poisoned` —
//! stays visible to the checker exactly as before; only the laying machinery moved
//! here). `Env` thunks forward to the checker's existing per-module accessors so
//! diagnostic text/order, cross-module id resolution, and the active-module swap
//! are byte-for-byte unchanged.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");

/// The type kind. `invalid` is the poison/error type: it absorbs further errors
/// so one mistake produces one diagnostic. `@"struct"` carries a `struct_id`
/// indexing the per-program struct table.
pub const Kind = enum(u8) { invalid, unit, int, bool, str, never, @"struct", @"enum" };

/// A type. A byte-foldable struct (not a tagged union) so it preserves `@memset`,
/// `node_types` triviality, and a stable fingerprint basis. A `@"struct"` kind
/// carries an index into the struct table; a `@"enum"` kind an index into the
/// parallel enum table; all other kinds leave both `no_struct`.
pub const Type = struct {
    kind: Kind,
    struct_id: u32 = no_struct,
    enum_id: u32 = no_struct,

    pub const no_struct: u32 = std.math.maxInt(u32);

    pub const invalid: Type = .{ .kind = .invalid };
    pub const unit: Type = .{ .kind = .unit };
    pub const int: Type = .{ .kind = .int };
    pub const @"bool": Type = .{ .kind = .bool };
    pub const str: Type = .{ .kind = .str };
    pub const never: Type = .{ .kind = .never };

    pub fn structT(id: u32) Type {
        return .{ .kind = .@"struct", .struct_id = id };
    }

    pub fn enumT(id: u32) Type {
        return .{ .kind = .@"enum", .enum_id = id };
    }

    pub fn eql(a: Type, b: Type) bool {
        return a.kind == b.kind and
            (a.kind != .@"struct" or a.struct_id == b.struct_id) and
            (a.kind != .@"enum" or a.enum_id == b.enum_id);
    }

    /// The one assignability relation: is a value of type `got` acceptable where a
    /// `want` is expected? Mirrors `merge`'s discipline so the two stay siblings —
    /// `invalid` is poison (either side absorbs, so a root error yields exactly one
    /// diagnostic and never cascades), `never` is bottom (a value that never exists
    /// fits any slot), and otherwise assignability is structural equality with NO
    /// implicit coercion. Returning `true` for poison/`never` means a caller guards
    /// `if (!assignable(want, got)) emit(...)` and stays silent on already-reported
    /// or unreachable values — exactly the manual `kind != .invalid` guards it
    /// replaces.
    pub fn assignable(want: Type, got: Type) bool {
        if (want.kind == .invalid or got.kind == .invalid) return true;
        if (got.kind == .never) return true;
        return eql(want, got);
    }

    pub fn isStruct(t: Type) bool {
        return t.kind == .@"struct";
    }

    pub fn isEnum(t: Type) bool {
        return t.kind == .@"enum";
    }
};

/// A resolved struct layout: a self-contained, index-free-ish snapshot threaded
/// into Codegen/Fingerprint. Field types still reference the struct table by id
/// (for nested structs), but offsets/size/align are precomputed here. Owned.
pub const Layout = struct {
    name: []const u8,
    field_names: [][]const u8,
    field_types: []Type,
    offsets: []u32,
    size: u32,
    @"align": u32,
};

/// A variant's form: a unit (no payload), a tuple (positional payload, no field
/// names), or a struct (named payload fields).
pub const VariantForm = enum(u8) { unit, tuple, @"struct" };

/// A resolved enum layout: a value tagged union — an 8-byte tag at offset 0, then
/// payload storage sized to the largest variant's payload at `payload_off`. Each
/// variant carries its payload field types + payload-LOCAL offsets (relative to
/// `payload_off`). Owned (parallel to `Layout`). Threaded read-only into Codegen.
pub const VariantLayout = struct {
    name: []const u8,
    form: VariantForm,
    field_names: [][]const u8,
    field_types: []Type,
    /// Payload-local offsets (add `payload_off` for the absolute byte offset).
    offsets: []u32,
};
pub const EnumLayout = struct {
    name: []const u8,
    variants: []VariantLayout,
    tag_size: u32,
    payload_off: u32,
    size: u32,
    @"align": u32,
};

/// A struct's resolved symbol: its decl node, name, and (after layout) per-field
/// names/types/offsets plus aggregate size/align. `state` guards the layout
/// recursion so a directly- or indirectly-recursive struct is caught once.
///
/// PUBLIC: the checker's frozen `Model` aliases this type and `BodyChecker` reads
/// `.state`/`.poisoned`/`.field_*`/`.size` at fingerprint + exhaustiveness time.
pub const LayoutState = enum { unseen, laying, done };
pub const StructSym = struct {
    decl_node: Ast.Index,
    name: []const u8,
    field_names: [][]const u8 = &.{},
    field_types: []Type = &.{},
    offsets: []u32 = &.{},
    size: u32 = 0,
    @"align": u32 = 1,
    state: LayoutState = .unseen,
    poisoned: bool = false,
    /// Owning module id (graph mode); 0 single-file. Layout switches to it.
    mod: u32 = 0,
    /// Whether the struct decl is `pub` (graph mode; pub-signature coherence).
    pub_export: bool = false,
};

/// One variant in the scratch enum table (during layout). `field_names`/`name`
/// are BORROWED source slices; `field_types`/`offsets` are owned arrays.
pub const VariantSym = struct {
    name: []const u8,
    form: VariantForm,
    field_names: [][]const u8 = &.{},
    field_types: []Type = &.{},
    offsets: []u32 = &.{},
    payload_size: u32 = 0,
    payload_align: u32 = 1,
};

/// An enum's resolved symbol. The tag is a fixed 8 bytes at offset 0; the payload
/// is sized to the largest variant and laid at `payload_off`. `state` guards the
/// layout recursion so a recursive enum is caught once.
pub const EnumSym = struct {
    decl_node: Ast.Index,
    name: []const u8,
    variants: []VariantSym = &.{},
    tag_size: u32 = 8,
    payload_off: u32 = 8,
    size: u32 = 0,
    @"align": u32 = 8,
    state: LayoutState = .unseen,
    poisoned: bool = false,
    /// Owning module id (graph mode); 0 single-file. Layout switches to it.
    mod: u32 = 0,
    /// Whether the enum decl is `pub` (graph mode; pub-signature coherence).
    pub_export: bool = false,
};

/// Natural size/align of a scalar/str type (struct sizes come from the table).
fn scalarSize(kind: Kind) u32 {
    return switch (kind) {
        .int, .bool => 8,
        .str => 16,
        else => 0,
    };
}
fn scalarAlign(kind: Kind) u32 {
    return switch (kind) {
        .int, .bool, .str => 8,
        else => 1,
    };
}
fn roundUp(n: u32, a: u32) u32 {
    if (a == 0) return n;
    return (n + a - 1) / a * a;
}

/// The narrowed checker context the layout recursion needs. The engine owns the
/// laying machinery but borrows the checker's tables + per-module accessors via
/// these thunks. `ctx` is the erased `*Typecheck`; each thunk casts it back and
/// forwards to the checker's existing method, so diagnostic text/order and the
/// active-module swap (graph mode) are byte-for-byte unchanged. The `emit*` thunks
/// are typed-per-message rather than a format-string passthrough so the format
/// string + args (hence the emitted bytes) stay fixed at the call site.
pub const Env = struct {
    gpa: std.mem.Allocator,
    /// The checker's tables; the engine mutates them in place (poison rides on
    /// the syms, not a return value).
    structs: *std.ArrayList(StructSym),
    enums: *std.ArrayList(EnumSym),
    ctx: *anyopaque,
    /// Swap the active module to `mod` (graph mode), returning the previous module
    /// for restore. Single-file is a no-op.
    gphSelect: *const fn (ctx: *anyopaque, mod: u32) u32,
    /// Resolve a type-reference node in the active module to a `Type`.
    typeFromNode: *const fn (ctx: *anyopaque, n: Ast.Index) Type,
    nameText: *const fn (ctx: *anyopaque, tok: u32) []const u8,
    byteOf: *const fn (ctx: *anyopaque, tok: u32) u32,
    /// "recursive type '{s}' has infinite size" at `byte`, naming `requester`.
    emitRecursive: *const fn (ctx: *anyopaque, byte: u32, requester: []const u8) error{OutOfMemory}!void,
    /// "empty struct '{s}' is not allowed" at `byte`.
    emitEmptyStruct: *const fn (ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void,
    /// "empty enum '{s}' is not allowed" at `byte`.
    emitEmptyEnum: *const fn (ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void,
    /// "field '{s}' cannot have type ()" at `byte`.
    emitUnitField: *const fn (ctx: *anyopaque, byte: u32, field: []const u8) error{OutOfMemory}!void,
    /// "variant '{s}' payload cannot have type ()" at `byte`.
    emitUnitPayload: *const fn (ctx: *anyopaque, byte: u32, variant: []const u8) error{OutOfMemory}!void,
    /// The active tree (after `gphSelect`). Resolved lazily inside the engine so
    /// the read happens AFTER the active-module swap (mirroring the checker).
    tree: *const fn (ctx: *anyopaque) Ast.Tree,
};

/// Lay out struct `id`: field offsets in declaration order with natural
/// alignment, size = aligned total, align = max field align. A `laying` field
/// of struct type means a cycle (direct or indirect) → infinite size, poisoned.
pub fn layoutStruct(env: Env, id: u32) error{OutOfMemory}!void {
    if (env.structs.items[id].state == .done) return;
    env.structs.items[id].state = .laying;

    // Graph mode: lay this struct out in ITS owning module's tree (a nested/qualified
    // field type may have switched the active module). Restore on the way out.
    const prev = env.gphSelect(env.ctx, env.structs.items[id].mod);
    defer _ = env.gphSelect(env.ctx, prev);
    const tree = env.tree(env.ctx);

    const decl = tree.nodes[env.structs.items[id].decl_node.int()];
    const field_nodes = Ast.rangeSlice(tree, decl.lhs.int());
    const n = field_nodes.len;

    // An empty struct lays out to size 0 — a zero-size aggregate is an ABI/codegen
    // hazard (no eightbytes, a degenerate sret). Reject it with a clean diagnostic.
    var empty_poison = false;
    if (n == 0) {
        try env.emitEmptyStruct(env.ctx, env.byteOf(env.ctx, decl.main_token), env.structs.items[id].name);
        empty_poison = true;
    }

    const names = try env.gpa.alloc([]const u8, n);
    errdefer env.gpa.free(names);
    const types = try env.gpa.alloc(Type, n);
    errdefer env.gpa.free(types);
    const offsets = try env.gpa.alloc(u32, n);
    errdefer env.gpa.free(offsets);

    var running: u32 = 0;
    var max_align: u32 = 1;
    var poisoned = false;
    for (field_nodes, 0..) |field_idx, i| {
        const field = tree.nodes[field_idx.int()];
        names[i] = env.nameText(env.ctx, field.main_token);
        const fty = env.typeFromNode(env.ctx, field.lhs);
        types[i] = fty;
        var fsize: u32 = 0;
        var falign: u32 = 1;
        if (fty.kind == .unit) {
            try env.emitUnitField(env.ctx, env.byteOf(env.ctx, field.main_token), names[i]);
            poisoned = true;
        } else if (fty.kind != .invalid) {
            const sz = try layoutReferent(env, fty, env.byteOf(env.ctx, decl.main_token), env.structs.items[id].name, &poisoned);
            fsize = sz.size;
            falign = sz.@"align";
        }
        const off = roundUp(running, falign);
        offsets[i] = off;
        running = off + fsize;
        if (falign > max_align) max_align = falign;
    }

    env.structs.items[id].field_names = names;
    env.structs.items[id].field_types = types;
    env.structs.items[id].offsets = offsets;
    env.structs.items[id].@"align" = max_align;
    env.structs.items[id].size = if (poisoned or empty_poison) 0 else roundUp(running, max_align);
    env.structs.items[id].poisoned = poisoned or empty_poison;
    env.structs.items[id].state = .done;
}

/// Size/align of a field/payload type, laying out a nested struct/enum on demand.
/// A `laying` referent means a cycle (direct or indirect): set `*requester_poison`
/// and emit a recursion diagnostic at `at` naming `requester`. A scalar/str uses
/// the natural sizes; `invalid`/`unit` size to 0 (the caller diagnoses `unit`).
fn layoutReferent(env: Env, ty: Type, at: u32, requester: []const u8, requester_poison: *bool) error{OutOfMemory}!struct { size: u32, @"align": u32 } {
    switch (ty.kind) {
        .@"struct" => {
            if (env.structs.items[ty.struct_id].state == .laying) {
                try env.emitRecursive(env.ctx, at, requester);
                requester_poison.* = true;
                return .{ .size = 0, .@"align" = 1 };
            }
            try layoutStruct(env, ty.struct_id);
            return .{ .size = env.structs.items[ty.struct_id].size, .@"align" = env.structs.items[ty.struct_id].@"align" };
        },
        .@"enum" => {
            if (env.enums.items[ty.enum_id].state == .laying) {
                try env.emitRecursive(env.ctx, at, requester);
                requester_poison.* = true;
                return .{ .size = 0, .@"align" = 1 };
            }
            try layoutEnum(env, ty.enum_id);
            return .{ .size = env.enums.items[ty.enum_id].size, .@"align" = env.enums.items[ty.enum_id].@"align" };
        },
        else => return .{ .size = scalarSize(ty.kind), .@"align" = scalarAlign(ty.kind) },
    }
}

/// Lay out enum `id`: an 8-byte tag at offset 0, then payload storage sized to the
/// largest variant's payload at `payload_off`. Each variant's payload fields get
/// payload-LOCAL offsets. A `laying` referent (direct/indirect cycle) poisons it.
pub fn layoutEnum(env: Env, id: u32) error{OutOfMemory}!void {
    if (env.enums.items[id].state == .done) return;
    env.enums.items[id].state = .laying;

    // Graph mode: lay this enum out in ITS owning module's tree. Restore on exit.
    const prev = env.gphSelect(env.ctx, env.enums.items[id].mod);
    defer _ = env.gphSelect(env.ctx, prev);
    const tree = env.tree(env.ctx);

    const decl = tree.nodes[env.enums.items[id].decl_node.int()];
    const variant_nodes = Ast.rangeSlice(tree, decl.lhs.int());
    const nv = variant_nodes.len;

    var poisoned = false;
    if (nv == 0) {
        try env.emitEmptyEnum(env.ctx, env.byteOf(env.ctx, decl.main_token), env.enums.items[id].name);
        poisoned = true;
    }

    const variants = try env.gpa.alloc(VariantSym, nv);
    errdefer env.gpa.free(variants);
    var vbuilt: usize = 0;
    errdefer for (variants[0..vbuilt]) |v| {
        env.gpa.free(v.field_names);
        env.gpa.free(v.field_types);
        env.gpa.free(v.offsets);
    };

    var max_payload_size: u32 = 0;
    var max_payload_align: u32 = 1;
    for (variant_nodes, 0..) |vnode_idx, vi| {
        const vnode = tree.nodes[vnode_idx.int()];
        const vname = env.nameText(env.ctx, vnode.main_token);
        var form: VariantForm = .unit;
        var payload_nodes: []const Ast.Index = &.{};
        var is_struct_form = false;
        switch (vnode.tag) {
            .enum_variant_unit => {},
            .enum_variant_tuple => {
                form = .tuple;
                payload_nodes = Ast.rangeSlice(tree, vnode.lhs.int());
            },
            .enum_variant_struct => {
                form = .@"struct";
                is_struct_form = true;
                payload_nodes = Ast.rangeSlice(tree, vnode.lhs.int());
            },
            else => {},
        }
        const np = payload_nodes.len;
        const fnames = try env.gpa.alloc([]const u8, if (is_struct_form) np else 0);
        errdefer env.gpa.free(fnames);
        const ftypes = try env.gpa.alloc(Type, np);
        errdefer env.gpa.free(ftypes);
        const foffs = try env.gpa.alloc(u32, np);
        errdefer env.gpa.free(foffs);

        var running: u32 = 0;
        var palign: u32 = 1;
        for (payload_nodes, 0..) |pnode_idx, pi| {
            // A tuple payload node is a type-ref; a struct payload node is a `param`
            // (name + type-ref in lhs).
            const pty: Type = if (is_struct_form) blk: {
                const pnode = tree.nodes[pnode_idx.int()];
                fnames[pi] = env.nameText(env.ctx, pnode.main_token);
                break :blk env.typeFromNode(env.ctx, pnode.lhs);
            } else env.typeFromNode(env.ctx, pnode_idx);
            ftypes[pi] = pty;
            var psize: u32 = 0;
            var pa: u32 = 1;
            if (pty.kind == .unit) {
                try env.emitUnitPayload(env.ctx, env.byteOf(env.ctx, vnode.main_token), vname);
                poisoned = true;
            } else if (pty.kind != .invalid) {
                const sz = try layoutReferent(env, pty, env.byteOf(env.ctx, decl.main_token), env.enums.items[id].name, &poisoned);
                psize = sz.size;
                pa = sz.@"align";
            }
            const off = roundUp(running, pa);
            foffs[pi] = off;
            running = off + psize;
            if (pa > palign) palign = pa;
        }
        const payload_size = roundUp(running, palign);
        variants[vi] = .{
            .name = vname,
            .form = form,
            .field_names = fnames,
            .field_types = ftypes,
            .offsets = foffs,
            .payload_size = payload_size,
            .payload_align = palign,
        };
        vbuilt += 1;
        if (payload_size > max_payload_size) max_payload_size = payload_size;
        if (palign > max_payload_align) max_payload_align = palign;
    }

    const tag_size: u32 = 8;
    const payload_off = roundUp(tag_size, max_payload_align);
    const aln = @max(@as(u32, 8), max_payload_align);
    env.enums.items[id].variants = variants;
    env.enums.items[id].tag_size = tag_size;
    env.enums.items[id].payload_off = payload_off;
    env.enums.items[id].@"align" = aln;
    env.enums.items[id].size = if (poisoned) 0 else roundUp(payload_off + max_payload_size, aln);
    env.enums.items[id].poisoned = poisoned;
    env.enums.items[id].state = .done;
}

pub fn snapshotLayouts(gpa: std.mem.Allocator, structs: []const StructSym) ![]Layout {
    const layouts = try gpa.alloc(Layout, structs.len);
    var built: usize = 0;
    errdefer {
        freeLayouts(gpa, layouts[0..built]);
        gpa.free(layouts);
    }
    for (structs, 0..) |s, i| {
        const fnames = try gpa.alloc([]const u8, s.field_names.len);
        var dn: usize = 0;
        errdefer {
            for (fnames[0..dn]) |x| gpa.free(x);
            gpa.free(fnames);
        }
        for (s.field_names, 0..) |nm, j| {
            fnames[j] = try gpa.dupe(u8, nm);
            dn += 1;
        }
        layouts[i] = .{
            .name = try gpa.dupe(u8, s.name),
            .field_names = @ptrCast(fnames),
            .field_types = try gpa.dupe(Type, s.field_types),
            .offsets = try gpa.dupe(u32, s.offsets),
            .size = s.size,
            .@"align" = s.@"align",
        };
        built += 1;
    }
    return layouts;
}

pub fn freeLayouts(gpa: std.mem.Allocator, layouts: []const Layout) void {
    for (layouts) |l| {
        gpa.free(l.name);
        for (l.field_names) |fn_| gpa.free(fn_);
        gpa.free(l.field_names);
        gpa.free(l.field_types);
        gpa.free(l.offsets);
    }
    gpa.free(layouts);
}

pub fn snapshotEnumLayouts(gpa: std.mem.Allocator, enums: []const EnumSym) ![]EnumLayout {
    const enum_layouts = try gpa.alloc(EnumLayout, enums.len);
    var built: usize = 0;
    errdefer {
        freeEnumLayouts(gpa, enum_layouts[0..built]);
        gpa.free(enum_layouts);
    }
    for (enums, 0..) |e, i| {
        const variants = try gpa.alloc(VariantLayout, e.variants.len);
        var vbuilt: usize = 0;
        errdefer {
            for (variants[0..vbuilt]) |v| {
                for (v.field_names) |x| gpa.free(x);
                gpa.free(v.field_names);
                gpa.free(v.field_types);
                gpa.free(v.offsets);
            }
            gpa.free(variants);
        }
        for (e.variants, 0..) |v, j| {
            const fnames = try gpa.alloc([]const u8, v.field_names.len);
            var dn: usize = 0;
            errdefer {
                for (fnames[0..dn]) |x| gpa.free(x);
                gpa.free(fnames);
            }
            for (v.field_names, 0..) |nm, k| {
                fnames[k] = try gpa.dupe(u8, nm);
                dn += 1;
            }
            variants[j] = .{
                .name = v.name,
                .form = v.form,
                .field_names = @ptrCast(fnames),
                .field_types = try gpa.dupe(Type, v.field_types),
                .offsets = try gpa.dupe(u32, v.offsets),
            };
            vbuilt += 1;
        }
        enum_layouts[i] = .{
            .name = try gpa.dupe(u8, e.name),
            .variants = variants,
            .tag_size = e.tag_size,
            .payload_off = e.payload_off,
            .size = e.size,
            .@"align" = e.@"align",
        };
        built += 1;
    }
    return enum_layouts;
}

pub fn freeEnumLayouts(gpa: std.mem.Allocator, enum_layouts: []const EnumLayout) void {
    for (enum_layouts) |e| {
        gpa.free(e.name);
        for (e.variants) |v| {
            for (v.field_names) |fn_| gpa.free(fn_);
            gpa.free(v.field_names);
            gpa.free(v.field_types);
            gpa.free(v.offsets);
        }
        gpa.free(e.variants);
    }
    gpa.free(enum_layouts);
}

// ---- boundary tests (R3) -------------------------------------------------
//
// These drive the GENUINE `layoutStruct`/`layoutEnum`/`layoutReferent` over a
// hand-built minimal `Ast.Tree` + a stub `Env`, with NO `check()` pass. The tree
// is real (the engine walks `decl.lhs` -> `rangeSlice` -> field/variant nodes and
// reads `main_token` for names/byte offsets exactly as in production); only the
// type-ref RESOLUTION is stubbed: `typeFromNode` looks up a pre-seeded `Type` per
// type-ref node index, so a field can name a (possibly recursive) struct/enum id
// without a resolver. `emit*` thunks capture diagnostics into a list. If the
// laying-guard, poison propagation, or offset math regress, these tests fail. The
// full-pass wiring (registration id-order + gphSelect + sink.sort) stays covered
// by the end-to-end tests in types.zig.

const testing = std.testing;

const CapturedDiag = struct { msg: []u8, byte: u32 };

/// A hand-built AST + the stub Env state. The engine reads `tree`/`nameText`/
/// `byteOf` for real; `type_of` maps a type-ref node index to its seeded `Type`.
const Harness = struct {
    gpa: std.mem.Allocator,
    nodes: std.ArrayList(Ast.Node),
    extra: std.ArrayList(u32),
    /// Synthetic tokens: index -> byte start. `nameText` returns `names[token]`.
    starts: std.ArrayList(u32),
    names: std.ArrayList([]const u8),
    /// type-ref node index -> resolved Type (the stubbed `typeFromNode`).
    type_of: std.AutoHashMapUnmanaged(Ast.Index, Type),
    structs: std.ArrayList(StructSym),
    enums: std.ArrayList(EnumSym),
    diags: std.ArrayList(CapturedDiag),

    fn init(gpa: std.mem.Allocator) Harness {
        return .{
            .gpa = gpa,
            .nodes = .empty,
            .extra = .empty,
            .starts = .empty,
            .names = .empty,
            .type_of = .empty,
            .structs = .empty,
            .enums = .empty,
            .diags = .empty,
        };
    }

    fn deinit(h: *Harness) void {
        for (h.diags.items) |d| h.gpa.free(d.msg);
        h.diags.deinit(h.gpa);
        for (h.enums.items) |e| {
            for (e.variants) |v| {
                h.gpa.free(v.field_names);
                h.gpa.free(v.field_types);
                h.gpa.free(v.offsets);
            }
            h.gpa.free(e.variants);
        }
        h.enums.deinit(h.gpa);
        for (h.structs.items) |s| {
            h.gpa.free(s.field_names);
            h.gpa.free(s.field_types);
            h.gpa.free(s.offsets);
        }
        h.structs.deinit(h.gpa);
        h.type_of.deinit(h.gpa);
        h.names.deinit(h.gpa);
        h.starts.deinit(h.gpa);
        h.extra.deinit(h.gpa);
        h.nodes.deinit(h.gpa);
    }

    /// Intern a token (text + a distinct byte start) and return its index.
    fn tok(h: *Harness, text: []const u8) !u32 {
        const i: u32 = @intCast(h.starts.items.len);
        try h.starts.append(h.gpa, i * 100); // distinct, checkable byte offsets
        try h.names.append(h.gpa, text);
        return i;
    }

    /// Append a node and return its index.
    fn node(h: *Harness, n: Ast.Node) !Ast.Index {
        const i = Ast.Index.from(@intCast(h.nodes.items.len));
        try h.nodes.append(h.gpa, n);
        return i;
    }

    /// Write a `{start, len}` range header over `items` into `extra`; return the
    /// header cell as an `Ast.Index` (what a Node's `lhs` stores).
    fn range(h: *Harness, items: []const Ast.Index) !Ast.Index {
        const start: u32 = @intCast(h.extra.items.len + 2);
        const header: u32 = @intCast(h.extra.items.len);
        try h.extra.append(h.gpa, start);
        try h.extra.append(h.gpa, @intCast(items.len));
        for (items) |it| try h.extra.append(h.gpa, it.int());
        return Ast.Index.from(header);
    }

    /// A `param`-shaped field node `name: <typeref>` whose type-ref resolves to
    /// `ty`. Returns the field node index.
    fn field(h: *Harness, name: []const u8, ty: Type) !Ast.Index {
        const tref = try h.node(.{ .tag = .identifier, .main_token = try h.tok("T"), .lhs = Ast.none, .rhs = Ast.none });
        try h.type_of.put(h.gpa, tref, ty);
        return h.node(.{ .tag = .param, .main_token = try h.tok(name), .lhs = tref, .rhs = Ast.none });
    }

    /// Register a struct decl over the given field nodes; returns its global id
    /// (== append index, mirroring `registerStructs`).
    fn struct_(h: *Harness, name: []const u8, fields: []const Ast.Index) !u32 {
        const hdr = try h.range(fields);
        const decl = try h.node(.{ .tag = .struct_decl, .main_token = try h.tok(name), .lhs = hdr, .rhs = Ast.none });
        const id: u32 = @intCast(h.structs.items.len);
        try h.structs.append(h.gpa, .{ .decl_node = decl, .name = name });
        return id;
    }

    /// Register an enum decl over the given variant nodes; returns its global id.
    fn enum_(h: *Harness, name: []const u8, variants: []const Ast.Index) !u32 {
        const hdr = try h.range(variants);
        const decl = try h.node(.{ .tag = .enum_decl, .main_token = try h.tok(name), .lhs = hdr, .rhs = Ast.none });
        const id: u32 = @intCast(h.enums.items.len);
        try h.enums.append(h.gpa, .{ .decl_node = decl, .name = name });
        return id;
    }

    /// A unit variant `Name`.
    fn vUnit(h: *Harness, name: []const u8) !Ast.Index {
        return h.node(.{ .tag = .enum_variant_unit, .main_token = try h.tok(name), .lhs = Ast.none, .rhs = Ast.none });
    }

    /// A tuple variant `Name(<types...>)` whose payload type-refs resolve to `tys`.
    fn vTuple(h: *Harness, name: []const u8, tys: []const Type) !Ast.Index {
        var refs: std.ArrayList(Ast.Index) = .empty;
        defer refs.deinit(h.gpa);
        for (tys) |ty| {
            const tref = try h.node(.{ .tag = .identifier, .main_token = try h.tok("T"), .lhs = Ast.none, .rhs = Ast.none });
            try h.type_of.put(h.gpa, tref, ty);
            try refs.append(h.gpa, tref);
        }
        const hdr = try h.range(refs.items);
        return h.node(.{ .tag = .enum_variant_tuple, .main_token = try h.tok(name), .lhs = hdr, .rhs = Ast.none });
    }

    /// A struct variant `Name { f: T, ... }`; `pairs` is name/type per field.
    fn vStruct(h: *Harness, name: []const u8, pairs: []const struct { []const u8, Type }) !Ast.Index {
        var fs: std.ArrayList(Ast.Index) = .empty;
        defer fs.deinit(h.gpa);
        for (pairs) |p| try fs.append(h.gpa, try h.field(p[0], p[1]));
        const hdr = try h.range(fs.items);
        return h.node(.{ .tag = .enum_variant_struct, .main_token = try h.tok(name), .lhs = hdr, .rhs = Ast.none });
    }

    fn tree(h: *Harness) Ast.Tree {
        return .{ .nodes = h.nodes.items, .extra = h.extra.items };
    }

    fn env(h: *Harness) Env {
        return .{
            .gpa = h.gpa,
            .structs = &h.structs,
            .enums = &h.enums,
            .ctx = @ptrCast(h),
            .gphSelect = stubGphSelect,
            .typeFromNode = stubTypeFromNode,
            .nameText = stubNameText,
            .byteOf = stubByteOf,
            .emitRecursive = stubEmitRecursive,
            .emitEmptyStruct = stubEmitEmptyStruct,
            .emitEmptyEnum = stubEmitEmptyEnum,
            .emitUnitField = stubEmitUnitField,
            .emitUnitPayload = stubEmitUnitPayload,
            .tree = stubTree,
        };
    }
};

fn hcast(ctx: *anyopaque) *Harness {
    return @ptrCast(@alignCast(ctx));
}

fn stubGphSelect(_: *anyopaque, mod: u32) u32 {
    return mod; // single-file no-op (returns prev == arg)
}
fn stubTree(ctx: *anyopaque) Ast.Tree {
    return hcast(ctx).tree();
}
fn stubTypeFromNode(ctx: *anyopaque, n: Ast.Index) Type {
    return hcast(ctx).type_of.get(n) orelse .invalid;
}
fn stubNameText(ctx: *anyopaque, t: u32) []const u8 {
    return hcast(ctx).names.items[t];
}
fn stubByteOf(ctx: *anyopaque, t: u32) u32 {
    return hcast(ctx).starts.items[t];
}
fn pushDiag(ctx: *anyopaque, byte: u32, comptime fmt: []const u8, args: anytype) error{OutOfMemory}!void {
    const h = hcast(ctx);
    const msg = try std.fmt.allocPrint(h.gpa, fmt, args);
    try h.diags.append(h.gpa, .{ .msg = msg, .byte = byte });
}
fn stubEmitRecursive(ctx: *anyopaque, byte: u32, requester: []const u8) error{OutOfMemory}!void {
    return pushDiag(ctx, byte, "recursive type '{s}' has infinite size", .{requester});
}
fn stubEmitEmptyStruct(ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void {
    return pushDiag(ctx, byte, "empty struct '{s}' is not allowed", .{name});
}
fn stubEmitEmptyEnum(ctx: *anyopaque, byte: u32, name: []const u8) error{OutOfMemory}!void {
    return pushDiag(ctx, byte, "empty enum '{s}' is not allowed", .{name});
}
fn stubEmitUnitField(ctx: *anyopaque, byte: u32, fld: []const u8) error{OutOfMemory}!void {
    return pushDiag(ctx, byte, "field '{s}' cannot have type ()", .{fld});
}
fn stubEmitUnitPayload(ctx: *anyopaque, byte: u32, variant: []const u8) error{OutOfMemory}!void {
    return pushDiag(ctx, byte, "variant '{s}' payload cannot have type ()", .{variant});
}

/// Count captured diagnostics whose message equals `want`.
fn countDiag(h: *Harness, want: []const u8) usize {
    var c: usize = 0;
    for (h.diags.items) |d| {
        if (std.mem.eql(u8, d.msg, want)) c += 1;
    }
    return c;
}

test "algebra: assignable structural equality with no coercion" {
    try testing.expect(Type.assignable(Type.int, Type.int));
    try testing.expect(Type.assignable(Type.structT(3), Type.structT(3)));
    try testing.expect(!Type.assignable(Type.int, Type.bool));
    try testing.expect(!Type.assignable(Type.structT(1), Type.structT(2)));
    try testing.expect(!Type.assignable(Type.int, Type.str));
}

test "algebra: invalid is poison (either side absorbs)" {
    try testing.expect(Type.assignable(Type.invalid, Type.int));
    try testing.expect(Type.assignable(Type.int, Type.invalid));
    try testing.expect(Type.assignable(Type.invalid, Type.invalid));
}

test "algebra: never is bottom (fits any want)" {
    try testing.expect(Type.assignable(Type.int, Type.never));
    try testing.expect(Type.assignable(Type.structT(0), Type.never));
    try testing.expect(!Type.assignable(Type.never, Type.int));
}

test "algebra: eql discriminates struct/enum ids" {
    try testing.expect(Type.eql(Type.structT(2), Type.structT(2)));
    try testing.expect(!Type.eql(Type.structT(2), Type.structT(3)));
    try testing.expect(Type.eql(Type.enumT(1), Type.enumT(1)));
    try testing.expect(!Type.eql(Type.enumT(1), Type.enumT(2)));
    // same id, different kind: not equal (a struct id is not an enum id).
    try testing.expect(!Type.eql(Type.structT(0), Type.enumT(0)));
}

test "engine: directly-recursive struct is poisoned with one diagnostic" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    // struct N { n: N } — its sole field names struct id 0 (itself).
    const f = try h.field("n", Type.structT(0));
    const id = try h.struct_("N", &.{f});
    try layoutStruct(h.env(), id);

    const s = h.structs.items[0];
    try testing.expect(s.poisoned);
    try testing.expectEqual(@as(u32, 0), s.size);
    try testing.expectEqual(LayoutState.done, s.state);
    try testing.expectEqual(@as(usize, 1), countDiag(&h, "recursive type 'N' has infinite size"));
}

test "engine: mutually-recursive struct cycle poisons the requester, one diagnostic" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    // struct A { b: B }  struct B { a: A } — A=0, B=1.
    const fa = try h.field("b", Type.structT(1));
    const a = try h.struct_("A", &.{fa});
    const fb = try h.field("a", Type.structT(0));
    const b = try h.struct_("B", &.{fb});
    // Drive in id order, exactly as run()/runGraph() do: layoutStruct(A) recurses
    // into B, whose `a` field closes the cycle while A is still `.laying`.
    try layoutStruct(h.env(), a);
    try layoutStruct(h.env(), b);

    // Exactly ONE recursion diagnostic, naming the REQUESTER mid-lay at the cycle
    // edge — that is B (its field `a` is the one pointing back at the laying A).
    try testing.expectEqual(@as(usize, 1), countDiag(&h, "recursive type 'B' has infinite size"));
    try testing.expectEqual(@as(usize, 1), h.diags.items.len); // no second diag
    // B (the requester) is poisoned + size 0; A (the referent) completes cleanly.
    try testing.expect(h.structs.items[1].poisoned);
    try testing.expectEqual(@as(u32, 0), h.structs.items[1].size);
    try testing.expect(!h.structs.items[0].poisoned);
    try testing.expectEqual(LayoutState.done, h.structs.items[0].state);
    try testing.expectEqual(LayoutState.done, h.structs.items[1].state);
}

test "engine: recursive enum is poisoned, and the fingerprint sentinel key fires" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    // enum E { A(E), B } — a tuple variant whose payload names enum id 0.
    const va = try h.vTuple("A", &.{Type.enumT(0)});
    const vb = try h.vUnit("B");
    const id = try h.enum_("E", &.{ va, vb });
    try layoutEnum(h.env(), id);

    const e = h.enums.items[0];
    try testing.expect(e.poisoned);
    try testing.expectEqual(@as(u32, 0), e.size);
    try testing.expectEqual(@as(usize, 1), countDiag(&h, "recursive type 'E' has infinite size"));
    // A poisoned-or-unfinished enum layout is the `<rec>` sentinel contract the
    // content fingerprint relies on; guard that contract here directly.
    try testing.expect(e.state != .done or e.poisoned);
}

test "engine: clean struct layout math — 3 int fields are 0/8/16 size 24 align 8" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    const fa = try h.field("a", Type.int);
    const fb = try h.field("b", Type.int);
    const fc = try h.field("c", Type.int);
    const id = try h.struct_("V3", &.{ fa, fb, fc });
    try layoutStruct(h.env(), id);

    const s = h.structs.items[0];
    try testing.expect(!s.poisoned);
    try testing.expectEqual(@as(usize, 3), s.offsets.len);
    try testing.expectEqual(@as(u32, 0), s.offsets[0]);
    try testing.expectEqual(@as(u32, 8), s.offsets[1]);
    try testing.expectEqual(@as(u32, 16), s.offsets[2]);
    try testing.expectEqual(@as(u32, 24), s.size);
    try testing.expectEqual(@as(u32, 8), s.@"align");
}

test "engine: clean enum layout math — struct variant + tag is size 32, payload at 8" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    // enum E { A { p:int, q:int, r:int }, B(int) }
    const va = try h.vStruct("A", &.{ .{ "p", Type.int }, .{ "q", Type.int }, .{ "r", Type.int } });
    const vb = try h.vTuple("B", &.{Type.int});
    const id = try h.enum_("E", &.{ va, vb });
    try layoutEnum(h.env(), id);

    const e = h.enums.items[0];
    try testing.expect(!e.poisoned);
    try testing.expectEqual(@as(u32, 8), e.tag_size);
    try testing.expectEqual(@as(u32, 8), e.payload_off);
    try testing.expectEqual(@as(u32, 32), e.size);
    try testing.expectEqual(@as(usize, 2), e.variants.len);
    // Struct-variant A's payload-local offsets.
    try testing.expectEqual(@as(u32, 0), e.variants[0].offsets[0]);
    try testing.expectEqual(@as(u32, 8), e.variants[0].offsets[1]);
    try testing.expectEqual(@as(u32, 16), e.variants[0].offsets[2]);
}

test "engine: small enum {C(int), N} is 16 bytes" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    const vc = try h.vTuple("C", &.{Type.int});
    const vn = try h.vUnit("N");
    const id = try h.enum_("E", &.{ vc, vn });
    try layoutEnum(h.env(), id);
    try testing.expectEqual(@as(u32, 16), h.enums.items[0].size);
}

test "engine: empty struct and empty enum are poisoned with the right text" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    const s = try h.struct_("X", &.{});
    const e = try h.enum_("Y", &.{});
    try layoutStruct(h.env(), s);
    try layoutEnum(h.env(), e);

    try testing.expect(h.structs.items[0].poisoned);
    try testing.expectEqual(@as(u32, 0), h.structs.items[0].size);
    try testing.expectEqual(@as(usize, 1), countDiag(&h, "empty struct 'X' is not allowed"));
    try testing.expect(h.enums.items[0].poisoned);
    try testing.expectEqual(@as(u32, 0), h.enums.items[0].size);
    try testing.expectEqual(@as(usize, 1), countDiag(&h, "empty enum 'Y' is not allowed"));
}

test "engine: unit-typed field/payload is rejected" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    const f = try h.field("u", Type.unit);
    const s = try h.struct_("S", &.{f});
    try layoutStruct(h.env(), s);
    try testing.expect(h.structs.items[0].poisoned);
    try testing.expectEqual(@as(usize, 1), countDiag(&h, "field 'u' cannot have type ()"));

    const vp = try h.vTuple("P", &.{Type.unit});
    const e = try h.enum_("E", &.{vp});
    try layoutEnum(h.env(), e);
    try testing.expect(h.enums.items[0].poisoned);
    try testing.expectEqual(@as(usize, 1), countDiag(&h, "variant 'P' payload cannot have type ()"));
}

test "engine: snapshot owns its strings and omits poison; poisoned struct snapshots size 0" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    const fa = try h.field("a", Type.int);
    const id = try h.struct_("V1", &.{fa});
    try layoutStruct(h.env(), id);

    const layouts = try snapshotLayouts(testing.allocator, h.structs.items);
    defer freeLayouts(testing.allocator, layouts);
    try testing.expectEqual(@as(u32, 8), layouts[0].size);
    // The snapshot dupes name/field_names rather than aliasing the syms.
    try testing.expect(layouts[0].name.ptr != h.structs.items[0].name.ptr);
    try testing.expect(layouts[0].field_names[0].ptr != h.structs.items[0].field_names[0].ptr);
    try testing.expectEqualStrings("V1", layouts[0].name);
    // `Layout` has no `poisoned` field — poison stays checker-side by construction.
    try testing.expect(!@hasField(Layout, "poisoned"));

    // A poisoned (recursive) struct snapshots to size 0.
    var h2 = Harness.init(testing.allocator);
    defer h2.deinit();
    const f = try h2.field("n", Type.structT(0));
    const rid = try h2.struct_("N", &.{f});
    try layoutStruct(h2.env(), rid);
    const ls = try snapshotLayouts(testing.allocator, h2.structs.items);
    defer freeLayouts(testing.allocator, ls);
    try testing.expectEqual(@as(u32, 0), ls[0].size);
}

test "engine: memoization — a second layoutStruct is a no-op (state stays done)" {
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    const fa = try h.field("a", Type.int);
    const id = try h.struct_("V1", &.{fa});
    try layoutStruct(h.env(), id);
    const size_before = h.structs.items[0].size;
    const fields_ptr = h.structs.items[0].field_types.ptr;
    try layoutStruct(h.env(), id); // must early-return, not re-alloc
    try testing.expectEqual(size_before, h.structs.items[0].size);
    try testing.expectEqual(fields_ptr, h.structs.items[0].field_types.ptr);
}
