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
///
/// `type_var` (APPENDED — ordinals are frozen) is a CHECK-TIME type variable:
/// it reuses `struct_id` as the generic-parameter ordinal and exists only inside a
/// generic template's decoded signature. It is substituted away to a concrete kind
/// during the serial monomorphization tail, BEFORE any `node_types` slot is frozen,
/// so it never reaches lower/codegen/layout (Debug-asserted there).
///
/// `app` (APPENDED — ordinals frozen) is a CHECK-TIME composite type
/// `Ctor[args..]` (a generic-struct application like `Box[int]`): it reuses
/// `struct_id` as an index into the interned composite table (`symbols/Composite.zig`).
/// Every reachable ground `app` is REIFIED to a fresh ordinary `struct_id` (its
/// `Layout` registered on the live tables) during the serial monomorphization tail,
/// BEFORE the layout snapshot, so codegen/fingerprint/cache see only plain `structT`
/// and an `app` never reaches lower/codegen (Debug-asserted there).
pub const Kind = enum(u8) { invalid, unit, int, bool, str, never, @"struct", @"enum", type_var, app };

/// The width class of an integer type. `plat` is the platform-word `int`/`uint`
/// (currently 64-bit); the numbered classes are the fixed-width `int8..int64`.
/// `enum(u3)` so it fits the 8-bit `IntDesc` packed struct with room to spare.
pub const IntWidth = enum(u3) { plat, w8, w16, w32, w64 };

/// The sign+width descriptor an `.int`-kind `Type` carries in its otherwise-spare
/// pad byte. Defaults to signed platform `int`, so `Type.int` (`.{ .kind = .int }`)
/// stays byte-identical to before this field existed. `packed struct(u8)` keeps
/// `@sizeOf(Type)==12` and makes the descriptor a single deterministic byte for the
/// fingerprint/dedup/order serializers.
pub const IntDesc = packed struct(u8) { signed: bool = true, width: IntWidth = .plat, _pad: u4 = 0 };

/// A type. A byte-foldable struct (not a tagged union) so it preserves `@memset`,
/// `node_types` triviality, and a stable fingerprint basis. A `@"struct"` kind
/// carries an index into the struct table; a `@"enum"` kind an index into the
/// parallel enum table; all other kinds leave both `no_struct`.
pub const Type = struct {
    kind: Kind,
    int_desc: IntDesc = .{},
    struct_id: u32 = no_struct,
    enum_id: u32 = no_struct,

    pub const no_struct: u32 = std.math.maxInt(u32);

    pub const invalid: Type = .{ .kind = .invalid };
    pub const unit: Type = .{ .kind = .unit };
    pub const int: Type = .{ .kind = .int };
    pub const @"bool": Type = .{ .kind = .bool };
    pub const str: Type = .{ .kind = .str };
    pub const never: Type = .{ .kind = .never };

    pub const uint: Type = .{ .kind = .int, .int_desc = .{ .signed = false } };
    pub const int8: Type = .{ .kind = .int, .int_desc = .{ .width = .w8 } };
    pub const int16: Type = .{ .kind = .int, .int_desc = .{ .width = .w16 } };
    pub const int32: Type = .{ .kind = .int, .int_desc = .{ .width = .w32 } };
    pub const int64: Type = .{ .kind = .int, .int_desc = .{ .width = .w64 } };
    pub const uint8: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w8 } };
    pub const uint16: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w16 } };
    pub const uint32: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w32 } };
    pub const uint64: Type = .{ .kind = .int, .int_desc = .{ .signed = false, .width = .w64 } };

    pub fn structT(id: u32) Type {
        return .{ .kind = .@"struct", .struct_id = id };
    }

    pub fn enumT(id: u32) Type {
        return .{ .kind = .@"enum", .enum_id = id };
    }

    /// A check-time type variable for generic-parameter `ordinal`. Reuses
    /// `struct_id` as the ordinal — NO widening, `@sizeOf(Type)` is unchanged.
    pub fn typeVar(ordinal: u32) Type {
        return .{ .kind = .type_var, .struct_id = ordinal };
    }

    pub fn isTypeVar(t: Type) bool {
        return t.kind == .type_var;
    }

    /// The generic-parameter ordinal of a `type_var` (read of the reused id field).
    pub fn typeVarOrd(t: Type) u32 {
        return t.struct_id;
    }

    /// A check-time composite `Ctor[args..]` type. Reuses `struct_id` as the
    /// interned composite-table index (`symbols/Composite.zig`) — NO widening.
    pub fn app(idx: u32) Type {
        return .{ .kind = .app, .struct_id = idx };
    }

    pub fn isApp(t: Type) bool {
        return t.kind == .app;
    }

    /// The composite-table index of an `app` (read of the reused id field). Two
    /// structurally-equal `app`s share an index within a run (content-addressed
    /// interning), so `eql`-by-id is correct.
    pub fn appIdx(t: Type) u32 {
        return t.struct_id;
    }

    pub fn eql(a: Type, b: Type) bool {
        return a.kind == b.kind and
            (a.kind != .@"struct" or a.struct_id == b.struct_id) and
            (a.kind != .@"enum" or a.enum_id == b.enum_id) and
            (a.kind != .type_var or a.struct_id == b.struct_id) and
            (a.kind != .app or a.struct_id == b.struct_id) and
            (a.kind != .int or @as(u8, @bitCast(a.int_desc)) == @as(u8, @bitCast(b.int_desc)));
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

    pub fn isInteger(t: Type) bool {
        return t.kind == .int;
    }

    pub fn isSigned(t: Type) bool {
        return t.kind == .int and t.int_desc.signed;
    }

    pub fn isPlatformInt(t: Type) bool {
        return t.kind == .int and t.int_desc.width == .plat;
    }

    /// The source spelling of an integer type (`int`/`uint`/`int8`.../`uint64`).
    /// Exhaustive over `IntWidth` (no `else` arm) so a new width class is a build
    /// error here rather than a silently-wrong name. Static strings — no alloc, so a
    /// borrowing accessor (`refs.typeName`, `Mono.mangle`) can return the slice directly.
    pub fn intName(t: Type) []const u8 {
        return switch (t.int_desc.width) {
            .plat => if (t.int_desc.signed) "int" else "uint",
            .w8 => if (t.int_desc.signed) "int8" else "uint8",
            .w16 => if (t.int_desc.signed) "int16" else "uint16",
            .w32 => if (t.int_desc.signed) "int32" else "uint32",
            .w64 => if (t.int_desc.signed) "int64" else "uint64",
        };
    }

    /// Append this type's flat leaf key — `(kind, struct_id, enum_id)` at fixed
    /// width — to `buf`. The single serializer every dedup/structural key routes
    /// through (`Composite.writeFlatKey`/`writeStructuralKey`, `Mono.writeKey`), so
    /// the byte encoding stays identical across them.
    pub fn appendKeyBytes(t: Type, gpa: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        try buf.append(gpa, @intFromEnum(t.kind));
        // The sign/width descriptor only for `.int` (self-delimiting off the leading
        // kind byte), so every non-int key stays byte-identical while `int8`/`int64`
        // never fold to one interned/dedup/mangle key.
        if (t.kind == .int) try buf.append(gpa, @as(u8, @bitCast(t.int_desc)));
        var w: [4]u8 = undefined;
        std.mem.writeInt(u32, &w, t.struct_id, .little);
        try buf.appendSlice(gpa, &w);
        std.mem.writeInt(u32, &w, t.enum_id, .little);
        try buf.appendSlice(gpa, &w);
    }

    /// The canonical leaf ordering over `(kind, struct_id, enum_id)` — the tie-break
    /// spine of `Mono.lessThan`'s total instance order.
    pub fn orderLeaf(a: Type, b: Type) std.math.Order {
        if (a.kind != b.kind) return std.math.order(@intFromEnum(a.kind), @intFromEnum(b.kind));
        // Two integer widths sharing the `.int` kind tie-break on the descriptor byte,
        // so `Mono.lessThan` stays a total order across widths (else the `-jN` mono sort
        // is nondeterministic).
        if (a.kind == .int and @as(u8, @bitCast(a.int_desc)) != @as(u8, @bitCast(b.int_desc)))
            return std.math.order(@as(u8, @bitCast(a.int_desc)), @as(u8, @bitCast(b.int_desc)));
        if (a.struct_id != b.struct_id) return std.math.order(a.struct_id, b.struct_id);
        return std.math.order(a.enum_id, b.enum_id);
    }

    comptime {
        // The byte-foldable `Type` never widens (locked decision): `type_var` reuses
        // `struct_id` as the ordinal, so appending the kind does not grow the struct.
        std.debug.assert(@sizeOf(Type) == 12);
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

/// The prelude generic-enum family a reified concrete instance belongs to, or
/// `.none` for any user/ordinary enum. Set on a reified `Option[T]`/`Result[T,E]`
/// instance (keyed off the prelude template id in `reifyAppToEnum`, NOT the name — a
/// user `enum Option` mangles to the same `Option$int` yet is a distinct template) and
/// copied through to its `EnumLayout`, so `lower` can recognize the compiler-provided
/// `is_some`/`unwrap`/… inherent methods per-instance and inline the tag test / payload
/// load. Never serialized (layouts are recomputed each typecheck), so adding it needs no
/// content-cache version bump.
pub const NativeEnumFamily = enum(u8) { none, option, result };

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
    native_family: NativeEnumFamily = .none,
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
    /// A generic TEMPLATE `struct Box[T] { .. }`: its `field_types` carry
    /// `type_var`/`App` PATTERNS, never a value type, so Phase 0b must SKIP laying it
    /// out (its type-var fields have no ABI). Only its reified concrete instances get
    /// a layout. `generic_params` are the ordered param NAMES (borrowed source slices;
    /// the outer array is owned by the checker's `structs` table).
    is_generic: bool = false,
    generic_params: []const []const u8 = &.{},
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
    /// A generic TEMPLATE `enum Either[L,R] { .. }`: its variants' payload
    /// `field_types` carry `type_var`/`App` PATTERNS, never a value type, so Phase 0b
    /// must SKIP laying it out (its type-var payloads have no ABI). Only its reified
    /// concrete instances get a layout. `generic_params` are the ordered param NAMES
    /// (borrowed source slices; the outer array is owned by the checker's `enums` table).
    /// Mirrors `StructSym.is_generic`/`generic_params`.
    is_generic: bool = false,
    generic_params: []const []const u8 = &.{},
    /// See `NativeEnumFamily`: `.option`/`.result` on a reified prelude instance, else
    /// `.none`. Set in `reifyAppToEnum`; copied into the snapshot `EnumLayout`.
    native_family: NativeEnumFamily = .none,
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
    /// Reify the interned composite `App` at `app_idx` to a fresh concrete type
    /// (registering its `Layout`/`EnumLayout` on the live tables, memoized) and return
    /// it. Used by `layoutStruct`/`layoutEnum` to turn a concrete generic
    /// field/payload type `b: Box[int]` / `e: Either[int,bool]` (which decodes to an
    /// `App`) into a plain `structT`/`enumT` BEFORE it is stored/laid out, so no `App`
    /// ever lands in a laid aggregate's `field_types`. Returns a `Type` (not a bare
    /// `struct_id`) so a generic-ENUM field/payload reifies to an `enumT`.
    reifyApp: *const fn (ctx: *anyopaque, app_idx: u32) error{OutOfMemory}!Type,
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
    const sizing = try env.gpa.alloc(Type, n);
    defer env.gpa.free(sizing);

    var unit_poison = false;
    for (field_nodes, 0..) |field_idx, i| {
        const field = tree.nodes[field_idx.int()];
        names[i] = env.nameText(env.ctx, field.main_token);
        const fty = env.typeFromNode(env.ctx, field.lhs);
        types[i] = fty;
        // A concrete generic-aggregate field `b: Box[int]` / `e: Either[int,bool]`
        // decodes to a composite `App`. Reify it on-demand to get the size/align
        // for offsets, but KEEP the `App` in `field_types` so Pass-C field/construction
        // checks compare it against the (same-interned) `App` a value expression produces
        // — the mono-tail rewrite turns both into the reified `structT`/`enumT` before
        // the snapshot. `reifyApp` returns the concrete type directly.
        const size_ty: Type = if (fty.isApp()) try env.reifyApp(env.ctx, fty.appIdx()) else fty;
        if (size_ty.kind == .unit) {
            try env.emitUnitField(env.ctx, env.byteOf(env.ctx, field.main_token), names[i]);
            unit_poison = true;
        }
        sizing[i] = size_ty;
    }

    const acc = try accumulateOffsets(env, sizing, offsets, env.byteOf(env.ctx, decl.main_token), env.structs.items[id].name);
    const poisoned = acc.poisoned or unit_poison or empty_poison;

    env.structs.items[id].field_names = names;
    env.structs.items[id].field_types = types;
    env.structs.items[id].offsets = offsets;
    env.structs.items[id].@"align" = acc.@"align";
    env.structs.items[id].size = if (poisoned) 0 else acc.size;
    env.structs.items[id].poisoned = poisoned;
    env.structs.items[id].state = .done;
}

/// Lay out a REIFIED generic-struct instance whose `field_names`/`field_types`
/// are ALREADY populated (by the monomorphization tail's substitution) — there is no
/// decl in the tree to read fields from. Computes offsets/size/align + poison over the
/// stored `field_types` via `layoutReferent`, using the SAME per-field offset formula
/// as `layoutStruct` (roundUp per field align; size = roundUp(total, max_align)), so
/// the reg-pair↔indirect ABI boundary is byte-identical to a hand-written struct.
/// The field types are concrete by construction (type-args are checked to be concrete
/// value types; nested `App` fields were reified to `structT` by `substReify`), so the
/// empty/unit/`App` diagnostic paths cannot fire here; any residual `unit`/`invalid`
/// is sized to 0 defensively (offsets stay well-defined).
pub fn layoutReified(env: Env, id: u32) error{OutOfMemory}!void {
    if (env.structs.items[id].state == .done) return;
    env.structs.items[id].state = .laying;
    const prev = env.gphSelect(env.ctx, env.structs.items[id].mod);
    defer _ = env.gphSelect(env.ctx, prev);

    const types = env.structs.items[id].field_types;
    const at = env.byteOf(env.ctx, env.tree(env.ctx).nodes[env.structs.items[id].decl_node.int()].main_token);
    const name = env.structs.items[id].name;

    const offsets = try env.gpa.alloc(u32, types.len);
    errdefer env.gpa.free(offsets);

    const acc = try accumulateOffsets(env, types, offsets, at, name);

    env.structs.items[id].offsets = offsets;
    env.structs.items[id].@"align" = acc.@"align";
    env.structs.items[id].size = if (acc.poisoned) 0 else acc.size;
    env.structs.items[id].poisoned = acc.poisoned;
    env.structs.items[id].state = .done;
}

/// Lay out a REIFIED generic-enum instance whose `variants` (name/form/
/// field_names/field_types) are ALREADY populated (by the monomorphization tail's
/// substitution) — there is no decl in the tree to read variants from. Computes each
/// variant's payload-local offsets + payload size/align via `layoutReferent`, then the
/// aggregate tag/payload_off/size/align, using the SAME formula as `layoutEnum` (8-byte
/// tag at 0; payload sized to the largest variant at `payload_off`), so the ABI is
/// byte-identical to a hand-written enum. The payload field types are concrete by
/// construction (nested `App`s were reified to `structT`/`enumT` by `substReify`), so
/// the empty/unit/`App` diagnostic paths cannot fire here; any residual `unit`/`invalid`
/// sizes to 0 defensively (offsets stay well-defined). Mirrors `layoutReified`.
pub fn layoutReifiedEnum(env: Env, id: u32) error{OutOfMemory}!void {
    if (env.enums.items[id].state == .done) return;
    env.enums.items[id].state = .laying;
    const prev = env.gphSelect(env.ctx, env.enums.items[id].mod);
    defer _ = env.gphSelect(env.ctx, prev);

    // A reified prelude-enum instance (Option$int/Result$int$str) inherits decl_node ==
    // Ast.none from its AST-less template; anchor the (poison-only) diagnostic at byte 0
    // rather than OOB-derefing the tree on maxInt(u32). A concrete instance never poisons.
    const decl_node = env.enums.items[id].decl_node;
    const at: u32 = if (decl_node == Ast.none) 0 else env.byteOf(env.ctx, env.tree(env.ctx).nodes[decl_node.int()].main_token);
    const name = env.enums.items[id].name;
    const variants = env.enums.items[id].variants;

    var poisoned = false;
    var max_payload_size: u32 = 0;
    var max_payload_align: u32 = 1;
    for (variants) |*v| {
        const foffs = try env.gpa.alloc(u32, v.field_types.len);
        errdefer env.gpa.free(foffs);
        const acc = try accumulatePayload(env, v.field_types, foffs, at, name);
        poisoned = poisoned or acc.poisoned;
        v.offsets = foffs;
        v.payload_size = acc.size;
        v.payload_align = acc.@"align";
        if (acc.size > max_payload_size) max_payload_size = acc.size;
        if (acc.@"align" > max_payload_align) max_payload_align = acc.@"align";
    }

    const tag_size: u32 = 8;
    const payload_off = roundUp(tag_size, max_payload_align);
    const aln = @max(@as(u32, 8), max_payload_align);
    env.enums.items[id].tag_size = tag_size;
    env.enums.items[id].payload_off = payload_off;
    env.enums.items[id].@"align" = aln;
    env.enums.items[id].size = if (poisoned) 0 else roundUp(payload_off + max_payload_size, aln);
    env.enums.items[id].poisoned = poisoned;
    env.enums.items[id].state = .done;
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

const Accum = struct { size: u32, @"align": u32, poisoned: bool };

/// The one per-field ABI offset formula, shared by the direct (`layoutStruct`) and
/// reified (`layoutReified`) struct paths so their reg-pair↔indirect boundary can never
/// drift apart. Each `sizing_types[i]` is placed at the next `roundUp(running, align)`
/// byte into `offsets_out[i]`; the aggregate `size = roundUp(total, max_align)` and
/// `align` fall out. A `laying` struct/enum referent poisons via `layoutReferent`
/// (surfaced in `.poisoned`); a `unit`/`invalid` sizing type contributes nothing. The
/// caller must have already reified any `App` and emitted any unit diagnostic — the
/// kernel is diagnostic-free (bar the recursion diagnostic intrinsic to `layoutReferent`).
fn accumulateOffsets(env: Env, sizing_types: []const Type, offsets_out: []u32, at: u32, requester: []const u8) error{OutOfMemory}!Accum {
    std.debug.assert(offsets_out.len == sizing_types.len);
    var running: u32 = 0;
    var max_align: u32 = 1;
    var poisoned = false;
    for (sizing_types, 0..) |sty, i| {
        var fsize: u32 = 0;
        var falign: u32 = 1;
        if (sty.kind != .invalid and sty.kind != .unit) {
            const sz = try layoutReferent(env, sty, at, requester, &poisoned);
            fsize = sz.size;
            falign = sz.@"align";
        }
        const off = roundUp(running, falign);
        offsets_out[i] = off;
        running = off + fsize;
        if (falign > max_align) max_align = falign;
    }
    return .{ .size = roundUp(running, max_align), .@"align" = max_align, .poisoned = poisoned };
}

/// An enum variant's payload lays out with the exact same formula as struct fields
/// (payload-LOCAL offsets, size = roundUp(total, max_align)); the distinct name keeps the
/// two enum payload callers (`layoutEnum`/`layoutReifiedEnum`) reading naturally while
/// sharing the one kernel.
fn accumulatePayload(env: Env, sizing_types: []const Type, offsets_out: []u32, at: u32, requester: []const u8) error{OutOfMemory}!Accum {
    return accumulateOffsets(env, sizing_types, offsets_out, at, requester);
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

        const sizing = try env.gpa.alloc(Type, np);
        defer env.gpa.free(sizing);
        for (payload_nodes, 0..) |pnode_idx, pi| {
            // A tuple payload node is a type-ref; a struct payload node is a `param`
            // (name + type-ref in lhs).
            const pty: Type = if (is_struct_form) blk: {
                const pnode = tree.nodes[pnode_idx.int()];
                fnames[pi] = env.nameText(env.ctx, pnode.main_token);
                break :blk env.typeFromNode(env.ctx, pnode.lhs);
            } else env.typeFromNode(env.ctx, pnode_idx);
            ftypes[pi] = pty;
            // A concrete generic-aggregate payload `v(Box[int])` / `v(Either[int,bool])`
            // decodes to a composite `App`. Reify it on-demand for its size/align
            // but KEEP the `App` in `field_types` (mirroring layoutStruct) so Pass-C
            // construction checks compare it against the same-interned `App`; the mono-tail
            // rewrite turns it into the reified `structT`/`enumT` before the snapshot, so
            // no `App` survives into the enum layout/fingerprint.
            const size_ty: Type = if (pty.isApp()) try env.reifyApp(env.ctx, pty.appIdx()) else pty;
            if (size_ty.kind == .unit) {
                try env.emitUnitPayload(env.ctx, env.byteOf(env.ctx, vnode.main_token), vname);
                poisoned = true;
            }
            sizing[pi] = size_ty;
        }
        const acc = try accumulatePayload(env, sizing, foffs, env.byteOf(env.ctx, decl.main_token), env.enums.items[id].name);
        poisoned = poisoned or acc.poisoned;
        variants[vi] = .{
            .name = vname,
            .form = form,
            .field_names = fnames,
            .field_types = ftypes,
            .offsets = foffs,
            .payload_size = acc.size,
            .payload_align = acc.@"align",
        };
        vbuilt += 1;
        if (acc.size > max_payload_size) max_payload_size = acc.size;
        if (acc.@"align" > max_payload_align) max_payload_align = acc.@"align";
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
            .native_family = e.native_family,
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
            .reifyApp = stubReifyApp,
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
fn stubReifyApp(_: *anyopaque, _: u32) error{OutOfMemory}!Type {
    // The layout unit tests never build a struct with a concrete `App` field, so this
    // is never invoked; the full reify path is covered end-to-end in types.zig.
    unreachable;
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

test "algebra: type_var round-trips its ordinal and eql is per-ordinal" {
    // The byte-foldable Type never widens: the ordinal rides `struct_id`.
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));
    const t0 = Type.typeVar(0);
    const t1 = Type.typeVar(1);
    try testing.expect(t0.isTypeVar());
    try testing.expectEqual(@as(u32, 0), t0.typeVarOrd());
    try testing.expectEqual(@as(u32, 1), t1.typeVarOrd());
    // Distinct ordinals are distinct types; same ordinal is equal.
    try testing.expect(Type.eql(t0, Type.typeVar(0)));
    try testing.expect(!Type.eql(t0, t1));
    // A type_var is not a struct even though it reuses struct_id (kind discriminates).
    try testing.expect(!Type.eql(t0, Type.structT(0)));
    try testing.expect(!t0.isStruct());
}

test "algebra: app round-trips its composite index and eql is per-index (M4)" {
    // The byte-foldable Type never widens: the composite index rides `struct_id`.
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));
    const a0 = Type.app(0);
    const a1 = Type.app(1);
    try testing.expect(a0.isApp());
    try testing.expectEqual(@as(u32, 0), a0.appIdx());
    try testing.expectEqual(@as(u32, 1), a1.appIdx());
    // Same interned index => eql (structurally-equal Apps share an index); distinct
    // indices are distinct types.
    try testing.expect(Type.eql(a0, Type.app(0)));
    try testing.expect(!Type.eql(a0, a1));
    // An app is neither a struct nor a type_var even though all three reuse struct_id
    // (kind discriminates), so eql-by-id never confuses them.
    try testing.expect(!Type.eql(a0, Type.structT(0)));
    try testing.expect(!Type.eql(a0, Type.typeVar(0)));
    try testing.expect(!a0.isStruct());
    // assignable delegates to eql: same-index apps are assignable, distinct are not.
    try testing.expect(Type.assignable(a0, Type.app(0)));
    try testing.expect(!Type.assignable(a0, a1));
}

test "algebra: integer widths are distinct byte-foldable types (M1)" {
    // The width/sign descriptor rides the otherwise-spare pad byte: no widening.
    try testing.expectEqual(@as(usize, 12), @sizeOf(Type));

    // eql discriminates width AND sign; plain int is unchanged.
    try testing.expect(!Type.eql(Type.int8, Type.int64));
    try testing.expect(!Type.eql(Type.int8, Type.int));
    try testing.expect(!Type.eql(Type.int, Type.uint));
    try testing.expect(Type.eql(Type.int, Type.int));
    try testing.expect(Type.eql(Type.int8, Type.int8));

    // Predicates.
    try testing.expect(Type.int8.isInteger() and Type.int8.isSigned() and !Type.int8.isPlatformInt());
    try testing.expect(Type.uint8.isInteger() and !Type.uint8.isSigned());
    try testing.expect(Type.int.isPlatformInt());
    try testing.expect(!Type.bool.isInteger());

    // Names.
    try testing.expectEqualStrings("int8", Type.int8.intName());
    try testing.expectEqualStrings("uint32", Type.uint32.intName());
    try testing.expectEqualStrings("int", Type.int.intName());
    try testing.expectEqualStrings("uint", Type.uint.intName());

    // appendKeyBytes separates widths but leaves plain int's key at one byte + ids.
    const gpa = testing.allocator;
    var k8: std.ArrayList(u8) = .empty;
    defer k8.deinit(gpa);
    var k64: std.ArrayList(u8) = .empty;
    defer k64.deinit(gpa);
    var ki: std.ArrayList(u8) = .empty;
    defer ki.deinit(gpa);
    try Type.int8.appendKeyBytes(gpa, &k8);
    try Type.int64.appendKeyBytes(gpa, &k64);
    try Type.int.appendKeyBytes(gpa, &ki);
    try testing.expect(!std.mem.eql(u8, k8.items, k64.items));
    try testing.expect(!std.mem.eql(u8, k8.items, ki.items));

    // orderLeaf is a strict total order across widths.
    try testing.expect(Type.orderLeaf(Type.int8, Type.int64) != .eq);
    try testing.expect(Type.orderLeaf(Type.int8, Type.int64) == Type.orderLeaf(Type.int8, Type.int64));
}

test "engine: layoutReified matches a hand-written struct's offsets/size/align (M4)" {
    // A reified generic instance is laid out from pre-populated field_types (no decl
    // fields in the tree); its ABI math must equal the tree-driven `layoutStruct`.
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    // A "template" decl node just to supply a decl_node/main_token for the byte offset;
    // its tree fields are NOT read by layoutReified.
    const tmpl = try h.struct_("Box", &.{try h.field("v", Type.int)});
    _ = tmpl;
    // Reified Box$int: one int field, populated directly.
    const names = try h.gpa.alloc([]const u8, 1);
    names[0] = "v";
    const types = try h.gpa.alloc(Type, 1);
    types[0] = Type.int;
    const decl = h.structs.items[0].decl_node;
    try h.structs.append(h.gpa, .{ .decl_node = decl, .name = "Box$int", .field_names = names, .field_types = types });
    const rid: u32 = @intCast(h.structs.items.len - 1);
    try layoutReified(h.env(), rid);
    const s = h.structs.items[rid];
    try testing.expect(!s.poisoned);
    try testing.expectEqual(@as(usize, 1), s.offsets.len);
    try testing.expectEqual(@as(u32, 0), s.offsets[0]);
    try testing.expectEqual(@as(u32, 8), s.size);
    try testing.expectEqual(@as(u32, 8), s.@"align");

    // Two int fields => offsets 0/8, size 16, align 8 — the reg-pair ABI boundary.
    const names2 = try h.gpa.alloc([]const u8, 2);
    names2[0] = "a";
    names2[1] = "b";
    const types2 = try h.gpa.alloc(Type, 2);
    types2[0] = Type.int;
    types2[1] = Type.int;
    try h.structs.append(h.gpa, .{ .decl_node = decl, .name = "Pair$int$int", .field_names = names2, .field_types = types2 });
    const rid2: u32 = @intCast(h.structs.items.len - 1);
    try layoutReified(h.env(), rid2);
    const s2 = h.structs.items[rid2];
    try testing.expectEqual(@as(u32, 0), s2.offsets[0]);
    try testing.expectEqual(@as(u32, 8), s2.offsets[1]);
    try testing.expectEqual(@as(u32, 16), s2.size);
}

test "engine: layoutReifiedEnum matches a hand-written enum's tag/payload_off/size (M6)" {
    // A reified generic-enum instance is laid out from PRE-POPULATED concrete variants
    // (no decl variants in the tree); its ABI math must equal tree-driven `layoutEnum`.
    var h = Harness.init(testing.allocator);
    defer h.deinit();
    // A decl node just to supply decl_node/main_token for the byte offset; its tree
    // variants are NOT read by layoutReifiedEnum.
    const tmpl = try h.enum_("L", &.{try h.vUnit("nil")});
    const decl = h.enums.items[tmpl].decl_node;

    // Reified `{ c(int), n }`: variant c is a tuple carrying one int; n is unit.
    const variants = try h.gpa.alloc(VariantSym, 2);
    const c_names = try h.gpa.alloc([]const u8, 0);
    const c_types = try h.gpa.alloc(Type, 1);
    c_types[0] = Type.int;
    variants[0] = .{ .name = "c", .form = .tuple, .field_names = c_names, .field_types = c_types };
    const n_names = try h.gpa.alloc([]const u8, 0);
    const n_types = try h.gpa.alloc(Type, 0);
    variants[1] = .{ .name = "n", .form = .unit, .field_names = n_names, .field_types = n_types };
    try h.enums.append(h.gpa, .{ .decl_node = decl, .name = "L$int", .variants = variants });
    const rid: u32 = @intCast(h.enums.items.len - 1);
    try layoutReifiedEnum(h.env(), rid);

    const e = h.enums.items[rid];
    try testing.expect(!e.poisoned);
    try testing.expectEqual(@as(u32, 8), e.tag_size);
    try testing.expectEqual(@as(u32, 8), e.payload_off);
    try testing.expectEqual(@as(u32, 16), e.size);
    try testing.expectEqual(@as(u32, 8), e.@"align");
    // c's single int payload lands at payload-local offset 0.
    try testing.expectEqual(@as(usize, 1), e.variants[0].offsets.len);
    try testing.expectEqual(@as(u32, 0), e.variants[0].offsets[0]);
    // n (unit) has no payload offsets.
    try testing.expectEqual(@as(usize, 0), e.variants[1].offsets.len);
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
