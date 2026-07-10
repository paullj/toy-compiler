//! Native registration of the implicit PRELUDE, mirroring how `print` is a
//! synthesized compiler entity with no module-graph / content-fingerprint surface. Runs
//! SERIALLY at the head of Phase 0c, before the per-module `registerProtocols` loop, so
//! the prelude protocol ids are a pure function of source (no hashmap/thread input).
//!
//! The `ProtoSpec` table drives the whole protocol set in ONE append order that is
//! LOAD-BEARING: `Eq=0, Ord=1, Add=2, Sub=3, Mul=4, Div=5, Hash=6, Display=7, From=8,
//! Into=9, TryInto=10` (`ConvErr` enum appended after `Result`).
//! Codegen, operator lowering and the fingerprint fold all resolve witnesses by these
//! ids, so the order must never shift. `Ordering` is built BEFORE the loop because
//! `Ord.cmp`'s return type references its enum id; `Option`/`Result` are built AFTER so
//! their enum ids fall past every user enum + `Ordering`. Every owned slice is
//! `gpa`-allocated (never a comptime literal) so the checker's teardown free loop treats
//! prelude and user protocols/enums uniformly.

const std = @import("std");

const Ast = @import("../ast/Ast.zig");
const LayoutEngine = @import("../layout/Engine.zig");
const Typecheck = @import("../types.zig");
const Type = Typecheck.Type;
const ProtocolSym = Typecheck.ProtocolSym;
const Conformance = Typecheck.Conformance;
const Prelude = Typecheck.Prelude;
const VariantSym = LayoutEngine.VariantSym;
const EnumSym = LayoutEngine.EnumSym;
const StructSym = LayoutEngine.StructSym;
const ModuleCtx = Typecheck.GraphCtx.ModuleCtx;

/// Which `Prelude.protocols` field a spec records its assigned id into.
const Slot = enum { eq, ord, add, sub, mul, div, hash, display, from, into, try_into };

/// A prelude protocol's return-type shape. `ordering` is resolved to `Type.enumT(id)`
/// against the runtime `Ordering` id; the rest are concrete or `Self` (`type_var(0)`).
/// `dst_t` is the protocol's first generic arg (`Dst`, `type_var(1)`).
const Ret = enum { self_t, bool_t, int_t, unit_t, ordering_t, dst_t };

const ProtoSpec = struct {
    slot: Slot,
    name: []const u8,
    method: []const u8,
    /// The single method's param types, `type_var`-encoded (`type_var(0)` = Self,
    /// `type_var(1)` = the protocol's first generic arg). Duped per protocol at build.
    params: []const Type,
    ret: Ret,
    /// The builtin scalar receivers that conform out of the box, in FIXED order.
    conf: []const Type = &.{},
    /// The protocol's generic params (only `From[Src]`); duped per protocol at build.
    generic_params: []const []const u8 = &.{},
};

const homogeneous: []const Type = &.{ Type.typeVar(0), Type.typeVar(0) };
const self_only: []const Type = &.{Type.typeVar(0)};
const from_params: []const Type = &.{Type.typeVar(1)};

// str/unit join int/bool for Eq/Hash/Display so the operator's uniform conformance check
// covers every scalar; arithmetic is int-ONLY (str concat allocates, deferred). Ord omits
// unit (no ordering). From registers NO builtin conformance (always needs an explicit impl).
const specs = [_]ProtoSpec{
    .{ .slot = .eq, .name = "Eq", .method = "eq", .params = homogeneous, .ret = .bool_t, .conf = &.{ Type.int, Type.bool, Type.str, Type.unit } },
    .{ .slot = .ord, .name = "Ord", .method = "cmp", .params = homogeneous, .ret = .ordering_t, .conf = &.{ Type.int, Type.str, Type.bool } },
    .{ .slot = .add, .name = "Add", .method = "add", .params = homogeneous, .ret = .self_t, .conf = &.{Type.int} },
    .{ .slot = .sub, .name = "Sub", .method = "sub", .params = homogeneous, .ret = .self_t, .conf = &.{Type.int} },
    .{ .slot = .mul, .name = "Mul", .method = "mul", .params = homogeneous, .ret = .self_t, .conf = &.{Type.int} },
    .{ .slot = .div, .name = "Div", .method = "div", .params = homogeneous, .ret = .self_t, .conf = &.{Type.int} },
    .{ .slot = .hash, .name = "Hash", .method = "hash", .params = self_only, .ret = .int_t, .conf = &.{ Type.int, Type.bool, Type.str, Type.unit } },
    .{ .slot = .display, .name = "Display", .method = "display", .params = self_only, .ret = .unit_t, .conf = &.{ Type.int, Type.bool, Type.str, Type.unit } },
    .{ .slot = .from, .name = "From", .method = "from", .params = from_params, .ret = .self_t, .generic_params = &.{"Src"} },
    .{ .slot = .into, .name = "Into", .method = "into", .params = self_only, .ret = .dst_t, .generic_params = &.{"Dst"} },
    // `TryInto.try_into`'s declared ret `dst_t` (=`Dst`) is a deliberate PLACEHOLDER: the
    // true `Result[Dst, ConvErr]` cannot be minted at registration (no `Composite`
    // interner here) and is synthesized per call site (BodyChecker). M3 ships no user
    // `impl .. has TryInto`, so `method_rets[try_into]` is never read; a future milestone
    // wiring `try_into` through the explicit-args/bound path MUST fix this first.
    .{ .slot = .try_into, .name = "TryInto", .method = "try_into", .params = self_only, .ret = .dst_t, .generic_params = &.{"Dst"} },
};

/// Register the whole prelude into the checker's global tables and return the discovered
/// ids. Takes the ArrayLists + module contexts directly (not the `Typecheck`) so it has no
/// dependency on the checker's private surface.
pub fn register(
    gpa: std.mem.Allocator,
    protocols: *std.ArrayList(ProtocolSym),
    conformances: *std.ArrayList(Conformance),
    enums: *std.ArrayList(EnumSym),
    structs: *std.ArrayList(StructSym),
    mods: []ModuleCtx,
) !Prelude {
    var prelude: Prelude = .{};

    const ordering_id = try registerOrdering(gpa, enums, mods);
    prelude.ordering_enum = ordering_id;

    prelude.char_struct = try registerChar(gpa, structs, mods);

    for (specs) |spec| {
        const methods = try gpa.alloc([]const u8, 1);
        methods[0] = spec.method;
        const params = try gpa.alloc([]const Type, 1);
        params[0] = try gpa.dupe(Type, spec.params);
        const rets = try gpa.alloc(Type, 1);
        rets[0] = switch (spec.ret) {
            .self_t => Type.typeVar(0),
            .bool_t => Type.bool,
            .int_t => Type.int,
            .unit_t => Type.unit,
            .ordering_t => Type.enumT(ordering_id),
            .dst_t => Type.typeVar(1),
        };
        const gparams: []const []const u8 = if (spec.generic_params.len > 0)
            try gpa.dupe([]const u8, spec.generic_params)
        else
            &.{};

        const pid: u32 = @intCast(protocols.items.len);
        setSlot(&prelude, spec.slot, pid);
        try protocols.append(gpa, .{
            .name = spec.name,
            .mod = 0,
            .pub_export = true,
            .decl_node = Ast.none,
            .methods = methods,
            .method_params = params,
            .method_rets = rets,
            .generic_params = gparams,
        });
        for (spec.conf) |recv| try conformances.append(gpa, .{ .protocol = pid, .recv = recv });
    }

    // char's Display is a hand-written UTF-8 ENCODER (lower.lowerCharDisplay), NOT a
    // structural `char(65)` derive. An explicit conformance row (as int/bool/str/unit have,
    // carrying no Method) makes `conformsTo(char, Display)` .direct, so BodyChecker records
    // no Display derive request and the structural tuple unit is never synthesized. Appended
    // after the specs loop so no protocol/struct/enum id shifts (conformances key on
    // (protocol, recv) via Type.eql, never by index).
    try conformances.append(gpa, .{
        .protocol = prelude.protocols.display.?,
        .recv = Type.structT(prelude.char_struct.?),
    });

    try registerOptionResult(gpa, enums, mods, &prelude);
    prelude.conv_err_enum = try registerConvErr(gpa, enums);
    return prelude;
}

fn setSlot(p: *Prelude, slot: Slot, id: u32) void {
    switch (slot) {
        .eq => p.protocols.eq = id,
        .ord => p.protocols.ord = id,
        .add => p.protocols.add = id,
        .sub => p.protocols.sub = id,
        .mul => p.protocols.mul = id,
        .div => p.protocols.div = id,
        .hash => p.protocols.hash = id,
        .display => p.protocols.display = id,
        .from => p.protocols.from = id,
        .into => p.protocols.into = id,
        .try_into => p.protocols.try_into = id,
    }
}

/// Native `enum ConvErr { out_of_range }`: the checker-internal error payload of a
/// `try_into`'s synthesized `Result[Dst, ConvErr]`. Hand-laid-out (`state == .done`) like
/// `Ordering`. Appended AFTER `Result` so no existing enum id shifts, and deliberately
/// NOT injected into any module's `enum_ids` — it is never named in source, only minted
/// by the `try_into` recognizer, so a user `enum ConvErr` never collides with it.
fn registerConvErr(gpa: std.mem.Allocator, enums: *std.ArrayList(EnumSym)) !u32 {
    return appendNativeUnitEnum(gpa, enums, "ConvErr", &.{"out_of_range"});
}

/// Append a native, AST-less all-unit-variant enum (`state == .done`, `size == 8`) whose
/// tag is the variant DECL INDEX. The shared shape behind `Ordering` and `ConvErr`;
/// generic value enums (`Option`/`Result`) don't fit (tuple payloads) and stay bespoke.
/// Returns the new enum id. Injection into `mods.enum_ids` is left to the caller.
fn appendNativeUnitEnum(
    gpa: std.mem.Allocator,
    enums: *std.ArrayList(EnumSym),
    name: []const u8,
    variant_names: []const []const u8,
) !u32 {
    const vars = try gpa.alloc(VariantSym, variant_names.len);
    for (variant_names, 0..) |vn, i| vars[i] = .{ .name = vn, .form = .unit };
    const id: u32 = @intCast(enums.items.len);
    try enums.append(gpa, .{
        .decl_node = Ast.none,
        .name = name,
        .mod = 0,
        .pub_export = true,
        .variants = vars,
        .size = 8,
        .state = .done,
    });
    return id;
}

/// Native `enum Ordering { lt, eq, gt }`: AST-less and hand-laid-out so Phase-0b
/// `layoutEnum` early-returns on `state == .done`. Its tag is the variant DECL INDEX
/// (lt=0/eq=1/gt=2), which `get_tag` and the comparison desugar depend on. Universally
/// nameable with no import (injected if-absent so a user `enum Ordering` shadow wins).
fn registerOrdering(gpa: std.mem.Allocator, enums: *std.ArrayList(EnumSym), mods: []ModuleCtx) !u32 {
    const ordering_id = try appendNativeUnitEnum(gpa, enums, "Ordering", &.{ "lt", "eq", "gt" });
    for (mods) |*m| {
        if (m.enum_ids.get("Ordering") == null) try m.enum_ids.put(gpa, "Ordering", ordering_id);
    }
    return ordering_id;
}

/// Native `struct char(uint32)`: the compiler-provided tuple/newtype struct a `char`
/// literal types to. Hand-laid-out (`state == .done`) so Phase-0b `layoutStruct`
/// early-returns rather than dereferencing the absent decl — its single `uint32` field
/// occupies a full 8-byte scalar slot (like every int width), so `size == 8`. Appended
/// AFTER every user struct so its id is a pure function of source, and injected into each
/// module's `struct_ids` if-absent so a user `struct char` shadow wins (the `Ordering`
/// enum + `byte` alias precedent). Ord/Eq/Hash derive over the inner `uint32` through the
/// ordinary structural path — no bespoke conformance rows. Every owned slice is
/// `gpa`-allocated so the checker's struct teardown frees prelude and user structs
/// uniformly (the borrowed `"char"` name + `"0"` field name are not freed there).
fn registerChar(gpa: std.mem.Allocator, structs: *std.ArrayList(StructSym), mods: []ModuleCtx) !u32 {
    const field_names = try gpa.alloc([]const u8, 1);
    field_names[0] = LayoutEngine.tuple_field_names[0]; // "0"
    const field_types = try gpa.alloc(Type, 1);
    field_types[0] = Type.uint32;
    const offsets = try gpa.alloc(u32, 1);
    offsets[0] = 0;
    const id: u32 = @intCast(structs.items.len);
    try structs.append(gpa, .{
        .decl_node = Ast.none,
        .name = "char",
        .mod = 0,
        .pub_export = true,
        .field_names = field_names,
        .field_types = field_types,
        .offsets = offsets,
        .size = 8,
        .@"align" = 8,
        .state = .done,
        .is_tuple = true,
    });
    for (mods) |*m| {
        if (m.struct_ids.get("char") == null) try m.struct_ids.put(gpa, "char", id);
    }
    return id;
}

/// Prelude generic value enums: `enum Option[T] { some(T), none }` and
/// `enum Result[T,E] { ok(T), err(E) }`, hand-built as generic TEMPLATES (the shape
/// `registerEnums` + `decodeTemplateVariants` produce, but AST-less). Appended AFTER every
/// user enum + `Ordering` so their ids stay a pure function of source. Variant order is
/// FIXED (some=0/none=1, ok=0/err=1) — the `?` operator resolves by variant ordinal.
fn registerOptionResult(
    gpa: std.mem.Allocator,
    enums: *std.ArrayList(EnumSym),
    mods: []ModuleCtx,
    prelude: *Prelude,
) !void {
    const option_gparams = try gpa.alloc([]const u8, 1);
    option_gparams[0] = "T";
    const option_vars = try gpa.alloc(VariantSym, 2);
    const option_some_ft = try gpa.alloc(Type, 1);
    option_some_ft[0] = Type.typeVar(0);
    option_vars[0] = .{ .name = "some", .form = .tuple, .field_types = option_some_ft };
    option_vars[1] = .{ .name = "none", .form = .unit };
    const option_id: u32 = @intCast(enums.items.len);
    try enums.append(gpa, .{
        .decl_node = Ast.none,
        .name = "Option",
        .mod = 0,
        .pub_export = true,
        .variants = option_vars,
        .size = 0,
        .@"align" = 8,
        .state = .done,
        .is_generic = true,
        .generic_params = option_gparams,
    });

    const result_gparams = try gpa.alloc([]const u8, 2);
    result_gparams[0] = "T";
    result_gparams[1] = "E";
    const result_vars = try gpa.alloc(VariantSym, 2);
    const result_ok_ft = try gpa.alloc(Type, 1);
    result_ok_ft[0] = Type.typeVar(0);
    const result_err_ft = try gpa.alloc(Type, 1);
    result_err_ft[0] = Type.typeVar(1);
    result_vars[0] = .{ .name = "ok", .form = .tuple, .field_types = result_ok_ft };
    result_vars[1] = .{ .name = "err", .form = .tuple, .field_types = result_err_ft };
    const result_id: u32 = @intCast(enums.items.len);
    try enums.append(gpa, .{
        .decl_node = Ast.none,
        .name = "Result",
        .mod = 0,
        .pub_export = true,
        .variants = result_vars,
        .size = 0,
        .@"align" = 8,
        .state = .done,
        .is_generic = true,
        .generic_params = result_gparams,
    });

    prelude.option_enum = option_id;
    prelude.result_enum = result_id;

    for (mods) |*m| {
        if (m.enum_ids.get("Option") == null) try m.enum_ids.put(gpa, "Option", option_id);
    }
    for (mods) |*m| {
        if (m.enum_ids.get("Result") == null) try m.enum_ids.put(gpa, "Result", result_id);
    }
}

const testing = std.testing;

test "M3: Into=9/TryInto=10 after From; ConvErr appended after Result" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var protocols: std.ArrayList(ProtocolSym) = .empty;
    var conformances: std.ArrayList(Conformance) = .empty;
    var enums: std.ArrayList(EnumSym) = .empty;
    var structs: std.ArrayList(StructSym) = .empty;
    var mods = [_]ModuleCtx{};

    const prelude = try register(gpa, &protocols, &conformances, &enums, &structs, &mods);

    // Appended after From=8, preserving every existing id.
    try testing.expectEqual(@as(?u32, 8), prelude.protocols.from);
    try testing.expectEqual(@as(?u32, 9), prelude.protocols.into);
    try testing.expectEqual(@as(?u32, 10), prelude.protocols.try_into);

    // Each carries generic_params={"Dst"} and registers NO builtin conformance rows.
    inline for (.{ prelude.protocols.into.?, prelude.protocols.try_into.? }) |pid| {
        const p = protocols.items[pid];
        try testing.expectEqual(@as(usize, 1), p.generic_params.len);
        try testing.expect(std.mem.eql(u8, p.generic_params[0], "Dst"));
        for (conformances.items) |c| try testing.expect(c.protocol != pid);
    }

    // ConvErr is appended AFTER Result (highest enum id) with one unit variant, so no
    // existing Ordering/Option/Result id shifts.
    try testing.expect(prelude.conv_err_enum != null);
    try testing.expect(prelude.conv_err_enum.? > prelude.result_enum.?);
    try testing.expect(prelude.conv_err_enum.? > prelude.option_enum.?);
    try testing.expectEqual(@as(u32, @intCast(enums.items.len - 1)), prelude.conv_err_enum.?);
    const ce = enums.items[prelude.conv_err_enum.?];
    try testing.expectEqual(@as(usize, 1), ce.variants.len);
    try testing.expect(std.mem.eql(u8, ce.variants[0].name, "out_of_range"));
}

test "M9: char is a hand-laid-out tuple struct(uint32), size 8, done" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    var protocols: std.ArrayList(ProtocolSym) = .empty;
    var conformances: std.ArrayList(Conformance) = .empty;
    var enums: std.ArrayList(EnumSym) = .empty;
    var structs: std.ArrayList(StructSym) = .empty;
    var mods = [_]ModuleCtx{};

    const prelude = try register(gpa, &protocols, &conformances, &enums, &structs, &mods);

    try testing.expect(prelude.char_struct != null);
    const c = structs.items[prelude.char_struct.?];
    try testing.expectEqualStrings("char", c.name);
    try testing.expect(c.is_tuple);
    try testing.expectEqual(LayoutEngine.LayoutState.done, c.state);
    try testing.expectEqual(@as(u32, 8), c.size);
    try testing.expectEqual(@as(usize, 1), c.field_types.len);
    try testing.expect(Type.eql(Type.uint32, c.field_types[0]));
    try testing.expectEqualStrings("0", c.field_names[0]);
    try testing.expectEqual(@as(u32, 0), c.offsets[0]);
}
