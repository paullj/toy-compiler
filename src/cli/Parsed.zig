//! Typed parse result: `Parsed(command)` reified from the comptime schema via `@Struct`, so each option and argument is a real typed field.

const std = @import("std");
const Spec = @import("Spec.zig");

const Attr = std.builtin.Type.StructField.Attributes;

/// One struct field per option and per positional of `cmd`. Subcommands are
/// deliberately excluded — dispatch to a subcommand's own `Parsed` is M6's
/// concern, so this type describes exactly the values `cmd` itself binds.
///
/// The generated struct carries no decls (a reified struct cannot), so every
/// helper below is a free function.
pub fn Parsed(comptime cmd: Spec.Command) type {
    const n = cmd.options.len + cmd.positionals.len;
    return comptime blk: {
        var names: [n][:0]const u8 = undefined;
        var types: [n]type = undefined;
        var attrs: [n]Attr = undefined;

        var i: usize = 0;
        for (cmd.options) |o| {
            const f = optionField(o);
            names[i] = fieldName(o);
            types[i] = f.T;
            attrs[i] = f.attr;
            i += 1;
        }
        for (cmd.positionals) |p| {
            const f = positionalField(p);
            names[i] = p.name;
            types[i] = f.T;
            attrs[i] = f.attr;
            i += 1;
        }

        break :blk @Struct(.auto, null, &names, &types, &attrs);
    };
}

/// A real Zig enum whose tags are the choice strings, so `p.emit == .ir` is
/// type-checked. Single-choice enums get a `u0` tag (via `-|1`), which is
/// valid. The same call must be reused for a `.set` field and an `.append`
/// element so the two are the identical type.
fn ChoiceEnum(comptime choices: []const [:0]const u8) type {
    const IntTag = std.math.IntFittingRange(0, choices.len -| 1);
    return @Enum(IntTag, .exhaustive, choices, &std.simd.iota(IntTag, choices.len));
}

/// The scalar Zig type a value maps to before action/arity overrides.
fn baseType(comptime v: Spec.ValueType) type {
    return switch (v) {
        .boolean => bool,
        .int => i64,
        .float => f64,
        .string => []const u8,
        .@"enum" => |choices| ChoiceEnum(choices),
    };
}

const Field = struct { T: type, attr: Attr };

/// Field type + default for an option, folding in action then required-ness:
///   set_true/boolean -> bool = false
///   append           -> []const Base = &.{}
///   count            -> u32 = 0
///   required scalar  -> T, NO default (the parser must fill it)
///   optional scalar  -> ?T = null (distinguishes absent from a supplied value)
fn optionField(comptime o: Spec.Option) Field {
    switch (o.action) {
        .set_true => return .{ .T = bool, .attr = defaultAttr(bool, false) },
        .count => return .{ .T = u32, .attr = defaultAttr(u32, 0) },
        .append => {
            const Base = baseType(o.value);
            const T = []const Base;
            return .{ .T = T, .attr = defaultAttr(T, &.{}) };
        },
        .set => {
            const Base = baseType(o.value);
            // A boolean-valued .set option is still a flag: bool default false.
            if (o.value == .boolean) return .{ .T = bool, .attr = defaultAttr(bool, false) };
            if (o.required) return .{ .T = Base, .attr = .{} };
            const T = ?Base;
            return .{ .T = T, .attr = defaultAttr(T, null) };
        },
    }
}

/// Field type + default for a positional, from arity:
///   one      -> T, NO default (always present when the command matches)
///   optional -> ?T = null
///   variadic -> []const T = &.{}
fn positionalField(comptime p: Spec.Positional) Field {
    const Base = baseType(p.value);
    return switch (p.arity) {
        .one => .{ .T = Base, .attr = .{} },
        .optional => .{ .T = ?Base, .attr = defaultAttr(?Base, null) },
        .variadic => .{ .T = []const Base, .attr = defaultAttr([]const Base, &.{}) },
    };
}

/// Long name with '-' -> '_' (so --dry-run becomes dry_run); if there is no
/// long, the single short byte is the field name. `Spec.checkCommand` already
/// guarantees at least one of long/short and uniqueness, so no collision or
/// empty-name handling is needed here. Public so `Parser` reuses the exact same
/// derivation instead of keeping a copy that could drift.
pub fn fieldName(comptime o: Spec.Option) [:0]const u8 {
    if (o.long) |l| {
        var buf: [l.len:0]u8 = undefined;
        for (l, 0..) |c, i| buf[i] = if (c == '-') '_' else c;
        buf[l.len] = 0;
        const out = buf;
        return &out;
    }
    return &[_:0]u8{o.short.?};
}

/// A field attribute carrying `v` as the default. The value is held by a
/// generic struct's `const value` decl, whose address is a stable comptime
/// pointer that survives being returned from this function and reused inside a
/// loop — a pointer to a plain function-local const does not resolve as a
/// comptime struct-field attribute. The const's type is exactly the field type,
/// so the `@ptrCast` reads the right bytes on load.
fn defaultAttr(comptime T: type, comptime v: T) Attr {
    return .{ .default_value_ptr = @ptrCast(&Default(T, v).value) };
}

fn Default(comptime T: type, comptime v: T) type {
    return struct {
        const value: T = v;
    };
}

test "smoke: trivial @Struct reify + @typeInfo/@hasField/@FieldType" {
    const T = comptime blk: {
        const names = [_][:0]const u8{ "all", "jobs" };
        const types = [_]type{ bool, i64 };
        const d_all: bool = false;
        const d_jobs: i64 = 0;
        const attrs = [_]Attr{
            .{ .default_value_ptr = @ptrCast(&d_all) },
            .{ .default_value_ptr = @ptrCast(&d_jobs) },
        };
        break :blk @Struct(.auto, null, &names, &types, &attrs);
    };
    try std.testing.expectEqual(2, @typeInfo(T).@"struct".fields.len);
    try std.testing.expect(@hasField(T, "all"));
    try std.testing.expect(!@hasField(T, "nope"));
    try std.testing.expect(bool == @FieldType(T, "all"));
    try std.testing.expect(i64 == @FieldType(T, "jobs"));
}

test "smoke: empty struct (0 fields) reifies" {
    const names = [_][:0]const u8{};
    const types = [_]type{};
    const attrs = [_]Attr{};
    const T = @Struct(.auto, null, &names, &types, &attrs);
    try std.testing.expectEqual(0, @typeInfo(T).@"struct".fields.len);
}

test "smoke: @Enum reify (multi + single-choice u0 tag)" {
    const E = ChoiceEnum(&.{ "ir", "asm", "obj" });
    try std.testing.expectEqual(3, @typeInfo(E).@"enum".fields.len);
    const x: E = .ir;
    try std.testing.expect(x == .ir);
    try std.testing.expect(x != .@"asm");

    const Single = ChoiceEnum(&.{"only"});
    try std.testing.expectEqual(u0, @typeInfo(Single).@"enum".tag_type);
    const s: Single = .only;
    try std.testing.expect(s == .only);
}

test "smoke: decl-backed defaults materialize in a loop" {
    // Guards the defaultAttr mechanism directly: slice / optional / scalar
    // defaults must all survive as comptime pointers when built via a loop.
    const T = comptime blk: {
        const names = [_][:0]const u8{ "items", "file", "n" };
        const types = [_]type{ []const i64, ?[]const u8, u32 };
        var attrs: [3]Attr = undefined;
        attrs[0] = defaultAttr([]const i64, &.{});
        attrs[1] = defaultAttr(?[]const u8, null);
        attrs[2] = defaultAttr(u32, 0);
        break :blk @Struct(.auto, null, &names, &types, &attrs);
    };
    const v = T{};
    try std.testing.expectEqual(0, v.items.len);
    try std.testing.expect(v.file == null);
    try std.testing.expectEqual(@as(u32, 0), v.n);
    const w = T{ .items = &.{ 1, 2 }, .file = "x.zig", .n = 3 };
    try std.testing.expectEqual(2, w.items.len);
    try std.testing.expectEqualStrings("x.zig", w.file.?);
    try std.testing.expectEqual(@as(u32, 3), w.n);
}

test "boolean option is bool defaulting false" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{.{ .long = "all", .short = 'a', .value = .boolean }},
    };
    const P = Parsed(cmd);
    try std.testing.expect(bool == @FieldType(P, "all"));
    const p = P{};
    try std.testing.expectEqual(false, p.all);
}

test "int option is i64, string is []const u8, float is f64 (all optional -> ?T)" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "jobs", .value = .{ .int = null } },
            .{ .long = "path", .value = .string },
            .{ .long = "ratio", .value = .{ .float = null } },
        },
    };
    const P = Parsed(cmd);
    try std.testing.expect(?i64 == @FieldType(P, "jobs"));
    try std.testing.expect(?[]const u8 == @FieldType(P, "path"));
    try std.testing.expect(?f64 == @FieldType(P, "ratio"));
    const p = P{};
    try std.testing.expect(p.jobs == null);
    try std.testing.expect(p.path == null);
    try std.testing.expect(p.ratio == null);
}

test "required scalar option is plain T with no default" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{.{ .long = "out", .value = .string, .required = true }},
    };
    const P = Parsed(cmd);
    try std.testing.expect([]const u8 == @FieldType(P, "out"));
    // No default: T{} would be a compile error, so the parser must fill `out`.
    const p = P{ .out = "a.o" };
    try std.testing.expectEqualStrings("a.o", p.out);
}

test "append option is a slice of the base type, default empty" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "def", .value = .string, .action = .append },
            .{ .long = "num", .value = .{ .int = null }, .action = .append },
        },
    };
    const P = Parsed(cmd);
    try std.testing.expect([]const []const u8 == @FieldType(P, "def"));
    try std.testing.expect([]const i64 == @FieldType(P, "num"));
    const p = P{};
    try std.testing.expectEqual(0, p.def.len);
    try std.testing.expectEqual(0, p.num.len);
}

test "count option is u32 defaulting 0" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{.{ .long = "verbose", .short = 'v', .action = .count }},
    };
    const P = Parsed(cmd);
    try std.testing.expect(u32 == @FieldType(P, "verbose"));
    const p = P{};
    try std.testing.expectEqual(@as(u32, 0), p.verbose);
}

test "enum option reifies to a real enum with the choices as tags" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{.{ .long = "emit", .value = .{ .@"enum" = &.{ "ir", "asm" } } }},
    };
    const P = Parsed(cmd);
    // Non-required scalar -> ?EnumType.
    const F = @FieldType(P, "emit");
    try std.testing.expect(@typeInfo(F) == .optional);
    const E = @typeInfo(F).optional.child;
    try std.testing.expectEqual(2, @typeInfo(E).@"enum".fields.len);
    const p = P{ .emit = .ir };
    try std.testing.expect(p.emit.? == .ir);
}

test "enum append field element type equals the enum set type" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{.{ .long = "emit", .value = .{ .@"enum" = &.{ "ir", "asm" } }, .action = .append }},
    };
    const P = Parsed(cmd);
    const F = @FieldType(P, "emit"); // []const ChoiceEnum(...)
    const Elem = @typeInfo(F).pointer.child;
    try std.testing.expectEqual(2, @typeInfo(Elem).@"enum".fields.len);
    const p = P{ .emit = &.{ .ir, .@"asm" } };
    try std.testing.expectEqual(2, p.emit.len);
    try std.testing.expect(p.emit[0] == .ir);
}

test "required enum is a plain enum (no ?); explicit .set boolean stays bool" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "emit", .value = .{ .@"enum" = &.{ "ir", "asm" } }, .required = true },
            .{ .long = "flag", .value = .boolean, .action = .set },
        },
    };
    const P = Parsed(cmd);
    const E = @FieldType(P, "emit");
    try std.testing.expect(@typeInfo(E) == .@"enum"); // plain enum, not ?E (required)
    try std.testing.expectEqual(2, @typeInfo(E).@"enum".fields.len);
    try std.testing.expect(bool == @FieldType(P, "flag")); // .set + .boolean is still a flag
    const p = P{ .emit = .ir }; // required emit has no default; flag defaults false
    try std.testing.expect(p.emit == .ir);
    try std.testing.expectEqual(false, p.flag);
}

test "positional arity: one -> T, optional -> ?T, variadic -> slice" {
    const cmd = Spec.Command{
        .name = "c",
        .positionals = &.{
            .{ .name = "input", .value = .string, .arity = .one },
            .{ .name = "out", .value = .string, .arity = .optional },
            .{ .name = "extra", .value = .string, .arity = .variadic },
        },
    };
    const P = Parsed(cmd);
    try std.testing.expect([]const u8 == @FieldType(P, "input"));
    try std.testing.expect(?[]const u8 == @FieldType(P, "out"));
    try std.testing.expect([]const []const u8 == @FieldType(P, "extra"));
    const p = P{ .input = "main.zig" };
    try std.testing.expectEqualStrings("main.zig", p.input);
    try std.testing.expect(p.out == null);
    try std.testing.expectEqual(0, p.extra.len);
}

test "field name derivation: --dry-run -> dry_run, short-only -> char" {
    const cmd = Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "dry-run", .value = .boolean },
            .{ .short = 'x', .value = .boolean },
        },
    };
    const P = Parsed(cmd);
    try std.testing.expect(@hasField(P, "dry_run"));
    try std.testing.expect(!@hasField(P, "dry-run"));
    try std.testing.expect(@hasField(P, "x"));
}

test "representative command binds options and positionals together" {
    const cmd = Spec.Command{
        .name = "toy",
        .options = &.{
            .{ .long = "all", .short = 'a', .value = .boolean },
            .{ .long = "jobs", .short = 'j', .value = .{ .int = .{ .min = 1, .max = 8 } } },
            .{ .long = "verbose", .short = 'v', .action = .count },
            .{ .long = "define", .short = 'D', .value = .string, .action = .append },
            .{ .long = "emit", .value = .{ .@"enum" = &.{ "ir", "asm", "obj" } } },
        },
        .positionals = &.{
            .{ .name = "input", .value = .string },
            .{ .name = "extra", .value = .string, .arity = .variadic },
        },
        // A subcommand must NOT contribute a field (M6 handles dispatch).
        .subcommands = &.{.{ .name = "build" }},
    };
    comptime Spec.validate(cmd);
    const P = Parsed(cmd);
    const fields = @typeInfo(P).@"struct".fields;
    try std.testing.expectEqual(7, fields.len); // 5 options + 2 positionals, 0 subcommands

    const p = P{ .input = "m.zig" };
    try std.testing.expectEqual(false, p.all);
    try std.testing.expect(p.jobs == null);
    try std.testing.expectEqual(@as(u32, 0), p.verbose);
    try std.testing.expectEqual(0, p.define.len);
    try std.testing.expect(p.emit == null);
    try std.testing.expectEqualStrings("m.zig", p.input);
    try std.testing.expectEqual(0, p.extra.len);

    try std.testing.expect(!@hasField(P, "build"));

    // COMPILE-ERROR DEMOS (uncomment any line to see the failure):
    //   _ = p.nope;          // error: no field named 'nope' in struct
    //   _ = p.jobs + true;   // error: incompatible types: '?i64' and 'bool'
    //   const q = P{};       // error: missing struct field: input (required positional)
    //   p.emit = "ir";       // error: expected optional enum, found []const u8
}
