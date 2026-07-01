//! Exit-free argument parser: matches short/long flags, coerces and validates values, dispatches subcommands, and collects errors into a Sink.
//!
//! The parser never prints and never exits. Every parse-semantic failure is
//! pushed to the caller's Sink and the walk CONTINUES (collect-all), so one call
//! surfaces every problem at once. Only OutOfMemory escapes as an error union;
//! all other outcomes travel in the returned Result.
//!
//! Ownership: `.append`/`.variadic` fields are the only heap the parser produces.
//! They live in a single `std.heap.ArenaAllocator` carried inside `Result`, so
//! `result.deinit()` frees every such slice in one shot regardless of which
//! variant was returned — deinit is uniform whether the arena is empty
//! (help/version/subcommand) or holds slices (ok, or errors after a partial
//! bind). The Sink's own list is grown with the caller's `gpa`
//! and owned/freed by the caller; Sink error strings are BORROWED from argv, so
//! the parser allocates none of them.

const std = @import("std");
const Spec = @import("Spec.zig");
const Parsed = @import("Parsed.zig");
const Sink = @import("Sink.zig");

/// Result of one `parse` call for `cmd`. A struct wrapper (not the bare union)
/// so the owning arena travels with the value; read `result.value` for the
/// contract union. `.subcommand` carries the comptime-matched index so the
/// caller can re-enter `parse` on `cmd.subcommands[index]` with `rest`.
///
/// Held by value once returned: the arena has internal pointers, so move it,
/// never copy it after allocation.
pub fn Result(comptime cmd: Spec.Command) type {
    return struct {
        arena: std.heap.ArenaAllocator,
        value: Value,

        pub const Value = union(enum) {
            ok: Parsed.Parsed(cmd),
            help: struct { path: []const u8, mode: enum { short, long } },
            version,
            subcommand: struct { index: usize, name: []const u8, rest: []const []const u8 },
            errors,
        };

        pub fn deinit(self: *@This()) void {
            self.arena.deinit();
        }

        pub fn unwrap(self: @This()) Value {
            return self.value;
        }
    };
}

/// Parse `argv` against `cmd`, pushing every failure to `sink` and continuing.
/// Returns a `Result(cmd)` the caller must `deinit`. Only OutOfMemory escapes.
pub fn parse(
    gpa: std.mem.Allocator,
    comptime cmd: Spec.Command,
    argv: []const []const u8,
    sink: *Sink.Sink,
) std.mem.Allocator.Error!Result(cmd) {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var p: Parsed.Parsed(cmd) = undefined;
    initDefaults(cmd, &p);

    var appends = AppendLists(cmd){};

    // seen[i] for options; pos_filled counts positionals bound so far. Both feed
    // the post-walk missing_required / conflict / requires sweep.
    var seen = [_]bool{false} ** cmd.options.len;
    var pos_filled: usize = 0;

    var positional_only = false;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];

        if (!positional_only) {
            if (std.mem.eql(u8, arg, "--")) {
                positional_only = true;
                continue;
            }
            switch (classify(arg)) {
                .long => |lv| {
                    if (std.mem.eql(u8, lv.name, "help"))
                        return .{ .arena = arena, .value = .{ .help = .{ .path = cmd.name, .mode = .long } } };
                    if (std.mem.eql(u8, lv.name, "version"))
                        return .{ .arena = arena, .value = .version };
                    try handleLong(cmd, &p, gpa, a, &appends, &seen, sink, argv, &i, lv);
                    continue;
                },
                .short_cluster => |bytes| {
                    switch (try handleShortCluster(cmd, &p, gpa, a, &appends, &seen, sink, argv, &i, bytes)) {
                        .none => {},
                        .help => return .{ .arena = arena, .value = .{ .help = .{ .path = cmd.name, .mode = .short } } },
                        .version => return .{ .arena = arena, .value = .version },
                    }
                    continue;
                },
                .positional, .dash_positional => {},
                .dashdash => unreachable, // handled above
            }
        }

        // Positional token (or anything after `--`). First bare positional that
        // matches a subcommand name stops the parent and hands back the tail.
        if (cmd.subcommands.len > 0 and pos_filled == 0 and !positional_only) {
            var dispatched = false;
            inline for (cmd.subcommands, 0..) |sub, j| {
                if (!dispatched and std.mem.eql(u8, arg, sub.name)) {
                    dispatched = true;
                    return .{ .arena = arena, .value = .{ .subcommand = .{
                        .index = j,
                        .name = sub.name,
                        .rest = argv[i + 1 ..],
                    } } };
                }
            }
        }

        try bindPositional(cmd, &p, gpa, a, &appends, sink, arg, &pos_filled);
    }

    try finishAppends(cmd, &p, a, &appends);
    try sweepRequiredAndRelations(cmd, gpa, &seen, pos_filled, sink);

    if (!sink.isEmpty()) return .{ .arena = arena, .value = .errors };
    return .{ .arena = arena, .value = .{ .ok = p } };
}

// ---- token classification -------------------------------------------------

const Long = struct { name: []const u8, inline_val: ?[]const u8 };

const Token = union(enum) {
    long: Long,
    short_cluster: []const u8,
    dashdash,
    dash_positional: []const u8,
    positional: []const u8,
};

/// Pure, allocation-free token step. A bare `-` is a positional (semantics 4);
/// `--foo`/`--foo=v` is a long (semantics 1); `-abc` is a short cluster
/// (semantics 2); `--` is a terminator. Value tokens are never classified — the
/// main loop grabs the value raw right after the flag that demands it, so a
/// negative number like `-5` reaches here only as a standalone token (then a
/// short cluster), never as an expected value (semantics 5 lives in that raw
/// grab, see the sem5 test).
fn classify(arg: []const u8) Token {
    if (arg.len == 0) return .{ .positional = arg };
    if (arg[0] != '-') return .{ .positional = arg };
    if (arg.len == 1) return .{ .dash_positional = arg }; // bare "-"
    if (std.mem.eql(u8, arg, "--")) return .dashdash;
    if (arg[1] == '-') {
        const body = arg[2..];
        if (std.mem.indexOfScalar(u8, body, '=')) |eq|
            return .{ .long = .{ .name = body[0..eq], .inline_val = body[eq + 1 ..] } };
        return .{ .long = .{ .name = body, .inline_val = null } };
    }
    return .{ .short_cluster = arg[1..] };
}

// ---- long options ----------------------------------------------------------

fn handleLong(
    comptime cmd: Spec.Command,
    p: *Parsed.Parsed(cmd),
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    appends: *AppendLists(cmd),
    seen: []bool,
    sink: *Sink.Sink,
    argv: []const []const u8,
    i: *usize,
    lv: Long,
) !void {
    inline for (cmd.options, 0..) |o, oi| {
        if (o.long) |l| {
            if (std.mem.eql(u8, l, lv.name)) {
                seen[oi] = true;
                if (comptime isFlag(o)) {
                    if (lv.inline_val != null) {
                        try sink.add(gpa, .{ .kind = .bad_value, .arg = argv[i.*], .expected = "no value (flag)" });
                        return;
                    }
                    applyFlag(cmd, o, p);
                    return;
                }
                var raw: []const u8 = undefined;
                if (lv.inline_val) |v| {
                    raw = v;
                } else if (i.* + 1 < argv.len) {
                    i.* += 1;
                    raw = argv[i.*];
                } else {
                    try sink.add(gpa, .{ .kind = .missing_value, .arg = argv[i.*], .expected = valueKindName(o.value) });
                    return;
                }
                try applyValue(cmd, o, oi, p, gpa, a, appends, sink, argv[i.*], raw);
                return;
            }
        }
    }
    try sink.add(gpa, .{ .kind = .unknown_flag, .arg = argv[i.*] });
}

// ---- short clusters --------------------------------------------------------

const ShortOutcome = enum { none, help, version };

fn handleShortCluster(
    comptime cmd: Spec.Command,
    p: *Parsed.Parsed(cmd),
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    appends: *AppendLists(cmd),
    seen: []bool,
    sink: *Sink.Sink,
    argv: []const []const u8,
    i: *usize,
    bytes: []const u8,
) !ShortOutcome {
    var k: usize = 0;
    while (k < bytes.len) : (k += 1) {
        const c = bytes[k];
        if (c == 'h') return .help;
        if (c == 'V') return .version;

        var matched = false;
        inline for (cmd.options, 0..) |o, oi| {
            if (o.short) |s| {
                if (!matched and s == c) {
                    matched = true;
                    seen[oi] = true;
                    if (comptime isFlag(o)) {
                        applyFlag(cmd, o, p);
                    } else {
                        // First value-taking short consumes the rest of the
                        // cluster ('-j4','-Ipath'); if it is the last byte, the
                        // value is the next argv token ('-j 4').
                        var raw: []const u8 = undefined;
                        if (k + 1 < bytes.len) {
                            raw = bytes[k + 1 ..];
                        } else if (i.* + 1 < argv.len) {
                            i.* += 1;
                            raw = argv[i.*];
                        } else {
                            try sink.add(gpa, .{ .kind = .missing_value, .arg = argv[i.*], .expected = valueKindName(o.value) });
                            return .none;
                        }
                        try applyValue(cmd, o, oi, p, gpa, a, appends, sink, argv[i.*], raw);
                        return .none; // value-taking short ends the cluster
                    }
                }
            }
        }
        if (!matched)
            try sink.add(gpa, .{ .kind = .unknown_flag, .arg = argv[i.*] });
    }
    return .none;
}

// ---- field binding ---------------------------------------------------------

fn isFlag(comptime o: Spec.Option) bool {
    return o.action == .set_true or o.action == .count or (o.action == .set and o.value == .boolean);
}

fn applyFlag(comptime cmd: Spec.Command, comptime o: Spec.Option, p: *Parsed.Parsed(cmd)) void {
    const name = comptime fieldName(o);
    switch (o.action) {
        .count => @field(p, name) +%= 1,
        else => @field(p, name) = true, // set_true or .set boolean
    }
}

fn applyValue(
    comptime cmd: Spec.Command,
    comptime o: Spec.Option,
    comptime oi: usize,
    p: *Parsed.Parsed(cmd),
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    appends: *AppendLists(cmd),
    sink: *Sink.Sink,
    arg: []const u8,
    raw: []const u8,
) !void {
    const name = comptime fieldName(o);
    if (comptime o.action == .append) {
        const Elem = ElemOf(cmd, name);
        const coerced = coerceInto(Elem, o.value, raw) orelse {
            try sink.add(gpa, .{ .kind = .bad_value, .arg = arg, .got = raw, .expected = valueKindName(o.value) });
            return;
        };
        try appends.appendOpt(oi, a, coerced);
    } else {
        const Scalar = ScalarOf(cmd, name);
        const coerced = coerceInto(Scalar, o.value, raw) orelse {
            try sink.add(gpa, .{ .kind = .bad_value, .arg = arg, .got = raw, .expected = valueKindName(o.value) });
            return;
        };
        writeScalar(cmd, name, Scalar, p, coerced);
    }
}

fn bindPositional(
    comptime cmd: Spec.Command,
    p: *Parsed.Parsed(cmd),
    gpa: std.mem.Allocator,
    a: std.mem.Allocator,
    appends: *AppendLists(cmd),
    sink: *Sink.Sink,
    arg: []const u8,
    pos_filled: *usize,
) !void {
    // Fill positionals left to right; a variadic (always last) soaks up the rest.
    inline for (cmd.positionals, 0..) |pp, pi| {
        if (pp.arity == .variadic) {
            if (pos_filled.* >= pi) {
                const Elem = ElemOf(cmd, pp.name);
                const coerced = coerceInto(Elem, pp.value, arg) orelse {
                    try sink.add(gpa, .{ .kind = .bad_value, .arg = arg, .got = arg, .expected = valueKindName(pp.value) });
                    return;
                };
                try appends.appendPos(pi, a, coerced);
                pos_filled.* += 1;
                return;
            }
        } else if (pos_filled.* == pi) {
            const Scalar = ScalarOf(cmd, pp.name);
            const coerced = coerceInto(Scalar, pp.value, arg) orelse {
                try sink.add(gpa, .{ .kind = .bad_value, .arg = arg, .got = arg, .expected = valueKindName(pp.value) });
                // Advance past the rejected slot so the next token binds to the
                // following positional, not back onto this failed one.
                pos_filled.* += 1;
                return;
            };
            writeScalar(cmd, pp.name, Scalar, p, coerced);
            pos_filled.* += 1;
            return;
        }
    }
    try sink.add(gpa, .{ .kind = .unexpected_arg, .arg = arg });
}

/// Write a coerced scalar into field `name`, wrapping into the optional if the
/// reified field is `?Base` (non-required scalar / optional positional).
fn writeScalar(comptime cmd: Spec.Command, comptime name: [:0]const u8, comptime Base: type, p: *Parsed.Parsed(cmd), coerced: Base) void {
    const F = @FieldType(Parsed.Parsed(cmd), name);
    if (@typeInfo(F) == .optional) {
        @field(p, name) = @as(Base, coerced);
    } else {
        @field(p, name) = coerced;
    }
}

/// Flush accumulated append/variadic lists into their slice fields.
fn finishAppends(comptime cmd: Spec.Command, p: *Parsed.Parsed(cmd), a: std.mem.Allocator, appends: *AppendLists(cmd)) !void {
    inline for (cmd.options, 0..) |o, oi| {
        if (o.action == .append)
            @field(p, fieldName(o)) = try appends.sliceOpt(oi, a);
    }
    inline for (cmd.positionals, 0..) |pp, pi| {
        if (pp.arity == .variadic)
            @field(p, pp.name) = try appends.slicePos(pi, a);
    }
}

// ---- required / conflicts / requires sweep --------------------------------

fn sweepRequiredAndRelations(
    comptime cmd: Spec.Command,
    gpa: std.mem.Allocator,
    seen: []const bool,
    pos_filled: usize,
    sink: *Sink.Sink,
) !void {
    inline for (cmd.options, 0..) |o, oi| {
        if (o.required and !seen[oi])
            try sink.add(gpa, .{ .kind = .missing_required, .arg = optName(o) });
    }
    inline for (cmd.positionals, 0..) |pp, pi| {
        if (pp.arity == .one and pos_filled <= pi)
            try sink.add(gpa, .{ .kind = .missing_required, .arg = pp.name });
    }
    inline for (cmd.options, 0..) |o, oi| {
        if (seen[oi]) {
            inline for (o.conflicts) |ref| {
                inline for (cmd.options, 0..) |other, oj| {
                    if (comptime optRefMatches(other, ref)) {
                        if (seen[oj])
                            try sink.add(gpa, .{ .kind = .conflict, .arg = optName(o), .where = ref });
                    }
                }
            }
            inline for (o.requires) |ref| {
                inline for (cmd.options, 0..) |other, oj| {
                    if (comptime optRefMatches(other, ref)) {
                        if (!seen[oj])
                            try sink.add(gpa, .{ .kind = .unmet_requirement, .arg = optName(o), .where = ref });
                    }
                }
            }
        }
    }
}

fn optRefMatches(comptime o: Spec.Option, comptime ref: []const u8) bool {
    if (o.long) |l| {
        if (std.mem.eql(u8, l, ref)) return true;
    }
    if (o.short) |s| {
        if (ref.len == 1 and ref[0] == s) return true;
    }
    return false;
}

fn optName(comptime o: Spec.Option) []const u8 {
    if (o.long) |l| return l;
    return &[_]u8{o.short.?};
}

// ---- defaults --------------------------------------------------------------

/// A reified Parsed has required fields with NO default, so `Parsed{}` won't
/// compile. Fill every field with its documented default first; the walk then
/// overwrites what argv provides. Required scalars/`.one` positionals are left
/// undefined and only read on the `.ok` path, which is gated by a filled/seen
/// check that lands them in `.errors` when missing.
fn initDefaults(comptime cmd: Spec.Command, p: *Parsed.Parsed(cmd)) void {
    inline for (cmd.options) |o| {
        const name = comptime fieldName(o);
        switch (o.action) {
            .set_true => @field(p, name) = false,
            .count => @field(p, name) = 0,
            .append => @field(p, name) = &.{},
            .set => {
                if (o.value == .boolean) {
                    @field(p, name) = false;
                } else if (!o.required) {
                    @field(p, name) = null;
                }
            },
        }
    }
    inline for (cmd.positionals) |pp| {
        switch (pp.arity) {
            .one => {},
            .optional => @field(p, pp.name) = null,
            .variadic => @field(p, pp.name) = &.{},
        }
    }
}

// ---- coercion --------------------------------------------------------------

/// Coerce `raw` into `Target` — the EXACT scalar type read off the Parsed field
/// (via `@FieldType`), applying Range / one-of validation. Returns null on any
/// failure (caller emits `.bad_value`). Taking `Target` from Parsed is what
/// makes enum coercion type-correct: a reified `@Enum` is a fresh type per call
/// site, so re-deriving one here would NOT equal the field's enum; instead we
/// `@enumFromInt` into the field's own enum. `std.meta.stringToEnum` can't be
/// used either — choice names need not be valid identifiers.
fn coerceInto(comptime Target: type, comptime v: Spec.ValueType, raw: []const u8) ?Target {
    switch (v) {
        .boolean => return true,
        .string => return raw,
        .int => |range| {
            const n = std.fmt.parseInt(i64, raw, 10) catch return null;
            if (range) |r| {
                if (r.min) |mn| if (n < mn) return null;
                if (r.max) |mx| if (n > mx) return null;
            }
            return n;
        },
        .float => |range| {
            const f = std.fmt.parseFloat(f64, raw) catch return null;
            if (range) |r| {
                if (r.min) |mn| if (f < mn) return null;
                if (r.max) |mx| if (f > mx) return null;
            }
            return f;
        },
        .@"enum" => |choices| {
            inline for (choices, 0..) |choice, idx| {
                if (std.mem.eql(u8, raw, choice)) return @enumFromInt(idx);
            }
            return null;
        },
    }
}

/// The scalar type stored directly in a non-list Parsed field `name`: `?T`
/// unwraps to `T`, everything else (incl. a `[]const u8` string) is itself.
/// Used for `.set` options and `.one`/`.optional` positionals.
fn ScalarOf(comptime cmd: Spec.Command, comptime name: [:0]const u8) type {
    const F = @FieldType(Parsed.Parsed(cmd), name);
    return switch (@typeInfo(F)) {
        .optional => |o| o.child,
        else => F,
    };
}

/// The element type of a list Parsed field `name` (`[]const Elem`): the slice
/// child. For a string append that child is itself `[]const u8`. Used for
/// `.append` options and `.variadic` positionals.
fn ElemOf(comptime cmd: Spec.Command, comptime name: [:0]const u8) type {
    const F = @FieldType(Parsed.Parsed(cmd), name);
    return @typeInfo(F).pointer.child;
}

fn valueKindName(comptime v: Spec.ValueType) []const u8 {
    return switch (v) {
        .boolean => "boolean",
        .int => "int",
        .float => "float",
        .string => "string",
        .@"enum" => "one of the choices",
    };
}

// ---- append accumulators ---------------------------------------------------

/// A comptime-shaped struct of typed ArrayLists — one field `o{oi}` per append
/// option, one `p{pi}` per variadic positional — keyed by original index so the
/// bind sites can `@field` by name. Non-append/non-variadic slots are absent.
/// All lists allocate into the parse arena; `Result.deinit` frees them at once.
fn AppendLists(comptime cmd: Spec.Command) type {
    return struct {
        opt_lists: OptLists = .{},
        pos_lists: PosLists = .{},

        const OptLists = optListStruct(cmd);
        const PosLists = posListStruct(cmd);

        fn appendOpt(self: *@This(), comptime oi: usize, a: std.mem.Allocator, v: anytype) !void {
            try @field(self.opt_lists, std.fmt.comptimePrint("o{d}", .{oi})).append(a, v);
        }
        fn appendPos(self: *@This(), comptime pi: usize, a: std.mem.Allocator, v: anytype) !void {
            try @field(self.pos_lists, std.fmt.comptimePrint("p{d}", .{pi})).append(a, v);
        }
        fn sliceOpt(self: *@This(), comptime oi: usize, a: std.mem.Allocator) ![]const ElemOf(cmd, fieldName(cmd.options[oi])) {
            return @field(self.opt_lists, std.fmt.comptimePrint("o{d}", .{oi})).toOwnedSlice(a);
        }
        fn slicePos(self: *@This(), comptime pi: usize, a: std.mem.Allocator) ![]const ElemOf(cmd, cmd.positionals[pi].name) {
            return @field(self.pos_lists, std.fmt.comptimePrint("p{d}", .{pi})).toOwnedSlice(a);
        }
    };
}

/// One ArrayList field `o{oi}` per append option, element type taken from the
/// Parsed slice field so it is byte-identical to what the slice will hold.
fn optListStruct(comptime cmd: Spec.Command) type {
    comptime {
        var n: usize = 0;
        for (cmd.options) |o| {
            if (o.action == .append) n += 1;
        }
        var names: [n][:0]const u8 = undefined;
        var types: [n]type = undefined;
        var attrs: [n]std.builtin.Type.StructField.Attributes = undefined;
        var k: usize = 0;
        for (cmd.options, 0..) |o, oi| {
            if (o.action == .append) {
                names[k] = std.fmt.comptimePrint("o{d}", .{oi});
                const L = std.ArrayList(ElemOf(cmd, fieldName(o)));
                types[k] = L;
                attrs[k] = defaultAttr(L, .empty);
                k += 1;
            }
        }
        return @Struct(.auto, null, &names, &types, &attrs);
    }
}

fn posListStruct(comptime cmd: Spec.Command) type {
    comptime {
        var n: usize = 0;
        for (cmd.positionals) |pp| {
            if (pp.arity == .variadic) n += 1;
        }
        var names: [n][:0]const u8 = undefined;
        var types: [n]type = undefined;
        var attrs: [n]std.builtin.Type.StructField.Attributes = undefined;
        var k: usize = 0;
        for (cmd.positionals, 0..) |pp, pi| {
            if (pp.arity == .variadic) {
                names[k] = std.fmt.comptimePrint("p{d}", .{pi});
                const L = std.ArrayList(ElemOf(cmd, pp.name));
                types[k] = L;
                attrs[k] = defaultAttr(L, .empty);
                k += 1;
            }
        }
        return @Struct(.auto, null, &names, &types, &attrs);
    }
}

fn defaultAttr(comptime T: type, comptime v: T) std.builtin.Type.StructField.Attributes {
    return .{ .default_value_ptr = @ptrCast(&Default(T, v).value) };
}

fn Default(comptime T: type, comptime v: T) type {
    return struct {
        const value: T = v;
    };
}

// ---- field name (mirrors Parsed's private derivation) ----------------------

/// Long name with '-' -> '_'; else the single short byte. Duplicates Parsed's
/// private `fieldName` — the two MUST stay byte-identical, guarded by the
/// '--dry-run' @hasField test below.
fn fieldName(comptime o: Spec.Option) [:0]const u8 {
    if (o.long) |l| {
        var buf: [l.len:0]u8 = undefined;
        for (l, 0..) |c, i| buf[i] = if (c == '-') '_' else c;
        buf[l.len] = 0;
        const out = buf;
        return &out;
    }
    return &[_:0]u8{o.short.?};
}

// ============================================================================
// Tests
// ============================================================================

const testing = std.testing;

fn repCmd() Spec.Command {
    return .{
        .name = "toy",
        .options = &.{
            .{ .long = "all", .short = 'a', .value = .boolean },
            .{ .long = "jobs", .short = 'j', .value = .{ .int = .{ .min = 1, .max = 8 } } },
            .{ .long = "verbose", .short = 'v', .action = .count },
            .{ .long = "define", .short = 'D', .value = .string, .action = .append },
            .{ .long = "emit", .value = .{ .@"enum" = &.{ "ir", "asm", "obj" } } },
            .{ .long = "include", .short = 'I', .value = .string },
        },
        .positionals = &.{
            .{ .name = "input", .value = .string },
            .{ .name = "extra", .value = .string, .arity = .variadic },
        },
    };
}

test "classify: long, long=val, short cluster, dash positional, terminator" {
    try testing.expect(classify("--foo").long.inline_val == null);
    try testing.expectEqualStrings("bar", classify("--foo=bar").long.inline_val.?);
    try testing.expectEqualStrings("abc", classify("-abc").short_cluster);
    try testing.expectEqualStrings("-", classify("-").dash_positional);
    try testing.expect(classify("--") == .dashdash);
    try testing.expectEqualStrings("plain", classify("plain").positional);
    // A standalone "-5" is a short cluster; negative-number VALUES are handled by
    // the raw value-grab, not classify (see sem5).
    try testing.expectEqualStrings("5", classify("-5").short_cluster);
}

test "fieldName mirrors Parsed: --dry-run -> dry_run" {
    const cmd = comptime Spec.Command{ .name = "c", .options = &.{.{ .long = "dry-run", .value = .boolean }} };
    const derived = comptime fieldName(cmd.options[0]);
    try testing.expect(@hasField(Parsed.Parsed(cmd), derived));
    try testing.expectEqualStrings("dry_run", derived);
}

test "sem1: long value both --opt=val and --opt val" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r1 = try parse(testing.allocator, cmd, &.{ "--jobs=4", "in.zig" }, &sink);
    defer r1.deinit();
    try testing.expectEqual(@as(i64, 4), r1.value.ok.jobs.?);

    var r2 = try parse(testing.allocator, cmd, &.{ "--jobs", "6", "in.zig" }, &sink);
    defer r2.deinit();
    try testing.expectEqual(@as(i64, 6), r2.value.ok.jobs.?);
}

test "sem2: short bundling trailing flags + value-taking short (-j4, -I path, -j 4)" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    // -av bundles two flags, then -j4 consumes rest of a fresh cluster.
    var r1 = try parse(testing.allocator, cmd, &.{ "-av", "-j4", "in.zig" }, &sink);
    defer r1.deinit();
    try testing.expect(r1.value.ok.all);
    try testing.expectEqual(@as(u32, 1), r1.value.ok.verbose);
    try testing.expectEqual(@as(i64, 4), r1.value.ok.jobs.?);

    // -Ipath: value-taking short takes rest of cluster.
    var r2 = try parse(testing.allocator, cmd, &.{ "-Ipath", "in.zig" }, &sink);
    defer r2.deinit();
    try testing.expectEqualStrings("path", r2.value.ok.include.?);

    // -j 4: value-taking short with empty rest takes next token.
    var r3 = try parse(testing.allocator, cmd, &.{ "-j", "4", "in.zig" }, &sink);
    defer r3.deinit();
    try testing.expectEqual(@as(i64, 4), r3.value.ok.jobs.?);
}

test "sem3: -- terminator sends the rest to positionals" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "in.zig", "--", "--all", "-j" }, &sink);
    defer r.deinit();
    try testing.expectEqualStrings("in.zig", r.value.ok.input);
    try testing.expectEqual(@as(usize, 2), r.value.ok.extra.len);
    try testing.expectEqualStrings("--all", r.value.ok.extra[0]);
    try testing.expectEqualStrings("-j", r.value.ok.extra[1]);
    try testing.expect(!r.value.ok.all);
}

test "sem4: bare - is a positional, not a flag" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{"-"}, &sink);
    defer r.deinit();
    try testing.expectEqualStrings("-", r.value.ok.input);
}

test "sem5: negative-number value accepted when a value is expected" {
    const cmd = comptime Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "off", .value = .{ .int = null } },
            .{ .long = "ratio", .value = .{ .float = null } },
        },
        .positionals = &.{.{ .name = "input", .value = .string }},
    };
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "--off", "-5", "--ratio=-1.5", "in" }, &sink);
    defer r.deinit();
    try testing.expectEqual(@as(i64, -5), r.value.ok.off.?);
    try testing.expectEqual(@as(f64, -1.5), r.value.ok.ratio.?);
}

test "sem6: unknown flag pushes error and continues" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "--nope", "in.zig" }, &sink);
    defer r.deinit();
    try testing.expect(r.value == .errors);
    try testing.expectEqual(@as(usize, 1), sink.count());
    try testing.expectEqual(Sink.Kind.unknown_flag, sink.items()[0].kind);
    try testing.expectEqualStrings("--nope", sink.items()[0].arg);
}

test "sem7: subcommand dispatch returns index + rest; caller re-enters" {
    const cmd = comptime Spec.Command{
        .name = "toy",
        .options = &.{.{ .long = "all", .short = 'a', .value = .boolean }},
        .subcommands = &.{
            .{ .name = "build", .options = &.{.{ .long = "release", .short = 'r', .value = .boolean }}, .positionals = &.{.{ .name = "target" }} },
        },
    };
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "-a", "build", "--release", "x" }, &sink);
    defer r.deinit();
    try testing.expect(r.value == .subcommand);
    try testing.expectEqual(@as(usize, 0), r.value.subcommand.index);
    try testing.expectEqualStrings("build", r.value.subcommand.name);
    try testing.expectEqual(@as(usize, 2), r.value.subcommand.rest.len);

    // Caller re-enters parse on the matched subcommand.
    inline for (cmd.subcommands, 0..) |sub, j| {
        if (j == r.value.subcommand.index) {
            var sub_r = try parse(testing.allocator, sub, r.value.subcommand.rest, &sink);
            defer sub_r.deinit();
            try testing.expect(sub_r.value.ok.release);
            try testing.expectEqualStrings("x", sub_r.value.ok.target);
        }
    }
}

test "collect-all: two independent errors in one parse" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "--nope", "--jobs", "99", "in.zig" }, &sink);
    defer r.deinit();
    try testing.expect(r.value == .errors);
    try testing.expectEqual(@as(usize, 2), sink.count());
    try testing.expectEqual(Sink.Kind.unknown_flag, sink.items()[0].kind); // --nope
    try testing.expectEqual(Sink.Kind.bad_value, sink.items()[1].kind); // 99 > max 8
}

test "coercion: int range, float range, enum one-of failures" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "--jobs", "0", "--emit", "wat", "in.zig" }, &sink);
    defer r.deinit();
    try testing.expect(r.value == .errors);
    try testing.expectEqual(@as(usize, 2), sink.count());
    try testing.expectEqual(Sink.Kind.bad_value, sink.items()[0].kind);
    try testing.expectEqualStrings("0", sink.items()[0].got); // below min 1
    try testing.expectEqualStrings("wat", sink.items()[1].got); // not a choice
}

test "coercion: valid enum binds the exact reified tag" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "--emit", "asm", "in.zig" }, &sink);
    defer r.deinit();
    try testing.expect(r.value.ok.emit.? == .@"asm");
}

test "append accumulates into an owned slice; ownership freed by arena deinit" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "-D", "A=1", "--define", "B=2", "-DC=3", "in.zig" }, &sink);
    defer r.deinit(); // frees the []const []const u8 append slice
    try testing.expectEqual(@as(usize, 3), r.value.ok.define.len);
    try testing.expectEqualStrings("A=1", r.value.ok.define[0]);
    try testing.expectEqualStrings("B=2", r.value.ok.define[1]);
    try testing.expectEqualStrings("C=3", r.value.ok.define[2]);
}

test "variadic positional collects the tail; count option repeats" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "-vv", "a.zig", "b.zig", "c.zig" }, &sink);
    defer r.deinit();
    try testing.expectEqual(@as(u32, 2), r.value.ok.verbose);
    try testing.expectEqualStrings("a.zig", r.value.ok.input);
    try testing.expectEqual(@as(usize, 2), r.value.ok.extra.len);
    try testing.expectEqualStrings("c.zig", r.value.ok.extra[1]);
}

test "optional positional binds when present, stays null when absent" {
    const cmd = comptime Spec.Command{
        .name = "c",
        .positionals = &.{
            .{ .name = "input", .value = .string },
            .{ .name = "out", .value = .string, .arity = .optional },
        },
    };
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r1 = try parse(testing.allocator, cmd, &.{ "a", "b" }, &sink);
    defer r1.deinit();
    try testing.expectEqualStrings("a", r1.value.ok.input);
    try testing.expectEqualStrings("b", r1.value.ok.out.?);

    var r2 = try parse(testing.allocator, cmd, &.{"a"}, &sink);
    defer r2.deinit();
    try testing.expectEqualStrings("a", r2.value.ok.input);
    try testing.expect(r2.value.ok.out == null);
}

test "missing-required option and positional both reported" {
    const cmd = comptime Spec.Command{
        .name = "c",
        .options = &.{.{ .long = "out", .short = 'o', .value = .string, .required = true }},
        .positionals = &.{.{ .name = "input", .value = .string }},
    };
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{}, &sink);
    defer r.deinit();
    try testing.expect(r.value == .errors);
    try testing.expectEqual(@as(usize, 2), sink.count());
    try testing.expectEqual(Sink.Kind.missing_required, sink.items()[0].kind);
    try testing.expectEqualStrings("out", sink.items()[0].arg);
    try testing.expectEqual(Sink.Kind.missing_required, sink.items()[1].kind);
    try testing.expectEqualStrings("input", sink.items()[1].arg);
}

test "missing_value when a value-taking option ends argv" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "in.zig", "--jobs" }, &sink);
    defer r.deinit();
    try testing.expect(r.value == .errors);
    try testing.expectEqual(Sink.Kind.missing_value, sink.items()[0].kind);
}

test "help short-circuits: -h short, --help long, before required checks" {
    const cmd = comptime Spec.Command{
        .name = "toy",
        .options = &.{.{ .long = "out", .value = .string, .required = true }},
    };
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var rs = try parse(testing.allocator, cmd, &.{"-h"}, &sink);
    defer rs.deinit();
    try testing.expect(rs.value == .help);
    try testing.expect(rs.value.help.mode == .short);
    try testing.expectEqualStrings("toy", rs.value.help.path);

    var rl = try parse(testing.allocator, cmd, &.{"--help"}, &sink);
    defer rl.deinit();
    try testing.expect(rl.value == .help);
    try testing.expect(rl.value.help.mode == .long);
    // required `out` never triggered an error because help short-circuits.
    try testing.expect(sink.isEmpty());
}

test "version short-circuits: -V and --version, before required checks" {
    const cmd = comptime Spec.Command{
        .name = "toy",
        .options = &.{.{ .long = "out", .value = .string, .required = true }},
    };
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var rs = try parse(testing.allocator, cmd, &.{"-V"}, &sink);
    defer rs.deinit();
    try testing.expect(rs.value == .version);

    var rl = try parse(testing.allocator, cmd, &.{"--version"}, &sink);
    defer rl.deinit();
    try testing.expect(rl.value == .version);
    try testing.expect(sink.isEmpty());
}

test "conflict and unmet_requirement reported from the sweep" {
    const cmd = comptime Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "a", .value = .boolean, .conflicts = &.{"b"} },
            .{ .long = "b", .value = .boolean },
            .{ .long = "x", .value = .boolean, .requires = &.{"y"} },
            .{ .long = "y", .value = .boolean },
        },
    };
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "--a", "--b", "--x" }, &sink);
    defer r.deinit();
    try testing.expect(r.value == .errors);
    var saw_conflict = false;
    var saw_unmet = false;
    for (sink.items()) |e| {
        if (e.kind == .conflict) saw_conflict = true;
        if (e.kind == .unmet_requirement) saw_unmet = true;
    }
    try testing.expect(saw_conflict);
    try testing.expect(saw_unmet);
}

test "ok path leaks nothing: deinit frees the arena under testing.allocator" {
    const cmd = comptime repCmd();
    var sink: Sink.Sink = .{};
    defer sink.deinit(testing.allocator);

    var r = try parse(testing.allocator, cmd, &.{ "-D", "x", "a.zig", "b.zig" }, &sink);
    r.deinit(); // testing.allocator fails the test on any leak.
    _ = &r;
}
