//! Comptime CLI schema: commands, options, positionals, value types (string/enum/int/float/bool), arity, ranges, and compile-time `validate()`.

const std = @import("std");

pub const Arity = enum { one, optional, variadic };

pub const Action = enum { set, append, set_true, count };

pub fn Range(comptime T: type) type {
    return struct { min: ?T = null, max: ?T = null };
}

pub const ValueType = union(enum) {
    string,
    boolean,
    int: ?Range(i64),
    float: ?Range(f64),
    @"enum": []const [:0]const u8,
};

pub const Option = struct {
    long: ?[:0]const u8 = null,
    short: ?u8 = null,
    value: ValueType = .boolean,
    action: Action = .set,
    required: bool = false,
    help: []const u8 = "",
    long_help: []const u8 = "",
    value_name: []const u8 = "VALUE",
    conflicts: []const []const u8 = &.{},
    requires: []const []const u8 = &.{},
};

pub const Positional = struct {
    name: [:0]const u8,
    value: ValueType = .string,
    arity: Arity = .one,
    help: []const u8 = "",
};

pub const Command = struct {
    name: [:0]const u8,
    about: []const u8 = "",
    long_about: []const u8 = "",
    options: []const Option = &.{},
    positionals: []const Positional = &.{},
    subcommands: []const Command = &.{},
};

pub const Cli = struct {
    name: [:0]const u8,
    version: []const u8,
    about: []const u8 = "",
    root: Command,
};

// Single update site for the auto-injected flags. M6 wires -h/--help and M7
// wires -V/--version; a user Option claiming any of these would silently shadow
// or double-register the injected one, so we reject it at the data-model boundary.
const reserved_long = [_][:0]const u8{ "help", "version" };
const reserved_short = [_]u8{ 'h', 'V' };

/// Pure comptime check: returns the first validation error message, or null if
/// `c` (and all its subcommands) are well-formed. Holds all logic so negative
/// cases are unit-testable without tripping `@compileError`; `validate` is the
/// thin lifter that turns a returned message into a build error.
pub fn checkCommand(comptime c: Command) ?[]const u8 {
    for (c.options) |o| {
        if (o.long == null and o.short == null)
            return "option must set at least one of `long` or `short`";
    }

    for (c.options, 0..) |o, i| {
        if (o.long) |l| {
            for (c.options[0..i]) |prev| {
                if (prev.long) |pl| {
                    if (std.mem.eql(u8, l, pl))
                        return "duplicate long option: --" ++ l;
                }
            }
        }
        if (o.short) |s| {
            for (c.options[0..i]) |prev| {
                if (prev.short) |ps| {
                    if (s == ps)
                        return "duplicate short option: -" ++ &[_]u8{s};
                }
            }
        }
    }

    for (c.options) |o| {
        if (o.long) |l| {
            for (reserved_long) |r| {
                if (std.mem.eql(u8, l, r))
                    return "option long name --" ++ l ++ " is reserved for the auto-injected flag";
            }
        }
        if (o.short) |s| {
            for (reserved_short) |r| {
                if (s == r)
                    return "option short name -" ++ &[_]u8{s} ++ " is reserved for the auto-injected flag";
            }
        }
    }

    for (c.options) |o| {
        if (o.value == .@"enum" and o.value.@"enum".len == 0)
            return "option enum value type must have at least one choice";
    }
    for (c.positionals) |p| {
        if (p.value == .@"enum" and p.value.@"enum".len == 0)
            return "positional enum value type must have at least one choice";
    }

    {
        var seen_variadic = false;
        for (c.positionals) |p| {
            if (seen_variadic)
                return "positional after a variadic is not allowed (variadic must be last)";
            if (p.arity == .variadic) seen_variadic = true;
        }
    }
    {
        // A required positional after an optional/variadic one can never bind:
        // the greedy left-to-right fill satisfies the optional first, so the
        // required slot spuriously reports "missing" on inputs meant for it.
        var seen_non_required = false;
        for (c.positionals) |p| {
            if (seen_non_required and p.arity == .one)
                return "required positional after an optional or variadic one is not allowed (required positionals must come first)";
            if (p.arity != .one) seen_non_required = true;
        }
    }
    {
        var variadic_count: usize = 0;
        for (c.positionals) |p| {
            if (p.arity == .variadic) variadic_count += 1;
        }
        if (variadic_count > 1)
            return "at most one variadic positional is allowed";
    }

    for (c.positionals, 0..) |p, i| {
        for (c.positionals[0..i]) |prev| {
            if (std.mem.eql(u8, p.name, prev.name))
                return "duplicate positional name: " ++ p.name;
        }
    }

    for (c.subcommands, 0..) |s, i| {
        for (c.subcommands[0..i]) |prev| {
            if (std.mem.eql(u8, s.name, prev.name))
                return "duplicate subcommand name: " ++ s.name;
        }
    }

    // conflicts/requires are string references into this command's own options;
    // a typo names an option that never exists and becomes a silent no-op, so
    // cross-check every reference at comptime.
    for (c.options) |o| {
        for (o.conflicts) |ref| {
            if (!hasOptionNamed(c, ref))
                return "conflicts references unknown option: " ++ ref;
        }
        for (o.requires) |ref| {
            if (!hasOptionNamed(c, ref))
                return "requires references unknown option: " ++ ref;
        }
    }

    for (c.subcommands) |s| {
        if (checkCommand(s)) |m| return m;
    }

    return null;
}

fn hasOptionNamed(comptime c: Command, comptime name: []const u8) bool {
    for (c.options) |o| {
        if (o.long) |l| {
            if (std.mem.eql(u8, l, name)) return true;
        }
        if (o.short) |s| {
            if (name.len == 1 and name[0] == s) return true;
        }
    }
    return false;
}

/// Compile-time assertion that `c` is well-formed. Emits no code and no bytes
/// for valid input; fires `@compileError` with the first problem otherwise.
pub fn validate(comptime c: Command) void {
    comptime {
        if (checkCommand(c)) |m| @compileError(m);
    }
}

test "valid representative spec passes" {
    const spec = Cli{
        .name = "toy",
        .version = "0.1.0",
        .about = "a toy compiler",
        .root = .{
            .name = "toy",
            .about = "root",
            .options = &.{
                .{ .long = "all", .short = 'a', .help = "build everything" },
                .{ .long = "level", .short = 'l', .value = .{ .int = .{ .min = 0, .max = 3 } } },
                .{ .long = "mode", .value = .{ .@"enum" = &.{ "fast", "small" } } },
            },
            .positionals = &.{
                .{ .name = "input", .value = .string },
                .{ .name = "extra", .arity = .variadic },
            },
            .subcommands = &.{
                .{
                    .name = "build",
                    .options = &.{
                        .{ .long = "release", .short = 'r' },
                    },
                    .positionals = &.{
                        .{ .name = "target" },
                    },
                },
            },
        },
    };
    comptime validate(spec.root);
    try std.testing.expect(comptime checkCommand(spec.root) == null);
}

test "structure and field assertions" {
    try std.testing.expectEqual(3, @typeInfo(Arity).@"enum".fields.len);
    try std.testing.expectEqual(4, @typeInfo(Action).@"enum".fields.len);
    try std.testing.expectEqual(5, @typeInfo(ValueType).@"union".fields.len);

    try std.testing.expect(@hasField(Option, "long"));
    try std.testing.expect(@hasField(Option, "short"));
    try std.testing.expect(@hasField(Option, "value"));
    try std.testing.expect(@hasField(Option, "action"));
    try std.testing.expect(@hasField(Option, "required"));
    try std.testing.expect(@hasField(Option, "conflicts"));
    try std.testing.expect(@hasField(Option, "requires"));

    try std.testing.expect(@hasField(Positional, "name"));
    try std.testing.expect(@hasField(Positional, "arity"));

    try std.testing.expect(@hasField(Command, "name"));
    try std.testing.expect(@hasField(Command, "options"));
    try std.testing.expect(@hasField(Command, "positionals"));
    try std.testing.expect(@hasField(Command, "subcommands"));

    try std.testing.expect(@hasField(Cli, "name"));
    try std.testing.expect(@hasField(Cli, "version"));
    try std.testing.expect(@hasField(Cli, "root"));

    const R = Range(i64);
    try std.testing.expect(@hasField(R, "min"));
    try std.testing.expect(@hasField(R, "max"));
}

test "reject option with neither long nor short" {
    const c = Command{
        .name = "c",
        .options = &.{.{ .value = .boolean }},
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "must set at least one") != null);
}

test "reject duplicate long option" {
    const c = Command{
        .name = "c",
        .options = &.{
            .{ .long = "verbose" },
            .{ .long = "verbose" },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "duplicate long") != null);
}

test "reject duplicate short option" {
    const c = Command{
        .name = "c",
        .options = &.{
            .{ .short = 'v' },
            .{ .short = 'v' },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "duplicate short") != null);
}

test "reject empty enum choices on option" {
    const c = Command{
        .name = "c",
        .options = &.{.{ .long = "mode", .value = .{ .@"enum" = &.{} } }},
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "option enum") != null);
}

test "reject empty enum choices on positional" {
    const c = Command{
        .name = "c",
        .positionals = &.{.{ .name = "mode", .value = .{ .@"enum" = &.{} } }},
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "positional enum") != null);
}

test "reject positional after variadic" {
    const c = Command{
        .name = "c",
        .positionals = &.{
            .{ .name = "files", .arity = .variadic },
            .{ .name = "trailing" },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "variadic must be last") != null);
}

test "reject required positional after an optional one" {
    const c = Command{
        .name = "c",
        .positionals = &.{
            .{ .name = "maybe", .arity = .optional },
            .{ .name = "must", .arity = .one },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "required positionals must come first") != null);
}

test "accept required-then-optional-then-variadic order" {
    const c = Command{
        .name = "c",
        .positionals = &.{
            .{ .name = "req", .arity = .one },
            .{ .name = "maybe", .arity = .optional },
            .{ .name = "rest", .arity = .variadic },
        },
    };
    try std.testing.expect(comptime checkCommand(c) == null);
}

test "reject two variadics" {
    const c = Command{
        .name = "c",
        .positionals = &.{
            .{ .name = "a", .arity = .variadic },
            .{ .name = "b", .arity = .variadic },
        },
    };
    // The positional-after-variadic check catches the second variadic first;
    // either message is a rejection, which is what we assert.
    try std.testing.expect(comptime checkCommand(c) != null);
}

test "reject reserved long flag" {
    const c = Command{
        .name = "c",
        .options = &.{.{ .long = "help" }},
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "reserved") != null);
}

test "reject reserved short flag" {
    const c = Command{
        .name = "c",
        .options = &.{.{ .short = 'V' }},
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "reserved") != null);
}

test "reject duplicate subcommand name" {
    const c = Command{
        .name = "c",
        .subcommands = &.{
            .{ .name = "build" },
            .{ .name = "build" },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "duplicate subcommand") != null);
}

test "reject duplicate positional name" {
    const c = Command{
        .name = "c",
        .positionals = &.{
            .{ .name = "src" },
            .{ .name = "src" },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "duplicate positional") != null);
}

test "reject conflicts referencing unknown option" {
    const c = Command{
        .name = "c",
        .options = &.{
            .{ .long = "a", .conflicts = &.{"nope"} },
            .{ .long = "b" },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "conflicts references unknown") != null);
}

test "reject requires referencing unknown option" {
    const c = Command{
        .name = "c",
        .options = &.{
            .{ .long = "a", .requires = &.{"ghost"} },
            .{ .long = "b" },
        },
    };
    const msg = comptime checkCommand(c);
    try std.testing.expect(msg != null);
    try std.testing.expect(std.mem.indexOf(u8, msg.?, "requires references unknown") != null);
}

test "accept conflicts referencing existing option by long or short" {
    const c = Command{
        .name = "c",
        .options = &.{
            .{ .long = "a", .short = 'a', .conflicts = &.{"b"}, .requires = &.{"a"} },
            .{ .long = "b" },
        },
    };
    try std.testing.expect(comptime checkCommand(c) == null);
}

test "error in subcommand is reported via recursion" {
    const c = Command{
        .name = "root",
        .subcommands = &.{
            .{
                .name = "child",
                .options = &.{.{ .value = .boolean }},
            },
        },
    };
    try std.testing.expect(comptime checkCommand(c) != null);
}
