//! Plain-text help and version output: usage lines, per-command and whole-CLI help (short and long), and `--version`.
//!
//! Renders straight from the comptime `Spec` — no runtime schema, no `Parser`
//! dependency. The driver maps a parse `Result`'s `.help`/`.version` outcome to
//! these entry points. Output is intentionally colourless: the cli⊥term layering
//! rule forbids importing `term/*`, and plain text keeps help byte-identical run
//! to run so it can sit under the snapshot gate without escape-sequence noise.

const std = @import("std");
const Spec = @import("Spec.zig");

pub const Mode = enum { short, long };

/// Two spaces between the left (flag/arg) column and the help column, matching
/// the conventional clap/argparse gutter.
const gutter = "  ";

/// `Usage: <name> [OPTIONS] <ARGS...>`, synthesized from the spec. Required
/// positionals appear bare, optional ones in brackets, and a variadic gains a
/// trailing `...`. `[OPTIONS]` is always emitted (every command has at least the
/// auto-injected `-h`/`-V`), and `<COMMAND>` when the command has subcommands.
pub fn renderUsage(comptime cmd: Spec.Command, out: *std.Io.Writer) !void {
    try out.print("Usage: {s}", .{cmd.name});
    try out.writeAll(" [OPTIONS]");
    if (cmd.subcommands.len != 0) try out.writeAll(" <COMMAND>");
    inline for (cmd.positionals) |p| {
        try out.writeByte(' ');
        try out.writeAll(usageArg(p));
    }
    try out.writeByte('\n');
}

/// Full help for a single command. SHORT uses `about` + per-item `help`; LONG
/// uses `long_about` (falling back to `about`) + per-item `long_help` (falling
/// back to `help`). Section order is fixed: usage, description, Commands,
/// Arguments, Options. The left column is padded to a single comptime-computed
/// width so every help column aligns and the output is byte-stable.
pub fn renderHelp(comptime cmd: Spec.Command, out: *std.Io.Writer, mode: Mode) !void {
    try renderUsage(cmd, out);

    const desc = switch (mode) {
        .short => cmd.about,
        .long => if (cmd.long_about.len != 0) cmd.long_about else cmd.about,
    };
    if (desc.len != 0) {
        try out.writeByte('\n');
        try out.writeAll(desc);
        try out.writeByte('\n');
    }

    const width = comptime leftWidth(cmd);

    if (cmd.subcommands.len != 0) {
        try out.writeAll("\nCommands:\n");
        inline for (cmd.subcommands) |s| {
            // Subcommands carry only `about`; there is no long variant to fall
            // back to, so both modes render the same one line here.
            try renderRow(out, s.name, s.about, "", width);
        }
    }

    if (cmd.positionals.len != 0) {
        try out.writeAll("\nArguments:\n");
        inline for (cmd.positionals) |p| {
            try renderRow(out, positionalLabel(p), rowHelp(p.help, "", mode), "", width);
        }
    }

    try out.writeAll("\nOptions:\n");
    inline for (cmd.options) |o| {
        try renderRow(out, optionLabel(o), rowHelp(o.help, o.long_help, mode), enumSuffix(o), width);
    }
    // The auto-injected flags are not in the spec's option list; list them last
    // so they always appear, matching the reservation in `Spec`.
    try renderRow(out, "-h, --help", "Print help", "", width);
    try renderRow(out, "-V, --version", "Print version", "", width);
}

/// `<name> <version>\n` for `--version`.
pub fn renderVersion(comptime cli: Spec.Cli, out: *std.Io.Writer) !void {
    try out.print("{s} {s}\n", .{ cli.name, cli.version });
}

/// Pick the help text for a row: `long_help` in LONG mode, falling back to
/// `help` when it is empty; always `help` in SHORT mode.
fn rowHelp(comptime help: []const u8, comptime long_help: []const u8, mode: Mode) []const u8 {
    return switch (mode) {
        .short => help,
        .long => if (long_help.len != 0) long_help else help,
    };
}

fn renderRow(
    out: *std.Io.Writer,
    comptime label: []const u8,
    help: []const u8,
    comptime suffix: []const u8,
    comptime width: usize,
) !void {
    try out.writeAll("  ");
    try out.writeAll(label);
    // `suffix` lands in the (last) help column, so it never disturbs the left-column
    // alignment `width` measures — an enum option with empty `help` still needs its
    // choices rendered, hence the two-part guard.
    if (help.len != 0 or suffix.len != 0) {
        try out.splatByteAll(' ', width - label.len);
        try out.writeAll(gutter);
        try out.writeAll(help);
        try out.writeAll(suffix);
    }
    try out.writeByte('\n');
}

/// The ` (one of: a|b|c)` help-column tail for an enum option, empty for every
/// other value type. Driven off the Spec's own choices so help and the
/// invalid-value error list the same set.
fn enumSuffix(comptime o: Spec.Option) []const u8 {
    return switch (o.value) {
        .@"enum" => |choices| " (one of: " ++ Spec.choicesJoined(choices) ++ ")",
        else => "",
    };
}

/// The left-column label for a positional in the Arguments section: the name in
/// angle brackets, plus `...` for a variadic.
fn positionalLabel(comptime p: Spec.Positional) []const u8 {
    return "<" ++ p.name ++ ">" ++ (if (p.arity == .variadic) "..." else "");
}

/// The usage-line spelling of a positional: bare when required, bracketed when
/// optional, with a trailing `...` on a variadic.
fn usageArg(comptime p: Spec.Positional) []const u8 {
    return switch (p.arity) {
        .one => "<" ++ p.name ++ ">",
        .optional => "[" ++ p.name ++ "]",
        .variadic => "[" ++ p.name ++ "]...",
    };
}

/// The left-column label for an option: `-s, --long <VALUE>`. Short and long are
/// both shown when present; the value name is appended only for value-taking
/// options (booleans and counters take none).
fn optionLabel(comptime o: Spec.Option) []const u8 {
    comptime {
        var s: []const u8 = "";
        if (o.short) |c| {
            s = s ++ "-" ++ [_]u8{c};
            if (o.long != null) s = s ++ ", ";
        } else {
            // Pad the missing short slot so long-only options line up under the
            // `--` column with those that have a short.
            s = s ++ "    ";
        }
        if (o.long) |l| s = s ++ "--" ++ l;
        if (takesValue(o)) s = s ++ " <" ++ o.value_name ++ ">";
        return s;
    }
}

fn takesValue(comptime o: Spec.Option) bool {
    return switch (o.value) {
        .boolean => o.action != .set_true and o.action != .count,
        else => true,
    };
}

/// Widest left-column label across every row this command will render,
/// including the auto-injected flags, so all help columns align.
fn leftWidth(comptime cmd: Spec.Command) usize {
    comptime {
        var w: usize = 0;
        for (cmd.subcommands) |s| w = @max(w, s.name.len);
        for (cmd.positionals) |p| w = @max(w, positionalLabel(p).len);
        for (cmd.options) |o| w = @max(w, optionLabel(o).len);
        w = @max(w, "-h, --help".len);
        w = @max(w, "-V, --version".len);
        return w;
    }
}

// ---- tests ----

const testing = std.testing;

const golden_cli = Spec.Cli{
    .name = "toy",
    .version = "0.1.0",
    .about = "a toy compiler",
    .root = .{
        .name = "toy",
        .about = "build and run toy programs",
        .long_about = "Build, check, and run toy programs from source.",
        .options = &.{
            .{ .long = "verbose", .short = 'v', .action = .set_true, .help = "Enable verbose output" },
            .{ .long = "jobs", .short = 'j', .value = .{ .int = null }, .value_name = "N", .help = "Parallel jobs", .long_help = "Number of parallel jobs to run" },
            .{ .long = "mode", .value = .{ .@"enum" = &.{ "fast", "small" } }, .value_name = "MODE", .help = "Optimization mode" },
            .{ .long = "define", .short = 'D', .action = .append, .value_name = "KEY=VAL", .help = "Define a variable" },
            .{ .long = "out", .short = 'o', .required = true, .value_name = "FILE", .help = "Output file" },
        },
        .positionals = &.{
            .{ .name = "input", .value = .string, .help = "Input source file" },
            .{ .name = "extra", .arity = .variadic, .help = "Extra files" },
        },
        .subcommands = &.{
            .{ .name = "build", .about = "Compile a program" },
        },
    },
};

test "renderUsage is a frozen, deterministic string" {
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderUsage(golden_cli.root, &w);
    try testing.expectEqualStrings(
        "Usage: toy [OPTIONS] <COMMAND> <input> [extra]...\n",
        w.buffered(),
    );
}

test "renderVersion is a frozen string" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderVersion(golden_cli, &w);
    try testing.expectEqualStrings("toy 0.1.0\n", w.buffered());
}

test "renderHelp short is a frozen, column-stable string" {
    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderHelp(golden_cli.root, &w, .short);
    try testing.expectEqualStrings(
        \\Usage: toy [OPTIONS] <COMMAND> <input> [extra]...
        \\
        \\build and run toy programs
        \\
        \\Commands:
        \\  build                   Compile a program
        \\
        \\Arguments:
        \\  <input>                 Input source file
        \\  <extra>...              Extra files
        \\
        \\Options:
        \\  -v, --verbose           Enable verbose output
        \\  -j, --jobs <N>          Parallel jobs
        \\      --mode <MODE>       Optimization mode (one of: fast|small)
        \\  -D, --define <KEY=VAL>  Define a variable
        \\  -o, --out <FILE>        Output file
        \\  -h, --help              Print help
        \\  -V, --version           Print version
        \\
    , w.buffered());
}

test "renderHelp long uses long_about and long_help fallbacks" {
    var buf: [1024]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderHelp(golden_cli.root, &w, .long);
    try testing.expectEqualStrings(
        \\Usage: toy [OPTIONS] <COMMAND> <input> [extra]...
        \\
        \\Build, check, and run toy programs from source.
        \\
        \\Commands:
        \\  build                   Compile a program
        \\
        \\Arguments:
        \\  <input>                 Input source file
        \\  <extra>...              Extra files
        \\
        \\Options:
        \\  -v, --verbose           Enable verbose output
        \\  -j, --jobs <N>          Number of parallel jobs to run
        \\      --mode <MODE>       Optimization mode (one of: fast|small)
        \\  -D, --define <KEY=VAL>  Define a variable
        \\  -o, --out <FILE>        Output file
        \\  -h, --help              Print help
        \\  -V, --version           Print version
        \\
    , w.buffered());
}

test "rendered output contains no escape bytes" {
    var buf: [1024]u8 = undefined;

    var uw = std.Io.Writer.fixed(&buf);
    try renderUsage(golden_cli.root, &uw);
    try testing.expect(std.mem.indexOfScalar(u8, uw.buffered(), 0x1b) == null);

    var hw = std.Io.Writer.fixed(&buf);
    try renderHelp(golden_cli.root, &hw, .short);
    try testing.expect(std.mem.indexOfScalar(u8, hw.buffered(), 0x1b) == null);

    var lw = std.Io.Writer.fixed(&buf);
    try renderHelp(golden_cli.root, &lw, .long);
    try testing.expect(std.mem.indexOfScalar(u8, lw.buffered(), 0x1b) == null);

    var vw = std.Io.Writer.fixed(&buf);
    try renderVersion(golden_cli, &vw);
    try testing.expect(std.mem.indexOfScalar(u8, vw.buffered(), 0x1b) == null);
}

test "optional positional is bracketed; long-only value-less option shows no value" {
    const cmd = comptime Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "dry-run", .action = .set_true, .help = "Do nothing" }, // long-only, no value
        },
        .positionals = &.{
            .{ .name = "maybe", .value = .string, .arity = .optional, .help = "maybe file" },
        },
    };
    var buf: [512]u8 = undefined;

    var uw = std.Io.Writer.fixed(&buf);
    try renderUsage(cmd, &uw);
    // optional positional appears bracketed in the usage line (the .optional branch)
    try testing.expect(std.mem.indexOf(u8, uw.buffered(), "[maybe]") != null);

    var hw = std.Io.Writer.fixed(&buf);
    try renderHelp(cmd, &hw, .short);
    const out = hw.buffered();
    // long-only option: the short slot is padded and no <VALUE> is appended
    try testing.expect(std.mem.indexOf(u8, out, "    --dry-run") != null);
    try testing.expect(std.mem.indexOf(u8, out, "--dry-run <") == null);
}

test "enum option help shows its choices" {
    const cmd = comptime Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "mode", .value = .{ .@"enum" = &.{ "fast", "small" } }, .value_name = "MODE", .help = "Optimization mode" },
        },
    };
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderHelp(cmd, &w, .short);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "(one of: fast|small)") != null);
}

test "enum option with empty help still renders its choices" {
    const cmd = comptime Spec.Command{
        .name = "c",
        .options = &.{
            .{ .long = "mode", .value = .{ .@"enum" = &.{ "fast", "small" } } },
        },
    };
    var buf: [512]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderHelp(cmd, &w, .short);
    try testing.expect(std.mem.indexOf(u8, w.buffered(), "(one of: fast|small)") != null);
}
