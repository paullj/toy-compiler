//! The `toy` compiler's CLI SCHEMA — a comptime `cli.Spec.Cli` the driver
//! (`main.zig`) parses with `cli.Parser`. This is the APP spec (it lives in the
//! driver, NOT in the `cli/*` framework); `main.zig` imports it as a sibling
//! `@import("Cli.zig")`, distinct from the framework's `toyc.cli.Cli` facade.
//!
//! Shape (locked): `build` (compile to a signed executable; the default action)
//! and `run` (build, then exec + report exit status) are real SUBCOMMANDS that
//! share the root command's option set + variadic `<file>` positional. The root
//! with no verb is the default build/inspect. The shared options + positional are
//! defined once as comptime consts and referenced from root/build/run so all three
//! carry the IDENTICAL surface.
//!
//! Every spelling + semantic here is chosen to match the pre-framework hand-rolled
//! parser BYTE-FOR-BYTE (examples/ scripts + diff.sh depend on them):
//!   - `-O` is SHORT-ONLY (int Range 0..1) so `-O0`/`-O1`/`-O 1` all parse and its
//!     Parsed field is `O`; `-O2` is now rejected as a bad value (strictly safer).
//!   - `--opt`/`--no-opt` carry pass-name LISTS the driver validates (the grammar
//!     can't express pass-name validity), so they are plain strings here.
//!   - `--color` is the new AUTO/NO_COLOR-aware colour knob (default auto).
//! The reserved `-h`/`--help`/`-V`/`--version` flags are AUTO-INJECTED by the
//! framework (`Spec.checkCommand` rejects them in the option list), so they are
//! NOT listed here.

const toyc = @import("toy_compiler");
const Spec = toyc.cli.Spec;

/// The option set shared verbatim by root, `build`, and `run` — defined once so
/// all three carry the identical surface (a divergence would be a spelling drift).
/// `--target` is deliberately NOT defaulted in-spec (the field stays `?[]const u8`);
/// the driver applies "native" when it is null, matching the old behaviour.
const shared_opts: []const Spec.Option = &.{
    .{ .long = "output", .short = 'o', .value = .string, .value_name = "PATH", .help = "Output path for the built binary (default: .toy/<stamp>/build/<name>)" },
    .{ .long = "target", .value = .string, .value_name = "TRIPLE", .help = "Compilation target (default: native)" },
    .{ .long = "emit", .value = .{ .@"enum" = &.{ "lex", "parse", "check", "ir" } }, .value_name = "STAGE", .help = "Inspect the pipeline instead of building (no binary)" },
    // -j: N>=1; the parser rejects -j0/negatives as a bad value via the Range.
    .{ .short = 'j', .value = .{ .int = .{ .min = 1, .max = null } }, .value_name = "N", .help = "Build worker threads (N>=1; -j1 = serial; default cpu-based)" },
    // -O is short-only => Parsed field `O`; -O0 / -O1 / -O 1 all parse, -O2 rejected.
    .{ .short = 'O', .value = .{ .int = .{ .min = 0, .max = 1 } }, .value_name = "LEVEL", .help = "IR optimization level (0 = none, 1 = all passes)" },
    .{ .long = "opt", .value = .string, .value_name = "LIST", .help = "Enable only these passes (fold,branch,dce,forward)" },
    .{ .long = "no-opt", .value = .string, .action = .append, .value_name = "PASS", .help = "Disable one pass from the current level (repeatable)" },
    .{ .long = "dump", .action = .set_true, .help = "Print the emit phase's artifact (tokens, or the AST)" },
    .{ .long = "codegen-stats", .action = .set_true, .help = "Print compiled-vs-cached function counts (with -o)" },
    .{ .long = "verify", .action = .set_true, .help = "Re-lower cached functions and assert they match (with -o)" },
    .{ .long = "force", .action = .set_true, .help = "Ignore the codegen cache; lower every function (with -o)" },
    .{ .long = "opt-stats", .action = .set_true, .help = "Print per-pass opt counters + dual metric (with -o; use --force)" },
    .{ .long = "timings", .action = .set_true, .help = "Print the per-stage wall-clock profile" },
    .{ .long = "color", .value = .{ .@"enum" = &.{ "auto", "always", "never" } }, .value_name = "WHEN", .help = "Colorize output (default auto: on when stdout is a tty)" },
};

/// The variadic input-file positional, shared by root/build/run.
const files_pos: []const Spec.Positional = &.{
    .{ .name = "file", .value = .string, .arity = .variadic, .help = "Input source file(s)" },
};

/// The whole `toy` CLI schema. `version` is `toyc.version.semver`
/// (`build_options.semver`, a comptime `[]const u8`) — the runtime `version.stamp()`
/// needs a buffer and so cannot feed a comptime spec; `--version` therefore prints
/// `toy <semver>` (see main.zig's deviation note).
pub const spec: Spec.Cli = .{
    .name = "toy",
    .version = toyc.version.semver,
    .about = "toy compiler (lexer + parser + name resolution + typecheck + codegen)",
    .root = .{
        .name = "toy",
        .about = "Build, check, and run toy programs from source",
        .long_about = "Compile toy programs. With no subcommand, a bare `toy <file>` builds a signed executable; --emit lex|parse|check|ir inspects the pipeline instead. build and run force a build.",
        .options = shared_opts,
        .positionals = files_pos,
        .subcommands = &.{
            .{ .name = "build", .about = "Compile to a signed executable (the default action)", .options = shared_opts, .positionals = files_pos },
            .{ .name = "run", .about = "Build, then execute the binary and report its exit status", .options = shared_opts, .positionals = files_pos },
        },
    },
};

// A malformed spec is a BUILD error (not a test failure): `Spec.validate` fires
// `@compileError` with the first problem at comptime.
comptime {
    Spec.validate(spec.root);
}
