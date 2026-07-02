//! The `toy` compiler's CLI schema: a comptime `cli.Spec.Cli` the driver
//! (`main.zig`) parses with `cli.Parser`. `build`/`run` are subcommands sharing
//! the root's option set + variadic `<file>` positional; a bare `toy` builds/inspects.
//!
//! Spellings match the old hand-rolled parser byte-for-byte (examples/ + diff.sh depend on them):
//! - `-O` is short-only (Range 0..1) so `-O0`/`-O1`/`-O 1` parse (`-O2` rejected).
//! - `--opt`/`--no-opt` are plain strings (the grammar can't validate pass names; the driver does).
//! - `-h`/`--help`/`-V`/`--version` are auto-injected by the framework, so not listed here.

const toyc = @import("toy_compiler");
const Spec = toyc.cli.Spec;

/// The option set shared verbatim by root, `build`, and `run` (defined once so
/// they can't drift). `--target` is not defaulted in-spec (stays `?[]const u8`);
/// the driver applies "native" when null, matching the old behaviour.
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
    // Render-time severity overrides (C3). Repeatable; take a code (R0001) or a band
    // letter (L/P/R/T). Fixed precedence ignore > warn > error; last match wins.
    // Render-only: they NEVER change the exit status.
    .{ .long = "error", .value = .string, .action = .append, .value_name = "CODE", .help = "Treat a diagnostic code or band as an error (repeatable, e.g. --error R0001 or --error T; render-only)" },
    .{ .long = "warn", .value = .string, .action = .append, .value_name = "CODE", .help = "Downgrade a diagnostic code or band to a warning (repeatable; render-only)" },
    .{ .long = "ignore", .value = .string, .action = .append, .value_name = "CODE", .help = "Suppress a diagnostic code or band from output (repeatable; render-only)" },
};

/// The variadic input-file positional, shared by root/build/run.
const files_pos: []const Spec.Positional = &.{
    .{ .name = "file", .value = .string, .arity = .variadic, .help = "Input source file(s)" },
};

/// `check`'s option set: the shared build/inspect options PLUS the check-only knobs.
/// `--format` picks the diagnostic wire form (pretty snippets vs stable NDJSON);
/// `--exit-zero` forces a 0 exit even with errors (for editors that read the stream,
/// not the status); `--watch` is accepted as a forward-compatible no-op stub so the
/// schema is stable when an incremental watch loop lands.
const check_opts: []const Spec.Option = shared_opts ++ [_]Spec.Option{
    .{ .long = "format", .value = .{ .@"enum" = &.{ "human", "ndjson" } }, .value_name = "FORM", .help = "Diagnostic output form (human snippets or line-delimited JSON; default human)" },
    .{ .long = "exit-zero", .action = .set_true, .help = "Always exit 0, even when diagnostics contain errors" },
    .{ .long = "watch", .action = .set_true, .help = "Re-check on file changes (not yet implemented; accepted as a no-op)" },
};

/// The single `<CODE>` positional for `toy explain` (e.g. `R0001`).
const explain_pos: []const Spec.Positional = &.{
    .{ .name = "code", .value = .string, .arity = .one, .help = "A diagnostic code, e.g. R0001" },
};

/// The whole `toy` CLI schema. `version` is the comptime `toyc.version.semver`
/// (the runtime `version.stamp()` needs a buffer, so can't feed a comptime spec);
/// `--version` therefore prints `toy <semver>` (see main.zig's deviation note).
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
            .{ .name = "check", .about = "Check syntax and types without building; report all diagnostics", .options = check_opts, .positionals = files_pos },
            .{ .name = "explain", .about = "Print the documentation for a diagnostic code", .positionals = explain_pos },
        },
    },
};

// A malformed spec is a build error: `Spec.validate` fires `@compileError`. The
// larger option set (the C3 --error/--warn/--ignore flags added here) pushes the
// comptime cross-check over the default 1000-branch budget, so raise the quota at this
// caller-side comptime site (not in the framework).
comptime {
    @setEvalBranchQuota(20_000);
    Spec.validate(spec.root);
}
