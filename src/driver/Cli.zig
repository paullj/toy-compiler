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
// Build-time DEV switch (see build.zig): gates the `--emit lex|parse|ir`
// pipeline-inspection flag on/off. Read at comptime so a release binary (dev_inspect
// false) never even registers `--emit`; the default Debug build (and every test
// binary) has it on, so the existing `--emit lex|parse` dump tests keep working.
// Reached through the library module's `version` (the single `build_options` owner) —
// importing `build_options` here directly would put that generated file in two modules.
pub const dev_inspect: bool = toyc.version.dev_inspect;

/// The option set shared verbatim by root, `build`, and `run` (defined once so
/// they can't drift). `--target` is not defaulted in-spec (stays `?[]const u8`);
/// the driver applies "native" when null, matching the old behaviour.
// The DEV pipeline-inspection flag, registered on root/build/run ONLY when
// `dev_inspect` is true (Debug builds + every test binary). The value list is
// `{lex,parse,ir}` — NOT `check`: `toy check` is its own subcommand now, so
// `toy --emit check` is rejected as an invalid value. A release build (dev_inspect
// false) drops this entirely, so the binary exposes no `--emit` at all.
const emit_opt: []const Spec.Option = if (dev_inspect) &.{
    .{ .long = "emit", .value = .{ .@"enum" = &.{ "lex", "parse", "ir" } }, .value_name = "STAGE", .help = "Inspect the pipeline instead of building (no binary; dev builds only)" },
} else &.{};

// The build/inspect options MINUS `--emit` (spliced in via `emit_opt` so it can be
// gated behind `dev_inspect`). `shared_opts` below is `pre_emit ++ emit_opt ++ post_emit`.
const pre_emit: []const Spec.Option = &.{
    .{ .long = "output", .short = 'o', .value = .string, .value_name = "PATH", .help = "Output path for the built binary (default: .toy/<stamp>/build/<name>)" },
    .{ .long = "target", .value = .string, .value_name = "TRIPLE", .help = "Compilation target (default: native)" },
};
const post_emit: []const Spec.Option = &.{
    // -j: N>=1; the parser rejects -j0/negatives as a bad value via the Range.
    .{ .short = 'j', .value = .{ .int = .{ .min = 1, .max = null } }, .value_name = "N", .help = "Build worker threads (N>=1; -j1 = serial; default cpu-based)" },
    // -O is short-only => Parsed field `O`; -O0 / -O1 / -O 1 all parse, -O2 rejected.
    .{ .short = 'O', .value = .{ .int = .{ .min = 0, .max = 1 } }, .value_name = "LEVEL", .help = "IR optimization level (0 = none, 1 = all passes)" },
    .{ .long = "opt", .value = .string, .value_name = "LIST", .help = "Enable only these passes (fold,branch,dce,forward)" },
    .{ .long = "no-opt", .value = .string, .action = .append, .value_name = "PASS", .help = "Disable one pass from the current level (repeatable)" },
    .{ .long = "dump", .action = .set_true, .help = "Print the emit phase's artifact (tokens, or the AST)" },
    .{ .long = "codegen-stats", .action = .set_true, .help = "Print compiled-vs-cached function counts (with -o)" },
    .{ .long = "verify", .action = .set_true, .help = "Re-lower cached functions and assert they match (with -o)" },
    .{ .long = "no-cache", .action = .set_true, .help = "Ignore the codegen cache; lower every function (with -o)" },
    .{ .long = "opt-stats", .action = .set_true, .help = "Print per-pass opt counters + dual metric (with -o; use --no-cache)" },
    .{ .long = "timings", .action = .set_true, .help = "Print the per-stage wall-clock profile" },
    .{ .long = "color", .value = .{ .@"enum" = &.{ "auto", "always", "never" } }, .value_name = "WHEN", .help = "Colorize output (default auto: on when stdout is a tty)" },
    // Render-time severity overrides. Repeatable; take a code (R0001) or a band
    // letter (L/P/R/T). Fixed precedence ignore > warn > error; last match wins.
    // Render-only: they NEVER change the exit status.
    .{ .long = "error", .value = .string, .action = .append, .value_name = "CODE", .help = "Treat a diagnostic code or band as an error (repeatable, e.g. --error R0001 or --error T; render-only)" },
    .{ .long = "warn", .value = .string, .action = .append, .value_name = "CODE", .help = "Downgrade a diagnostic code or band to a warning (repeatable; render-only)" },
    .{ .long = "ignore", .value = .string, .action = .append, .value_name = "CODE", .help = "Suppress a diagnostic code or band from output (repeatable; render-only)" },
};

/// The option set shared verbatim by root, `build`, and `run` (defined once so
/// they can't drift). `--emit` sits between `pre_emit` and `post_emit` and is present
/// only when `dev_inspect` is true (Debug builds + test binaries); a release build
/// drops it entirely. `--target` is not defaulted in-spec (stays `?[]const u8`); the
/// driver applies "native" when null, matching the old behaviour.
const shared_opts: []const Spec.Option = pre_emit ++ emit_opt ++ post_emit;

/// The variadic input-file positional, shared by root/build/run.
const files_pos: []const Spec.Positional = &.{
    .{ .name = "file", .value = .string, .arity = .variadic, .help = "Input source file(s)" },
};

/// `check`'s OWN option set — deliberately NOT the shared build/inspect options. `check`
/// reports diagnostics without building, so nothing codegen/emit/opt related belongs
/// here: only the diagnostic-shaping + input knobs. `--format` picks the wire form
/// (pretty snippets vs stable NDJSON); `--error/--warn/--ignore` are the repeatable
/// severity overrides; `--error-on-warning` promotes any surviving warning to an error
/// for the exit gate; `--exit-zero` forces a 0 exit even with errors (editors that read
/// the stream, not the status); `--watch` is a forward-compatible no-op stub; `--color`
/// and `--target` mirror the shared spellings.
const check_opts: []const Spec.Option = &.{
    .{ .long = "format", .value = .{ .@"enum" = &.{ "human", "ndjson" } }, .value_name = "FORM", .help = "Diagnostic output form (human snippets or line-delimited JSON; default human)" },
    .{ .long = "error", .value = .string, .action = .append, .value_name = "CODE", .help = "Treat a diagnostic code or band as an error (repeatable, e.g. --error R0001 or --error T; render-only)" },
    .{ .long = "warn", .value = .string, .action = .append, .value_name = "CODE", .help = "Downgrade a diagnostic code or band to a warning (repeatable; render-only)" },
    .{ .long = "ignore", .value = .string, .action = .append, .value_name = "CODE", .help = "Suppress a diagnostic code or band from output (repeatable; render-only)" },
    .{ .long = "error-on-warning", .action = .set_true, .help = "Exit non-zero if any warning survives the severity config" },
    .{ .long = "exit-zero", .action = .set_true, .help = "Always exit 0, even when diagnostics contain errors" },
    .{ .long = "watch", .action = .set_true, .help = "Re-check on file changes (not yet implemented; accepted as a no-op)" },
    .{ .long = "color", .value = .{ .@"enum" = &.{ "auto", "always", "never" } }, .value_name = "WHEN", .help = "Colorize output (default auto: on when stdout is a tty)" },
    .{ .long = "target", .value = .string, .value_name = "TRIPLE", .help = "Compilation target (default: native)" },
};

/// The optional `<CODE>` positional for `toy explain` (e.g. `R0001`); omitted for
/// `toy explain --list` (or the bare word `list`).
const explain_pos: []const Spec.Positional = &.{
    .{ .name = "code", .value = .string, .arity = .optional, .help = "A diagnostic code, e.g. R0001 (omit with --list)" },
};

/// `toy explain` options: `--list` enumerates every code with its severity + title.
const explain_opts: []const Spec.Option = &.{
    .{ .long = "list", .action = .set_true, .help = "List every diagnostic code with its severity and title" },
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
        .long_about = "Compile toy programs. With no subcommand, a bare `toy <file>` builds a signed executable; in a dev build --emit lex|parse|ir inspects the pipeline instead. build and run force a build; check reports diagnostics without building.",
        .options = shared_opts,
        .positionals = files_pos,
        .subcommands = &.{
            .{ .name = "build", .about = "Compile to a signed executable (the default action)", .options = shared_opts, .positionals = files_pos },
            .{ .name = "run", .about = "Build, then execute the binary and report its exit status", .options = shared_opts, .positionals = files_pos },
            .{ .name = "check", .about = "Check syntax and types without building; report all diagnostics", .options = check_opts, .positionals = files_pos },
            .{ .name = "explain", .about = "Print the documentation for a diagnostic code, or --list all codes", .options = explain_opts, .positionals = explain_pos },
        },
    },
};

// A malformed spec is a build error: `Spec.validate` fires `@compileError`. The
// larger option set (the --error/--warn/--ignore flags added here) pushes the
// comptime cross-check over the default 1000-branch budget, so raise the quota at this
// caller-side comptime site (not in the framework).
comptime {
    @setEvalBranchQuota(20_000);
    Spec.validate(spec.root);
}

/// True when `opts` registers a long flag named `long`. Comptime-usable so the
/// invariant below is checked at build time.
fn hasLong(comptime opts: []const Spec.Option, comptime long: []const u8) bool {
    for (opts) |o| {
        if (o.long) |l| if (std.mem.eql(u8, l, long)) return true;
    }
    return false;
}

const std = @import("std");

// LOAD-BEARING INVARIANT: the presence of `--emit` on the build/inspect option set is
// tied EXACTLY to `dev_inspect`. A Debug build (or any test binary) has it; a
// ReleaseFast build must not. `check`'s own option set NEVER carries `--emit` in either
// mode (it is a diagnostics-only subcommand). Fired at comptime so it holds for the main
// exe AND every test binary, in every optimize mode.
comptime {
    if (hasLong(shared_opts, "emit") != dev_inspect)
        @compileError("`--emit` presence on shared_opts must equal build_options.dev_inspect");
    if (hasLong(check_opts, "emit"))
        @compileError("`check` must never register `--emit`");
    // check_opts must carry ONLY the restricted, non-codegen knobs.
    for ([_][]const u8{ "output", "j", "O", "opt", "no-opt", "dump", "codegen-stats", "verify", "no-cache", "opt-stats", "timings", "emit" }) |banned| {
        if (hasLong(check_opts, banned))
            @compileError("`check` option set must not contain build/codegen/opt flags");
    }
}
