//! The `toy` compiler's CLI schema: a comptime `cli.Spec.Cli` the driver
//! (`main.zig`) parses with `cli.Parser`. `build`/`run` are subcommands sharing
//! the root's option set + variadic `<file>` positional; a bare `toy` builds/inspects.
//!
//! Spellings tests/corpora depends on:
//! - `-O` is short-only (Range 0..1) so `-O0`/`-O1`/`-O 1` parse (`-O2` rejected).
//! - `--opt`/`--no-opt` are repeatable enums over the `Opt.Pass` names, one pass each.
//! - `-h`/`--help`/`-V`/`--version` are auto-injected by the framework, so not listed here.

const toyc = @import("toy_compiler");
const Spec = @import("cli/Cli.zig").Spec;
const Decide = toyc.Decide;
// Build-time DEV switch (see build.zig): gates the `--emit lex|parse|ir`
// pipeline-inspection flag on/off. Read at comptime so a release binary (dev_inspect
// false) never even registers `--emit`; the default Debug build (and every test
// binary) has it on, so the existing `--emit lex|parse` dump tests keep working.
// Reached through the library module's `version` (the single `build_options` owner) —
// importing `build_options` here directly would put that generated file in two modules.
pub const dev_inspect: bool = toyc.version.dev_inspect;

// The DEV pipeline-inspection flag, registered on root/build/run ONLY when
// `dev_inspect` is true (Debug builds + every test binary). The value list is
// `{lex,parse,ir}` — NOT `check`: `toy check` is its own subcommand now, so
// `toy --emit check` is rejected as an invalid value. A release build (dev_inspect
// false) drops this entirely, so the binary exposes no `--emit` at all.
const emit_opt: []const Spec.Option = if (dev_inspect) &.{
    .{ .long = "emit", .value = .{ .@"enum" = &.{ "lex", "parse", "ir" } }, .value_name = "STAGE", .help = "Inspect the pipeline instead of building (no binary; dev builds only)" },
} else &.{};

/// Lets a user module import `core/*` (R0013 off), for the raw heap and descriptor
/// self-checks that have no std surface until the testing epic. Dev builds only, so a
/// release binary keeps `core/` private.
const dev_core_opt: []const Spec.Option = if (dev_inspect) &.{
    .{ .long = "dev-core-imports", .action = .set_true, .help = "Let user modules import core/* (dev builds only)" },
} else &.{};

/// `--target`, shared by every command that takes it, so `check` validates it too.
const target_opt: Spec.Option = .{ .long = "target", .value = .{ .@"enum" = &toyc.Driver.targets }, .value_name = "TARGET", .help = "Compilation target (default: " ++ toyc.Driver.default_target ++ ")" };

const pass_names = std.meta.fieldNames(toyc.Opt.Pass);

// The build/inspect options MINUS the dev-only ones (`emit_opt`, `dev_core_opt`, spliced
// in so they can be gated behind `dev_inspect`). `shared_opts` below is `pre_emit ++ emit_opt ++ dev_core_opt ++ post_emit`.
const pre_emit: []const Spec.Option = &.{
    .{ .long = "output", .short = 'o', .value = .string, .value_name = "PATH", .help = "Output path for the built binary (default: .toy/<stamp>/build/<name>)" },
    target_opt,
};
const post_emit: []const Spec.Option = &[_]Spec.Option{
    // -j: N>=1; the parser rejects -j0/negatives as a bad value via the Range.
    .{ .short = 'j', .value = .{ .int = .{ .min = 1, .max = null } }, .value_name = "N", .help = "Build worker threads (N>=1; -j1 = serial; default cpu-based)" },
    // -O is short-only => Parsed field `O`; -O0 / -O1 / -O 1 all parse, -O2 rejected.
    .{ .short = 'O', .value = .{ .int = .{ .min = 0, .max = 1 } }, .value_name = "LEVEL", .help = "IR optimization level (0 = none, 1 = all passes; any pass on also keeps non-escaping closure envs in frames)" },
    .{ .long = "opt", .value = .{ .@"enum" = pass_names }, .action = .append, .value_name = "PASS", .help = "Enable only the named passes (repeatable)" },
    .{ .long = "no-opt", .value = .{ .@"enum" = pass_names }, .action = .append, .value_name = "PASS", .help = "Disable one pass from the current level (repeatable)" },
    .{ .long = "dump", .action = .set_true, .help = "Print the emit phase's artifact (tokens, or the AST)" },
    .{ .long = "codegen-stats", .action = .set_true, .help = "Print compiled-vs-cached function counts (with -o)" },
    .{ .long = "verify", .action = .set_true, .help = "Re-lower cached functions and assert they match (with -o)" },
    .{ .long = "no-cache", .action = .set_true, .help = "Ignore the codegen cache; lower every function (with -o)" },
    .{ .long = "opt-stats", .action = .set_true, .help = "Print per-pass opt counters + dual metric (with -o; use --no-cache)" },
    .{ .long = "no-specialize", .action = .set_true, .help = "Call function-typed arguments of generic functions through the erased instance" },
    .{ .long = "timings", .action = .set_true, .help = "Print the per-stage wall-clock profile" },
    .{ .long = "color", .value = .{ .@"enum" = &.{ "auto", "always", "never" } }, .value_name = "WHEN", .help = "Colorize output (default auto: on when stdout is a tty)" },
} ++ sev_opts;

// Per-code severity overrides, shared by build/run and check. Repeatable; --warn/--allow
// take only a warning code or band, since an error stops the build whatever it is shown
// as. The driver checks the values (Decide.canonicalSpec), as a string flag cannot.
// Fixed precedence allow > warn > deny; last match wins.
const sev_opts: []const Spec.Option = &.{
    .{ .long = "deny", .value = .string, .action = .append, .value_name = "CODE", .help = "Treat a diagnostic code or band (" ++ Decide.sevBands(false) ++ ") as an error (repeatable, e.g. --deny R0001)" },
    .{ .long = "warn", .value = .string, .action = .append, .value_name = "CODE", .help = "Keep a warning code or band (" ++ Decide.sevBands(true) ++ ") a warning, overriding --deny (repeatable; not for error codes)" },
    .{ .long = "allow", .value = .string, .action = .append, .value_name = "CODE", .help = "Suppress a warning code or band (" ++ Decide.sevBands(true) ++ ") from output (repeatable; not for error codes)" },
    .{ .long = "deny-warnings", .action = .set_true, .help = "Treat any surviving warning as an error (non-zero exit)" },
};

/// The option set shared verbatim by root, `build`, and `run` (defined once so
/// they can't drift). `--emit` sits between `pre_emit` and `post_emit` and is present
/// only when `dev_inspect` is true (Debug builds + test binaries); a release build
/// drops it entirely. `--target` is not defaulted in-spec; the driver applies
/// `Driver.default_target` when it is absent.
const shared_opts: []const Spec.Option = pre_emit ++ emit_opt ++ dev_core_opt ++ post_emit;

/// The variadic input-file positional, shared by root/build/run.
const files_pos: []const Spec.Positional = &.{
    .{ .name = "file", .value = .string, .arity = .variadic, .help = "Input source file(s)" },
};

/// `check`'s OWN option set — deliberately NOT the shared build/inspect options. `check`
/// reports diagnostics without building, so nothing codegen/emit/opt related belongs
/// here: only the diagnostic-shaping + input knobs. `--format` picks the wire form
/// (pretty snippets vs stable NDJSON); `--deny/--warn/--allow` are the repeatable
/// severity overrides; `--deny-warnings` promotes any surviving warning to an error
/// for the exit gate; `--exit-zero` forces a 0 exit even with errors (editors that read
/// the stream, not the status); `--watch` is a forward-compatible no-op stub; `--color`,
/// `--target` and `--timings` mirror the shared spellings.
const check_opts: []const Spec.Option = &[_]Spec.Option{
    .{ .long = "format", .value = .{ .@"enum" = &.{ "human", "ndjson" } }, .value_name = "FORM", .help = "Diagnostic output form (human snippets or line-delimited JSON; default human)" },
} ++ sev_opts ++ &[_]Spec.Option{
    .{ .long = "exit-zero", .action = .set_true, .help = "Always exit 0, even when diagnostics contain errors" },
    .{ .long = "watch", .action = .set_true, .help = "Re-check on file changes (not yet implemented; accepted as a no-op)" },
    .{ .long = "color", .value = .{ .@"enum" = &.{ "auto", "always", "never" } }, .value_name = "WHEN", .help = "Colorize output (default auto: on when stdout is a tty)" },
    target_opt,
    .{ .long = "timings", .action = .set_true, .help = "Print the per-stage wall-clock profile" },
} ++ dev_core_opt;

/// The build options `check` does not take, so that using one names it as a build option.
const check_declined: []const Spec.Option = blk: {
    var list: []const Spec.Option = &.{};
    for (shared_opts) |o| {
        const taken = if (o.long) |l| hasLong(check_opts, l) else false;
        if (!taken) list = list ++ &[_]Spec.Option{o};
    }
    break :blk list;
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
            .{ .name = "check", .about = "Check syntax and types without building; report all diagnostics", .options = check_opts, .positionals = files_pos, .declined = .{ .owner = "build", .options = check_declined } },
            .{ .name = "explain", .about = "Print the documentation for a diagnostic code, or --list all codes", .options = explain_opts, .positionals = explain_pos },
            // The language server over stdio (JSON-RPC). No options or positionals: it
            // talks LSP on stdin/stdout, so an empty option set is intentional (the
            // framework accepts `.options = &.{}`).
            .{ .name = "lsp", .about = "Run the language server over stdio (JSON-RPC)" },
        },
    },
};

// A malformed spec is a build error: `Spec.validate` fires `@compileError`. The
// larger option set (the --deny/--warn/--allow flags added here) pushes the
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
    if (hasLong(shared_opts, "dev-core-imports") != dev_inspect or hasLong(check_opts, "dev-core-imports") != dev_inspect)
        @compileError("`--dev-core-imports` presence must equal build_options.dev_inspect");
    if (hasLong(check_opts, "emit"))
        @compileError("`check` must never register `--emit`");
    // check_opts must carry ONLY the restricted, non-codegen knobs.
    for ([_][]const u8{ "output", "j", "O", "opt", "no-opt", "dump", "codegen-stats", "verify", "no-cache", "opt-stats", "no-specialize", "emit" }) |banned| {
        if (hasLong(check_opts, banned))
            @compileError("`check` option set must not contain build/codegen/opt flags");
    }
}
