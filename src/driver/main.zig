//! `toyc` CLI entry point.
//!
//! Usage:
//!   toyc [--emit lex|parse|check] [--dump] [--target <triple>] <file...>
//!
//! Runs the pipeline (lex → parse → check) over each file in parallel, with an
//! on-disk cache, and prints a summary. `--dump` prints the artifact of the emit
//! phase: tokens for `lex`, the AST S-expression for `parse`, and the AST plus a
//! per-function signature summary for `check` (name resolution + typecheck).
//! `-o` and `--emit asm` carry on through codegen → link → sign, all hung off
//! this same driver.

const std = @import("std");
const Io = std.Io;
// The CLI entry is the exe module's root; it lives in driver/ (the engine), but
// because a Zig module cannot import files above its root source file, it reaches
// the rest of the compiler through the `toy_compiler` library module (rooted at
// src/root.zig) rather than via relative `../` paths.
const toyc = @import("toy_compiler");
const Driver = toyc.Driver;
const Ast = toyc.Ast;
const Codegen = toyc.Codegen;
const version = toyc.version;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    // Streaming (not positional) — stdout may be a pipe, tty, or append target,
    // none of which are seekable. Positional writes corrupt ordering there.
    var stdout_writer = Io.File.stdout().writerStreaming(io, &stdout_buf);
    const out = &stdout_writer.interface;

    var dump = false;
    var emit: Driver.Emit = .parse;
    var target: []const u8 = "native";
    var out_path: ?[]const u8 = null;
    var codegen_stats = false;
    var mode: Codegen.Mode = .normal;
    var paths: std.ArrayList([]const u8) = .empty;
    defer paths.deinit(gpa);

    var args = init.minimal.args.iterate();
    _ = args.skip(); // program name
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--dump")) {
            dump = true;
        } else if (std.mem.eql(u8, arg, "--codegen-stats")) {
            codegen_stats = true;
        } else if (std.mem.eql(u8, arg, "--verify")) {
            mode = .verify;
        } else if (std.mem.eql(u8, arg, "--force")) {
            mode = .force;
        } else if (std.mem.eql(u8, arg, "--emit")) {
            const v = args.next() orelse return argError(out, "--emit requires a value (lex|parse|check|asm)");
            if (std.mem.eql(u8, v, "lex")) {
                emit = .lex;
            } else if (std.mem.eql(u8, v, "parse")) {
                emit = .parse;
            } else if (std.mem.eql(u8, v, "check")) {
                emit = .check;
            } else if (std.mem.eql(u8, v, "asm")) {
                emit = .assembly;
            } else {
                return argError(out, "--emit must be 'lex', 'parse', 'check', or 'asm'");
            }
        } else if (std.mem.eql(u8, arg, "-o")) {
            out_path = args.next() orelse return argError(out, "-o requires an output path");
        } else if (std.mem.eql(u8, arg, "--target")) {
            target = args.next() orelse return argError(out, "--target requires a value");
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try usage(out);
            return;
        } else {
            try paths.append(gpa, arg);
        }
    }

    if (paths.items.len == 0) {
        // Asking to emit (-o / --emit asm) with no input is an error, not usage.
        if (out_path != null or emit == .assembly) {
            try argError(out, "no input file");
            std.process.exit(1);
        }
        try usage(out);
        return;
    }

    // Code emission (`-o` or `--emit asm`) is locked to aarch64-macos in M1.
    if ((out_path != null or emit == .assembly) and !isAarch64Macos(target)) {
        try argError(out, "code emission only supports aarch64-macos in M1");
        std.process.exit(1);
    }

    // `-o`: lower `main` and write a signed, runnable executable.
    if (out_path) |path| {
        std.process.exit(try emitExecutable(gpa, io, out, target, paths.items, path, mode, codegen_stats));
    }

    // `--emit asm`: print the generated assembly listing for `main`; no file out.
    if (emit == .assembly) {
        std.process.exit(try emitAsm(gpa, io, out, target, paths.items));
    }

    const results = try Driver.run(gpa, io, emit, target, paths.items);
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }

    const failures = try report(out, results, emit, target, dump);
    try out.flush();
    // Signal compilation failure to scripts/CI. (Flush first; exit skips defers.)
    if (failures > 0) std.process.exit(1);
}

/// True if `target` names the aarch64-macos triple we can emit for (or `native`,
/// which on this host is aarch64-macos). Accepts the common spellings.
fn isAarch64Macos(target: []const u8) bool {
    const ok = [_][]const u8{
        "native",
        "aarch64-macos",
        "arm64-macos",
        "aarch64-apple-macos",
        "aarch64-apple-darwin",
    };
    for (ok) |t| if (std.mem.eql(u8, target, t)) return true;
    return false;
}

/// The basename of a path (after the last '/'), used as the code-signing
/// identifier. Falls back to the whole string when there is no separator.
fn basename(path: []const u8) []const u8 {
    if (std.mem.lastIndexOfScalar(u8, path, '/')) |i| return path[i + 1 ..];
    return path;
}

/// Run lex→parse→check→codegen→link→sign for a single input and write the signed
/// executable to `out_path` (mode 0o755). Returns the process exit code: 0 on
/// success, 1 on any failure (front-end errors, no `main`, unsupported node).
fn emitExecutable(
    gpa: std.mem.Allocator,
    io: Io,
    out: *Io.Writer,
    target: []const u8,
    paths: []const []const u8,
    out_path: []const u8,
    mode: Codegen.Mode,
    codegen_stats: bool,
) !u8 {
    if (paths.len != 1) {
        try argError(out, "-o takes exactly one input file");
        return 1;
    }

    const results = try Driver.run(gpa, io, .check, target, paths);
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }
    const r = &results[0];

    // Front-end errors (read/parse/resolve/type) -> print and fail.
    if (r.err != null) {
        _ = try report(out, results, .check, target, false);
        try out.flush();
        return 1;
    }

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    var lowered = try Driver.lowerProgram(gpa, io, cache, target, r, mode, false);
    switch (lowered) {
        .err => |e| {
            try printEmitError(out, r, e);
            try out.flush();
            return 1;
        },
        .ok => |*lp| {
            defer lp.deinit(gpa);
            if (lp.diags.len > 0) {
                for (lp.diags) |d| {
                    try printEmitError(out, r, .{ .message = d.message, .byte_offset = d.byte_offset });
                }
                try out.flush();
                return 1;
            }

            // How many fns were freshly lowered vs. served from the codegen cache.
            if (codegen_stats) {
                try out.print("codegen: compiled={d} cached={d}\n", .{ lp.codegen_compiled, lp.codegen_cached });
                try out.flush();
            }

            const image = Driver.buildImage(
                gpa,
                basename(out_path),
                lp.text,
                lp.entry_off,
                lp.cstrings,
                lp.data_relocs,
                lp.uses_write,
            ) catch |err| switch (err) {
                error.CodeTooLarge => {
                    try argError(out, "program too large for codegen (code + strings exceed one 16 KB __TEXT page)");
                    return 1;
                },
                else => |e| return e,
            };
            defer gpa.free(image);

            try writeExecutable(io, out_path, image);
            return 0;
        },
    }
}

/// `--emit asm`: lower the whole program and print its assembly listing to
/// stdout (one `_<name>:` block per function; calls render as `bl _<fn>`).
/// Returns the process exit code (0 success, 1 on any failure).
fn emitAsm(
    gpa: std.mem.Allocator,
    io: Io,
    out: *Io.Writer,
    target: []const u8,
    paths: []const []const u8,
) !u8 {
    if (paths.len != 1) {
        try argError(out, "--emit asm takes exactly one input file");
        return 1;
    }

    const results = try Driver.run(gpa, io, .assembly, target, paths);
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }
    const r = &results[0];

    if (r.err != null) {
        _ = try report(out, results, .check, target, false);
        try out.flush();
        return 1;
    }

    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);
    var lowered = try Driver.lowerProgram(gpa, io, cache, target, r, .normal, true);
    switch (lowered) {
        .err => |e| {
            try printEmitError(out, r, e);
            try out.flush();
            return 1;
        },
        .ok => |*lp| {
            defer lp.deinit(gpa);
            if (lp.diags.len > 0) {
                for (lp.diags) |d| {
                    try printEmitError(out, r, .{ .message = d.message, .byte_offset = d.byte_offset });
                }
                try out.flush();
                return 1;
            }
            if (lp.listing) |listing| try out.writeAll(listing);
            try out.flush();
            return 0;
        },
    }
}

/// Write `image` to `path` as an executable (mode 0o755). `createFile`'s
/// `permissions` is subject to the process umask, so we also `setPermissions`
/// explicitly afterwards to guarantee the executable bits land.
fn writeExecutable(io: Io, path: []const u8, image: []const u8) !void {
    const exec_perms: Io.File.Permissions = .fromMode(0o755);
    var file = try Io.Dir.cwd().createFile(io, path, .{ .permissions = exec_perms });
    defer file.close(io);
    try file.writeStreamingAll(io, image);
    try file.setPermissions(io, exec_perms);
}

/// Render a code-emission error as `file:line:col: message` (or `file: message`
/// when there is no source location), matching the front-end diagnostic style.
fn printEmitError(out: *Io.Writer, r: *const Driver.FileResult, e: Driver.EmitError) !void {
    if (e.byte_offset) |off| {
        const loc = lineCol(r.source, off);
        try out.print("{s}:{d}:{d}: error: {s}\n", .{ r.path, loc.line, loc.col, e.message });
    } else {
        try out.print("{s}: error: {s}\n", .{ r.path, e.message });
    }
}

fn argError(out: *Io.Writer, message: []const u8) !void {
    try out.print("error: {s}\n", .{message});
    try out.flush();
}

fn usage(out: *Io.Writer) !void {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    try out.print(
        \\toyc {s} — toy compiler (lexer + parser + name resolution + typecheck)
        \\
        \\usage: toyc [options] <file...>
        \\  --emit lex|parse|check|asm  how far to run the pipeline (default: parse)
        \\  -o <path>         emit a signed, runnable executable (aarch64-macos only)
        \\  --dump            print the emit phase's artifact (tokens, or the AST)
        \\  --target <triple> compilation target (default: native)
        \\  --codegen-stats   print compiled-vs-cached function counts (with -o)
        \\  --verify          re-lower cached functions and assert they match (with -o)
        \\  --force           ignore the codegen cache; lower every function (with -o)
        \\
    , .{version.stamp(&stamp_buf)});
    try out.flush();
}

/// Print the per-file summary; returns the number of files that failed (so the
/// caller can set a non-zero exit status).
fn report(out: *Io.Writer, results: []const Driver.FileResult, emit: Driver.Emit, target: []const u8, dump: bool) !usize {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    try out.print("compiler {s}  target {s}  emit {t}\n", .{ version.stamp(&stamp_buf), target, emit });

    var total_tokens: usize = 0;
    var total_bytes: usize = 0;
    var lex_hits: usize = 0;
    var parse_hits: usize = 0;
    var failures: usize = 0;

    try out.print("{s: <28} {s: >8} {s: >7} {s: >6}  {s}\n", .{ "file", "bytes", "tokens", "nodes", "status" });
    try out.writeAll("-" ** 72 ++ "\n");

    for (results) |r| {
        total_bytes += r.source.len;
        if (r.tokens_cached) lex_hits += 1;
        if (r.nodes_cached) parse_hits += 1;

        if (r.err) |err| {
            failures += 1;
            try printFailure(out, r, err);
            continue;
        }

        // Every token stream ends in a synthetic .eof we don't count as "real".
        const real_tokens = r.tokens.len -| 1;
        total_tokens += real_tokens;

        try out.print("{s: <28} {d: >8} {d: >7} ", .{ r.path, r.source.len, real_tokens });
        if (r.parsed) try out.print("{d: >6}", .{r.nodes.len}) else try out.print("{s: >6}", .{"-"});
        try out.print("  {s}\n", .{cacheNote(r, emit)});

        if (dump) try dumpArtifact(out, r, emit);
    }

    try out.writeAll("-" ** 72 ++ "\n");
    try out.print(
        "{d} file(s): {d} bytes, {d} tokens, cache hits lex={d} parse={d}, {d} failure(s)\n",
        .{ results.len, total_bytes, total_tokens, lex_hits, parse_hits, failures },
    );
    return failures;
}

/// Short per-file status: which phases were fresh vs. served from cache.
fn cacheNote(r: Driver.FileResult, emit: Driver.Emit) []const u8 {
    const lex = if (r.tokens_cached) "cached" else "lexed";
    if (emit == .lex) return lex;
    // Resolution runs in-memory (uncached); report it explicitly when reached.
    if (emit == .check and r.checked) return "checked";
    const parse = if (r.nodes_cached) "cached" else "parsed";
    if (r.tokens_cached and r.nodes_cached) return "cached";
    if (!r.tokens_cached and !r.nodes_cached) return "lexed+parsed";
    // Mixed: name the phase that actually ran.
    return if (r.tokens_cached) parse else lex;
}

fn printFailure(out: *Io.Writer, r: Driver.FileResult, err: anyerror) !void {
    if (r.diag) |d| {
        // Parse error: a single diagnostic.
        const loc = lineCol(r.source, d.byte_offset);
        try out.print("{s: <28} {d: >8} {d: >7} {s: >6}  error: {s} at {d}:{d}\n", .{
            r.path, r.source.len, r.tokens.len -| 1, "-", d.message, loc.line, loc.col,
        });
    } else if (err == error.ResolveError) {
        // Resolution error(s): list each one as file:line:col under the row.
        try out.print("{s: <28} {d: >8} {d: >7} {s: >6}  error: {d} resolve diagnostic(s)\n", .{
            r.path, r.source.len, r.tokens.len -| 1, "-", if (r.resolve) |res| res.diags.len else 0,
        });
        if (r.resolve) |res| for (res.diags) |d| {
            const loc = lineCol(r.source, d.byte_offset);
            try out.print("    {s}:{d}:{d}: {s}\n", .{ r.path, loc.line, loc.col, d.message });
        };
    } else if (err == error.TypeError) {
        // Type error(s): list each one as file:line:col under the row.
        try out.print("{s: <28} {d: >8} {d: >7} {d: >6}  error: {d} type diagnostic(s)\n", .{
            r.path, r.source.len, r.tokens.len -| 1, r.nodes.len, if (r.typecheck) |tc| tc.diags.len else 0,
        });
        if (r.typecheck) |tc| for (tc.diags) |d| {
            const loc = lineCol(r.source, d.byte_offset);
            try out.print("    {s}:{d}:{d}: {s}\n", .{ r.path, loc.line, loc.col, d.message });
        };
    } else {
        try out.print("{s: <28} {s: >8} {s: >7} {s: >6}  error: {t}\n", .{ r.path, "-", "-", "-", err });
    }
}

fn dumpArtifact(out: *Io.Writer, r: Driver.FileResult, emit: Driver.Emit) !void {
    switch (emit) {
        .lex => for (r.tokens, 0..) |tok, i| {
            try out.print("    [{d: >4}] {s: <12} {d: >5}..{d: <5} {s}\n", .{
                i, @tagName(tok.tag), tok.start, tok.end,
                if (tok.tag == .eof) "" else tok.text(r.source),
            });
        },
        .parse => {
            try out.writeAll("    ");
            try Ast.render(out, .{ .nodes = r.nodes, .extra = r.extra }, r.tokens, r.source);
            try out.writeByte('\n');
        },
        // Check dump: the AST plus a per-function signature summary with the
        // inferred return type, then a tally of diagnostics (0 when clean).
        // `assembly` shares the front-end with `check`; its artifact (the asm
        // listing) is printed by `emitAsm`, not the table dumper.
        .check, .assembly => {
            try out.writeAll("    ");
            try Ast.render(out, .{ .nodes = r.nodes, .extra = r.extra }, r.tokens, r.source);
            try out.writeByte('\n');
            try dumpCheck(out, r);
        },
    }
}

/// Per-function summary under `--emit check --dump`: each function's resolved
/// signature with its inferred return type, then a diagnostic count.
fn dumpCheck(out: *Io.Writer, r: Driver.FileResult) !void {
    if (r.nodes.len == 0) return;
    const tree: Ast.Tree = .{ .nodes = r.nodes, .extra = r.extra };
    const prog = r.nodes[Ast.root(r.nodes)];
    if (prog.tag != .program) return;

    const tc = r.typecheck;
    for (Ast.rangeSlice(tree, prog.lhs)) |fn_idx| {
        const decl = r.nodes[fn_idx];
        if (decl.tag != .fn_decl) continue;
        const name = r.tokens[decl.main_token].text(r.source);
        const proto = Ast.protoAt(tree, decl.lhs);

        try out.print("    fn {s}(", .{name});
        for (proto.params, 0..) |p_idx, i| {
            if (i != 0) try out.writeAll(", ");
            const p = r.nodes[p_idx];
            const p_name = r.tokens[p.main_token].text(r.source);
            const p_type = r.tokens[r.nodes[p.lhs].main_token].text(r.source);
            try out.print("{s}: {s}", .{ p_name, p_type });
        }
        const ret = if (proto.ret_type == Ast.none)
            "()"
        else if (r.nodes[proto.ret_type].tag == .literal_unit)
            "()"
        else
            r.tokens[r.nodes[proto.ret_type].main_token].text(r.source);
        // `rhs` is the body block; its inferred type isn't tracked, so report the
        // declared return spelling — the typechecker has already verified it.
        try out.print(") -> {s}\n", .{ret});
    }

    const n_diags = if (tc) |t| t.diags.len else 0;
    try out.print("    checked: {d} diagnostic(s)\n", .{n_diags});
}

const LineCol = struct { line: usize, col: usize };

fn lineCol(source: []const u8, offset: u32) LineCol {
    var line: usize = 1;
    var col: usize = 1;
    const end = @min(offset, source.len);
    for (source[0..end]) |c| {
        if (c == '\n') {
            line += 1;
            col = 1;
        } else col += 1;
    }
    return .{ .line = line, .col = col };
}
