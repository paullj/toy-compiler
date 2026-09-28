//! Manifest-driven corpus harness over `tests/corpora/**` (integration).
//!
//! A single `.zon` manifest (`corpus_manifest.zon`) enumerates every corpus and the
//! MODE it runs in, so adding a corpus is a data edit, not a new harness. Two modes:
//!
//! `.diagnostics` — the exhaustive two-way `#~ ERROR [CODE] <substr>` matcher.
//!   Each fixture is a real toy program carrying `#~ ERROR [CODE] <substr>`
//!   annotations. Because `#` is the toy line-comment character (the lexer skips
//!   `# ...` to end-of-line) the whole annotation is an ordinary comment the
//!   compiler ignores; the `~` marker is what THIS harness keys on. An annotation
//!   applies to the line it sits ON: it expects an error-severity diagnostic on that
//!   line whose `code == [CODE]` and whose message CONTAINS `<substr>`. The harness
//!   runs the in-process front-end (`Driver.pipeline` in `.check` mode), maps every
//!   diagnostic to its 1-based (line, code, message) via `SourceMap`, and matches
//!   EXHAUSTIVELY in BOTH directions (every annotation claims >=1 diagnostic; every
//!   emitted error is claimed by some annotation). A diagnostic that moves line,
//!   changes code, or loses its message trips the harness.
//!
//! `.expect` — the `# expect:` directive harness (ported from the old `check.sh`
//!   pair). A fixture carries `# expect: exit <N>` / `# expect: stdout "<bytes>"` /
//!   `# expect: compile-error "<substr>"`. A compile-error fixture must FAIL the
//!   front-end with a diagnostic containing the substring; every other fixture must
//!   compile, run, and match its exit code and/or stdout. The build+run is the same
//!   in-process path the real `toy build` uses (`Graph.discover` -> `resolveGraph`
//!   -> `checkGraph` -> `lowerGraphProgram` -> `buildImage`), so it follows imports
//!   and multi-file module graphs. Codegen+exec is gated on macOS/aarch64; off that
//!   target a clean front-end counts as covered but does not run the binary.
//!
//! Each corpus asserts a `min_count` floor so a moved/renamed/removed fixture (or a
//! silently-inert walk) fails the suite, and the whole harness asserts it covered
//! SOMETHING. `diff.sh` (byte-identity) and `incremental.sh` (cache-cutoff) test
//! compiler invariants, not corpus expectations, and stay as standalone scripts.
//!
//! Run: `zig build test`, or `zig build test-bin` then
//! `./zig-out/bin/toy-integration-test`.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const toyc = @import("toy_compiler");

const Driver = toyc.Driver;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;
const DriverCodegen = toyc.DriverCodegen;
const Cache = toyc.Cache;
const codes = toyc.diagnostics.codes;
const SourceMap = toyc.term.render.SourceMap;
const Diagnostic = toyc.diagnostics.Diagnostic.Diagnostic;
const Token = toyc.Token;
const Span = toyc.diagnostics.model.Span;

const testing = std.testing;

/// Codegen+exec only runs where the back-end can emit and the kernel can exec what
/// it emits (the signed Mach-O path). Off this target the front-end still runs, so a
/// clean fixture is counted as covered but its binary is not built.
const can_exec = builtin.os.tag == .macos and builtin.cpu.arch == .aarch64;

/// One parsed `#~ ERROR [CODE] <substr>` annotation. `line` is 1-based (the line
/// the comment sits on). `matched` is flipped true when a diagnostic satisfies it.
const Annotation = struct {
    line: usize,
    code: []const u8,
    substr: []const u8,
    matched: bool = false,
};

/// A diagnostic reduced to what the matcher compares against. `matched` flips true
/// when an annotation claims it; an unmatched one at the end is a failure.
const EmittedError = struct {
    line: usize,
    code: codes.Code,
    message: []const u8,
    matched: bool = false,
};

/// Parse the `#~ ERROR [CODE] <substr>` annotation out of one source line, or null
/// if the line carries none. The marker is exactly `#~` (a toy comment `#` plus a
/// `~`); a plain `#` comment, or a mistyped marker (`#-`, `# ~`, `#~~`), yields no
/// annotation so a typo can't silently disable an expectation. After the marker the
/// keyword `ERROR`, then `[CODE]`, then the rest of the line (trimmed) is the
/// substring the message must contain.
fn parseAnnotation(line_text: []const u8, line_1based: usize) ?Annotation {
    const marker = std.mem.indexOf(u8, line_text, "#~") orelse return null;
    var rest = std.mem.trimStart(u8, line_text[marker + 2 ..], " \t");
    // Require the exact `ERROR ` keyword (a `~` immediately followed by more `~`,
    // e.g. `#~~`, leaves `rest` not starting with ERROR -> null, rejecting it).
    const kw = "ERROR";
    if (!std.mem.startsWith(u8, rest, kw)) return null;
    rest = std.mem.trimStart(u8, rest[kw.len..], " \t");
    // `[CODE]`.
    if (rest.len == 0 or rest[0] != '[') return null;
    const close = std.mem.indexOfScalar(u8, rest, ']') orelse return null;
    const code = rest[1..close];
    const substr = std.mem.trim(u8, rest[close + 1 ..], " \t\r");
    return .{ .line = line_1based, .code = code, .substr = substr };
}

/// Collect every annotation in `source` (one pass, tracking line numbers by
/// counting '\n'). Caller owns the returned list.
fn collectAnnotations(gpa: std.mem.Allocator, source: []const u8) !std.ArrayList(Annotation) {
    var list: std.ArrayList(Annotation) = .empty;
    errdefer list.deinit(gpa);
    var line_no: usize = 1;
    var it = std.mem.splitScalar(u8, source, '\n');
    while (it.next()) |line| {
        if (parseAnnotation(line, line_no)) |a| try list.append(gpa, a);
        line_no += 1;
    }
    return list;
}

/// Run the in-process front-end over `path` and reduce every ERROR-severity
/// diagnostic to an `EmittedError` (1-based line via `SourceMap`). Parse-tainted
/// files stop at parse (their `diags`); a clean parse yields resolve then (if
/// resolve was clean) typecheck diagnostics — we gather whichever are present.
/// Caller owns the returned list.
fn collectErrors(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    path: []const u8,
) !std.ArrayList(EmittedError) {
    var r: Driver.FileResult = .{ .path = path };
    // A per-file front-end error (e.g. a hard IO failure) would surface as `r.err`;
    // a diagnostic-bearing file returns normally with the diags populated. We only
    // care about the collected diagnostics, so ignore the `error.ParseError`/
    // `.ResolveError`/`.TypeError` sentinels the pipeline stores in `r.err`.
    Driver.pipeline(gpa, io, cache, .check, "native", &r, 0) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {}, // a compile diagnostic, not a harness failure
    };
    defer r.deinit(gpa);

    var sm = try SourceMap.init(gpa, path, r.source);
    defer sm.deinit(gpa);

    var out: std.ArrayList(EmittedError) = .empty;
    errdefer out.deinit(gpa);

    const push = struct {
        fn go(list: *std.ArrayList(EmittedError), a: std.mem.Allocator, s: *const SourceMap, diags: []const Diagnostic, tokens: []const Token, bad_spans: *usize) !void {
            for (diags) |d| {
                if (!checkSpans(s, tokens, d)) bad_spans.* += 1;
                // Only ERROR-severity diagnostics are matched against `#~ ERROR`
                // annotations (the stored registry-default severity; the render-time
                // config never rewrites it).
                if (d.severity != .err) continue;
                // Duplicate the borrowed message: `r` is freed on return, but the
                // matcher (and any failure message) reads it afterwards.
                const msg = try a.dupe(u8, d.message);
                try list.append(a, .{
                    .line = s.lineIndex(d.byte_offset) + 1, // 0-based -> 1-based
                    .code = d.code,
                    .message = msg,
                });
            }
        }
    }.go;

    var bad_spans: usize = 0;
    if (r.diags.len > 0) try push(&out, gpa, &sm, r.diags, r.tokens, &bad_spans);
    if (r.resolve) |res| try push(&out, gpa, &sm, res.diags, r.tokens, &bad_spans);
    if (r.typecheck) |tc| try push(&out, gpa, &sm, tc.diags, r.tokens, &bad_spans);
    if (bad_spans > 0) return error.SpanInvariant;

    return out;
}

/// Every diagnostic's spans must be real source extents: non-empty (short of EOF), starting
/// at a token's start and ending at a token's end, with balanced brackets inside. An
/// emit site that forgets its span, or gives a subtree span that cuts a bracket pair,
/// fails here rather than as a stray underline in an editor.
fn checkSpans(sm: *const SourceMap, tokens: []const Token, d: Diagnostic) bool {
    var ok = checkSpan(sm, tokens, d, "primary", d.span());
    if (d.relatedSpan()) |rs| ok = checkSpan(sm, tokens, d, "related", rs) and ok;
    return ok;
}

fn checkSpan(sm: *const SourceMap, tokens: []const Token, d: Diagnostic, which: []const u8, sp: Span) bool {
    const why: ?[]const u8 = blk: {
        if (sp.start >= sm.bytes.len) break :blk null; // an EOF error has nothing to cover
        if (sp.end <= sp.start) break :blk "is empty";
        var first: ?usize = null;
        var last: ?usize = null;
        for (tokens, 0..) |t, i| {
            if (t.start == sp.start) first = i;
            if (t.end == sp.end) last = i;
        }
        if (first == null) break :blk "does not start at a token";
        if (last == null or last.? < first.?) break :blk "does not end at a token";
        // One token is always a whole thing, even a stray `)` a parse error points at.
        if (first.? == last.?) break :blk null;
        var depth: i32 = 0;
        for (tokens[first.? .. last.? + 1]) |t| {
            switch (t.tag) {
                .l_paren, .l_bracket, .l_brace => depth += 1,
                .r_paren, .r_bracket, .r_brace => depth -= 1,
                else => {},
            }
            if (depth < 0) break :blk "closes a bracket it does not open";
        }
        if (depth != 0) break :blk "leaves a bracket open";
        break :blk null;
    };
    if (why) |w| {
        const lc = sm.lineCol(sp.start);
        std.debug.print("{s}:{d}:{d}: {s} span of \"{s}\" {s}: \"{s}\"\n", .{
            sm.name, lc.line, lc.col, which, d.message, w, sm.bytes[sp.start..@min(sp.end, sm.bytes.len)],
        });
        return false;
    }
    return true;
}

/// `checkSpans` over a multi-module graph's diagnostics, each against its own module.
fn checkGraphSpans(gpa: std.mem.Allocator, graph: *const Graph.Graph, diags: []const Diagnostic) !void {
    var bad: usize = 0;
    for (diags) |d| {
        const m = &graph.modules[if (d.scope == toyc.DiagnosticSink.NO_SCOPE) graph.entry_index else d.scope];
        var sm = try SourceMap.init(gpa, m.file, m.source);
        defer sm.deinit(gpa);
        if (!checkSpans(&sm, m.tokens, d)) bad += 1;
    }
    if (bad > 0) return error.SpanInvariant;
}

fn freeErrors(gpa: std.mem.Allocator, errs: *std.ArrayList(EmittedError)) void {
    for (errs.items) |e| gpa.free(@constCast(e.message));
    errs.deinit(gpa);
}

/// The core exhaustive matcher, factored out so a NEGATIVE unit test can drive it
/// directly with a mismatched annotation set and assert rejection. Mutates the
/// `matched` flags in place; returns the first failure message (allocated with
/// `gpa`) or null on a full two-way match.
fn matchExhaustive(
    gpa: std.mem.Allocator,
    file: []const u8,
    annotations: []Annotation,
    errors: []EmittedError,
) !?[]const u8 {
    // Forward: every annotation must match >=1 emitted error on its line.
    for (annotations) |*a| {
        for (errors) |*e| {
            if (e.line != a.line) continue;
            const code_str = codes.str(e.code) orelse continue; // uncoded can't match a `[CODE]`
            if (!std.mem.eql(u8, code_str, a.code)) continue;
            if (std.mem.indexOf(u8, e.message, a.substr) == null) continue;
            a.matched = true;
            e.matched = true;
        }
        if (!a.matched) {
            return try std.fmt.allocPrint(
                gpa,
                "{s}:{d}: annotation [{s}] \"{s}\" matched no error diagnostic on that line",
                .{ file, a.line, a.code, a.substr },
            );
        }
    }
    // Reverse: every emitted error must have been claimed by some annotation.
    for (errors) |e| {
        if (e.matched) continue;
        const code_str = codes.str(e.code) orelse "(none)";
        return try std.fmt.allocPrint(
            gpa,
            "{s}:{d}: emitted error [{s}] \"{s}\" has no matching #~ ERROR annotation",
            .{ file, e.line, code_str, e.message },
        );
    }
    return null;
}

/// Check one fixture end-to-end: collect its annotations + emitted errors and run
/// the exhaustive two-way match. Returns the failure message (or null on success);
/// caller frees a non-null message.
fn checkFixture(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    path: []const u8,
) !?[]const u8 {
    const source = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(source);

    var anns = try collectAnnotations(gpa, source);
    defer anns.deinit(gpa);
    // A UI fixture with no annotations is almost certainly a mistake (a silent
    // no-op); require at least one.
    if (anns.items.len == 0) {
        return try std.fmt.allocPrint(gpa, "{s}: fixture has no #~ ERROR annotations", .{path});
    }

    var errs = try collectErrors(gpa, io, cache, path);
    defer freeErrors(gpa, &errs);

    return matchExhaustive(gpa, path, anns.items, errs.items);
}

// ---------------------------------------------------------------------------
// `.expect` mode: the `# expect:` directive harness (ported from check.sh).
// ---------------------------------------------------------------------------

const Mode = enum { diagnostics, expect };
/// How a corpus enumerates its fixtures. `single_file` walks `*.toy`; `module_dir`
/// treats each directory containing `main.toy` as one multi-file program whose
/// non-entry files are discovered from that entry.
const Unit = enum { single_file, module_dir };

const Corpus = struct {
    path: []const u8,
    mode: Mode,
    unit: Unit,
    /// The floor of fixtures this corpus must contribute. A moved/renamed/removed
    /// fixture drops the count below the floor and fails the suite; `0` tolerates a
    /// declared-but-empty corpus (the regression placeholder) without going inert.
    min_count: usize,
    /// First-path-component names to skip during a `single_file` walk (e.g. the
    /// `modules` subtree lives under `language-features` but is its own corpus).
    prune: []const []const u8 = &.{},
};

const manifest: []const Corpus = @import("corpus_manifest.zon");

/// The parsed `# expect:` directives of one fixture. `any` gates the "a fixture with
/// no directive is a mistake" check (matching the old bash). `stdout`/`cerr` are owned
/// when non-null/non-empty (see `deinit`).
const Expect = struct {
    any: bool = false,
    exit: ?u8 = null,
    stdout: ?[]const u8 = null,
    cerr_expected: bool = false,
    cerr: []const u8 = "",

    fn deinit(self: *Expect, gpa: std.mem.Allocator) void {
        if (self.stdout) |s| gpa.free(s);
        if (self.cerr.len != 0) gpa.free(self.cerr);
    }
};

/// Decode the escape set the fixtures use in quoted directive bodies (`\n \t \" \\`);
/// any other byte (including raw UTF-8) passes through verbatim. Caller owns the result.
fn decodeEscapes(gpa: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < raw.len) : (i += 1) {
        if (raw[i] == '\\' and i + 1 < raw.len) {
            const next = raw[i + 1];
            const decoded: ?u8 = switch (next) {
                'n' => '\n',
                't' => '\t',
                '"' => '"',
                '\\' => '\\',
                else => null,
            };
            if (decoded) |d| {
                try out.append(gpa, d);
                i += 1;
                continue;
            }
        }
        try out.append(gpa, raw[i]);
    }
    return out.toOwnedSlice(gpa);
}

/// Slice from the first `"` to the last `"` (exclusive of the quotes), mirroring the
/// old bash greedy `.*` between quotes. Null if there is no quoted body.
fn betweenQuotes(s: []const u8) ?[]const u8 {
    const first = std.mem.indexOfScalar(u8, s, '"') orelse return null;
    const last = std.mem.lastIndexOfScalar(u8, s, '"') orelse return null;
    if (last <= first) return null;
    return s[first + 1 .. last];
}

/// Parse every `# expect:` directive out of `source`. Recognized directives:
/// `exit <N>`, `stdout "<bytes>"`, `compile-error "<substr>"`. Caller `deinit`s.
fn parseExpect(gpa: std.mem.Allocator, source: []const u8) !Expect {
    var e: Expect = .{};
    errdefer e.deinit(gpa);
    var it = std.mem.splitScalar(u8, source, '\n');
    while (it.next()) |line| {
        const prefix = "# expect:";
        if (!std.mem.startsWith(u8, line, prefix)) continue;
        const body = std.mem.trim(u8, line[prefix.len..], " \t\r");
        if (std.mem.startsWith(u8, body, "exit ")) {
            const n = std.mem.trim(u8, body["exit ".len..], " \t\r");
            e.exit = std.fmt.parseInt(u8, n, 10) catch continue;
            e.any = true;
        } else if (std.mem.startsWith(u8, body, "stdout")) {
            if (betweenQuotes(body)) |raw| {
                if (e.stdout) |old| gpa.free(old);
                e.stdout = try decodeEscapes(gpa, raw);
                e.any = true;
            }
        } else if (std.mem.startsWith(u8, body, "compile-error")) {
            e.cerr_expected = true;
            e.any = true;
            const raw = body["compile-error".len..];
            const inner = betweenQuotes(raw) orelse std.mem.trim(u8, raw, " \t\r");
            if (inner.len != 0) {
                if (e.cerr.len != 0) gpa.free(e.cerr);
                e.cerr = try gpa.dupe(u8, inner);
            }
        }
    }
    return e;
}

/// The outcome of building (and, on target, running) one `.expect` fixture.
const RunResult = union(enum) {
    /// The front-end rejected the program; the payload is the rendered diagnostics
    /// (owned) the `compile-error` substring is matched against.
    compile_error: []const u8,
    /// Front-end was clean but this host cannot exec the emitted binary; counted as
    /// covered without an exit/stdout check.
    skipped_exec,
    /// The program compiled and ran; `stdout` is owned.
    ran: struct { term: std.process.Child.Term, stdout: []const u8 },

    fn deinit(self: *RunResult, gpa: std.mem.Allocator) void {
        switch (self.*) {
            .compile_error => |s| gpa.free(s),
            .ran => |r| gpa.free(r.stdout),
            .skipped_exec => {},
        }
    }
};

fn countErr(diags: []const Diagnostic) usize {
    var n: usize = 0;
    for (diags) |d| {
        if (d.severity == .err) n += 1;
    }
    return n;
}

/// Render every error-severity diagnostic as `[code]: message` (omitting `[…]` for an
/// uncoded diagnostic), one per line. Reconstructs exactly what the `compile-error`
/// substrings are written against: a `[Pxxxx]:` prefix for coded parse/type errors and
/// the bare message for structural ones. Caller owns the result.
fn renderDiags(gpa: std.mem.Allocator, diags: []const Diagnostic) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (diags) |d| {
        if (d.severity != .err) continue;
        const line = if (codes.str(d.code)) |c|
            try std.fmt.allocPrint(gpa, "[{s}]: {s}\n", .{ c, d.message })
        else
            try std.fmt.allocPrint(gpa, "{s}\n", .{d.message});
        defer gpa.free(line);
        try out.appendSlice(gpa, line);
    }
    return out.toOwnedSlice(gpa);
}

/// Render a structural discovery error. A parse failure carries the module's full
/// coded parse-diagnostic list, so route it through `renderDiags` to keep the
/// `[Pxxxx]:` prefix the substrings expect; other structural errors (cycle, missing
/// import, …) only carry `message`.
fn renderGraphErr(gpa: std.mem.Allocator, err: Graph.Error) ![]u8 {
    if (err.parse_diags.len != 0) return renderDiags(gpa, err.parse_diags);
    return std.fmt.allocPrint(gpa, "{s}\n", .{err.message});
}

/// Build `entry` through the real whole-program path (discover -> resolve ->
/// typecheck -> codegen -> link) and, on an exec-capable host, run it. `entry` must be
/// a cwd-RELATIVE path: the whole-program front-end reads every module through
/// `Io.Dir.cwd()`, which rejects an absolute path. Returns the first failing stage as
/// `.compile_error`, or the run outcome. `out_name` must be a cwd-relative path unique
/// to this fixture (duplicate basenames otherwise collide on the emitted binary).
/// Caller `deinit`s the result.
fn compileAndRun(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    entry: []const u8,
    out_name: []const u8,
) !RunResult {
    var graph = try Graph.discover(gpa, io, cache, "aarch64-macos", entry, null, null);
    defer graph.deinit(gpa);
    if (graph.err) |err| {
        try checkGraphSpans(gpa, &graph, err.parse_diags);
        return .{ .compile_error = try renderGraphErr(gpa, err) };
    }

    var res = try ResolveGraph.resolveGraph(gpa, &graph);
    defer res.deinit(gpa);
    // Typechecking a resolve-broken graph risks spurious diagnostics or a crash, so
    // report the resolve failure and stop before `checkGraph`.
    try checkGraphSpans(gpa, &graph, res.diags);
    if (countErr(res.diags) > 0) return .{ .compile_error = try renderDiags(gpa, res.diags) };

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, io, 0);
    defer tc.deinit(gpa);
    try checkGraphSpans(gpa, &graph, tc.diags);
    if (countErr(tc.diags) > 0) return .{ .compile_error = try renderDiags(gpa, tc.diags) };

    if (!can_exec) return .skipped_exec;

    var lowered = try DriverCodegen.lowerGraphProgram(gpa, io, cache, "aarch64-macos", &graph, &res, &tc, .normal, .O0, null, null, 0);
    const lp = switch (lowered) {
        .ok => |*ok| ok,
        // A codegen `EmitError` after a clean typecheck is a real back-end failure,
        // not a fixture expectation — surface it as a diagnostic string so `judge`
        // reports "compile failed" against it rather than swallowing it.
        .err => |em| return .{ .compile_error = try std.fmt.allocPrint(gpa, "codegen error: {s}\n", .{em.message}) },
    };
    defer lp.deinit(gpa);
    if (lp.diags.len > 0) return .{ .compile_error = try std.fmt.allocPrint(gpa, "codegen diagnostics ({d})\n", .{lp.diags.len}) };

    const image = try DriverCodegen.buildImage(io, gpa, out_name, lp.text, lp.entry_off, lp.cstrings, lp.data_relocs);
    defer gpa.free(image);
    {
        const perms: Io.File.Permissions = .fromMode(0o755);
        var f = try Io.Dir.cwd().createFile(io, out_name, .{ .permissions = perms });
        defer f.close(io);
        try f.writeStreamingAll(io, image);
        try f.setPermissions(io, perms);
    }
    const abs = try Io.Dir.cwd().realPathFileAlloc(io, out_name, gpa);
    defer gpa.free(abs);
    {
        var cs = try std.process.spawn(io, .{ .argv = &.{ "codesign", "-v", abs } });
        if ((exitCode(try cs.wait(io)) orelse 1) != 0) {
            return .{ .compile_error = try std.fmt.allocPrint(gpa, "codesign rejected {s}\n", .{out_name}) };
        }
    }
    // Capture stderr too (and discard it): a fixture that panics writes a backtrace to
    // fd 2, and under `zig build test` the test binary's inherited stderr is part of the
    // build runner's `--listen=-` results stream — leaking a child backtrace onto it
    // corrupts that protocol. Piping + draining stderr keeps it off the inherited fd.
    var child = try std.process.spawn(io, .{ .argv = &.{abs}, .stdout = .pipe, .stderr = .pipe });
    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(got);
    var errr = child.stderr.?.readerStreaming(io, &.{});
    const err_bytes = try errr.interface.allocRemaining(gpa, .limited(1 << 16));
    gpa.free(err_bytes);
    const term = try child.wait(io);
    return .{ .ran = .{ .term = term, .stdout = got } };
}

/// The exit code of a finished child, or `null` for a signal death (a bounds/panic
/// trap): reading `.exited` on a `.signal` term is checked-illegal, so funnel every
/// term through this switch and treat a signal as a mismatch.
fn exitCode(term: std.process.Child.Term) ?u8 {
    return switch (term) {
        .exited => |c| c,
        else => null,
    };
}

/// Compare a fixture's `# expect:` directives against its build/run outcome, in the
/// same branch order the old bash used. Returns the failure message (owned) or null on
/// a pass. Pure over its inputs, so the negative-control tests drive it directly.
fn judge(gpa: std.mem.Allocator, name: []const u8, e: Expect, r: RunResult) !?[]const u8 {
    if (!e.any) return try std.fmt.allocPrint(gpa, "{s}: no # expect: directive", .{name});

    if (e.cerr_expected) {
        switch (r) {
            .ran => return try std.fmt.allocPrint(gpa, "{s}: expected compile-error but it compiled", .{name}),
            .compile_error => |text| {
                if (e.cerr.len != 0 and std.mem.indexOf(u8, text, e.cerr) == null) {
                    return try std.fmt.allocPrint(gpa, "{s}: diagnostic missing \"{s}\" (got: {s})", .{ name, e.cerr, text });
                }
                return null;
            },
            .skipped_exec => return null,
        }
    }

    switch (r) {
        .compile_error => |text| return try std.fmt.allocPrint(gpa, "{s}: compile failed: {s}", .{ name, text }),
        .skipped_exec => return null,
        .ran => |run| {
            if (e.exit) |want| {
                if (exitCode(run.term) != want) {
                    if (exitCode(run.term)) |got| {
                        return try std.fmt.allocPrint(gpa, "{s}: exit {d}, want {d}", .{ name, got, want });
                    }
                    return try std.fmt.allocPrint(gpa, "{s}: died by signal, want exit {d}", .{ name, want });
                }
            }
            if (e.stdout) |want| {
                if (!std.mem.eql(u8, std.mem.trimEnd(u8, run.stdout, "\n"), std.mem.trimEnd(u8, want, "\n"))) {
                    return try std.fmt.allocPrint(gpa, "{s}: stdout mismatch (got \"{s}\", want \"{s}\")", .{ name, run.stdout, want });
                }
            }
            return null;
        },
    }
}

/// True if `rel_path`'s first path component is one of `prune`.
fn isPruned(rel_path: []const u8, prune: []const []const u8) bool {
    for (prune) |name| {
        if (std.mem.startsWith(u8, rel_path, name) and
            rel_path.len > name.len and rel_path[name.len] == '/') return true;
    }
    return false;
}

/// Run one `.expect` fixture end-to-end and return its failure message (owned) or null.
/// `entry` is the cwd-relative entry path — used both to read the directives and, as the
/// display name, in any failure message.
fn checkExpectFixture(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    entry: []const u8,
    out_name: []const u8,
) !?[]const u8 {
    const source = try Io.Dir.cwd().readFileAlloc(io, entry, gpa, .unlimited);
    defer gpa.free(source);

    var e = try parseExpect(gpa, source);
    defer e.deinit(gpa);

    var r = try compileAndRun(gpa, io, cache, entry, out_name);
    defer r.deinit(gpa);

    return judge(gpa, entry, e, r);
}

test "corpus: every manifest corpus matches its expectations" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    // `openCache` writes the cache dir into `dir_buf` and returns a `Cache` that
    // BORROWS it, so the buffer must outlive every `cache` use — keep it on THIS
    // frame (a helper that owns the buffer would return a dangling `.dir`).
    var dir_buf: [Driver.cache_dir_buf_len]u8 = undefined;
    const cache = try Driver.openCache(io, &dir_buf);

    // Emitted binaries land here (relative to cwd), so duplicate fixture basenames
    // across the corpus can't clobber each other.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var out_dir_buf: [64]u8 = undefined;
    const out_dir = std.fmt.bufPrint(&out_dir_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;

    var global_total: usize = 0;
    var bin_index: usize = 0;
    for (manifest) |c| {
        var count: usize = 0;
        var root = try Io.Dir.cwd().openDir(io, c.path, .{ .iterate = true });
        defer root.close(io);
        var walker = try root.walk(gpa);
        defer walker.deinit();

        while (try walker.next(io)) |entry| {
            if (entry.kind != .file) continue;
            switch (c.unit) {
                .single_file => {
                    if (!std.mem.endsWith(u8, entry.path, ".toy")) continue;
                    if (isPruned(entry.path, c.prune)) continue;
                },
                .module_dir => {
                    if (!std.mem.eql(u8, std.fs.path.basename(entry.path), "main.toy")) continue;
                },
            }

            // The in-process front-end reads every fixture through
            // `Io.Dir.cwd().readFileAlloc`, which rejects an absolute path
            // (`error.BadPathName`); keep the path cwd-RELATIVE (as the old harnesses
            // did) so both the diagnostics `pipeline` and the `.expect` discover read
            // it, and so module imports resolve against the entry's relative dir.
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const rel = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ c.path, entry.path });

            count += 1;
            const failure = switch (c.mode) {
                .diagnostics => try checkFixture(gpa, io, cache, rel),
                .expect => blk: {
                    const out_name = try std.fmt.allocPrint(gpa, "{s}/f{d}", .{ out_dir, bin_index });
                    defer gpa.free(out_name);
                    bin_index += 1;
                    break :blk try checkExpectFixture(gpa, io, cache, rel, out_name);
                },
            };
            if (failure) |msg| {
                defer gpa.free(msg);
                std.debug.print("corpus mismatch [{s}]: {s}\n", .{ c.path, msg });
                return error.CorpusMismatch;
            }
        }

        // A corpus that should be non-empty must contribute its floor: a fixture that
        // was moved, renamed, or deleted drops the count and fails here.
        if (count < c.min_count) {
            std.debug.print("corpus [{s}]: found {d} fixtures, expected >= {d}\n", .{ c.path, count, c.min_count });
            return error.CorpusUndercount;
        }
        global_total += count;
    }
    // No manifest edit can leave the whole harness silently inert.
    try testing.expect(global_total > 0);
}

test "ui harness: parseAnnotation accepts the exact marker and rejects near-misses" {
    // Accepted: the canonical form.
    {
        const a = parseAnnotation("  aaa  #~ ERROR [R0001] undeclared identifier 'aaa'", 2).?;
        try testing.expectEqual(@as(usize, 2), a.line);
        try testing.expectEqualStrings("R0001", a.code);
        try testing.expectEqualStrings("undeclared identifier 'aaa'", a.substr);
    }
    // Rejected: a plain comment (no `~`).
    try testing.expect(parseAnnotation("  # ERROR [R0001] x", 1) == null);
    // Rejected: a mistyped marker.
    try testing.expect(parseAnnotation("  #- ERROR [R0001] x", 1) == null);
    // Rejected: `#~` without the ERROR keyword.
    try testing.expect(parseAnnotation("  #~ [R0001] x", 1) == null);
    // Rejected: no `[CODE]` bracket.
    try testing.expect(parseAnnotation("  #~ ERROR R0001 x", 1) == null);
    // A line with no annotation at all.
    try testing.expect(parseAnnotation("  return 0", 1) == null);
}

test "ui harness: the matcher REJECTS a wrong code, a wrong substring, and an unannotated error" {
    // This proves the two-way match is genuinely exhaustive: a deliberately
    // mismatched annotation/diagnostic set must fail.
    const gpa = testing.allocator;

    // 1) Wrong code: annotation says T0004 but the error is R0001 -> forward-match
    //    fails (unmatched annotation).
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "T0004", .substr = "undeclared" }};
        var errs = [_]EmittedError{.{ .line = 2, .code = .R0001, .message = "undeclared identifier 'x'" }};
        const failure = (try matchExhaustive(gpa, "t.toy", &anns, &errs)).?;
        defer gpa.free(failure);
        try testing.expect(std.mem.indexOf(u8, failure, "matched no error diagnostic") != null);
    }
    // 2) Wrong substring: code matches, message does not contain the substring.
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "R0001", .substr = "TOTALLY WRONG" }};
        var errs = [_]EmittedError{.{ .line = 2, .code = .R0001, .message = "undeclared identifier 'x'" }};
        const failure = (try matchExhaustive(gpa, "t.toy", &anns, &errs)).?;
        defer gpa.free(failure);
        try testing.expect(std.mem.indexOf(u8, failure, "matched no error diagnostic") != null);
    }
    // 3) Unannotated error: an emitted error with NO covering annotation -> reverse
    //    match fails (unmatched diagnostic).
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "R0001", .substr = "aaa" }};
        var errs = [_]EmittedError{
            .{ .line = 2, .code = .R0001, .message = "undeclared identifier 'aaa'" },
            .{ .line = 3, .code = .R0001, .message = "undeclared identifier 'bbb'" }, // no annotation
        };
        const failure = (try matchExhaustive(gpa, "t.toy", &anns, &errs)).?;
        defer gpa.free(failure);
        try testing.expect(std.mem.indexOf(u8, failure, "no matching #~ ERROR annotation") != null);
    }
    // 4) A correct pairing yields NO failure (the positive control for the matcher).
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "R0001", .substr = "aaa" }};
        var errs = [_]EmittedError{.{ .line = 2, .code = .R0001, .message = "undeclared identifier 'aaa'" }};
        try testing.expect((try matchExhaustive(gpa, "t.toy", &anns, &errs)) == null);
    }
}

fn expectJudgeFailure(gpa: std.mem.Allocator, e: Expect, r: RunResult, needle: []const u8) !void {
    const msg = (try judge(gpa, "t.toy", e, r)).?;
    defer gpa.free(msg);
    try testing.expect(std.mem.indexOf(u8, msg, needle) != null);
}

test "expect harness: judge REJECTS a wrong exit, a stdout mismatch, and a mis-expected compile-error" {
    // The in-suite proof that `.expect` mode is not inert: a deliberately wrong
    // outcome must produce a failure message, keyed exactly like the old bash.
    const gpa = testing.allocator;

    // Wrong exit: expected 0, program exited 1.
    try expectJudgeFailure(
        gpa,
        .{ .any = true, .exit = 0 },
        .{ .ran = .{ .term = .{ .exited = 1 }, .stdout = "" } },
        "exit",
    );
    // stdout mismatch: expected "x", got "y".
    try expectJudgeFailure(
        gpa,
        .{ .any = true, .stdout = "x" },
        .{ .ran = .{ .term = .{ .exited = 0 }, .stdout = "y" } },
        "stdout mismatch",
    );
    // A compile-error fixture that compiled and ran instead.
    try expectJudgeFailure(
        gpa,
        .{ .any = true, .cerr_expected = true },
        .{ .ran = .{ .term = .{ .exited = 0 }, .stdout = "" } },
        "expected compile-error but it compiled",
    );
    // A compile-error fixture whose diagnostic lacks the expected substring.
    try expectJudgeFailure(
        gpa,
        .{ .any = true, .cerr_expected = true, .cerr = "ZZZ" },
        .{ .compile_error = "[R0001]: something else\n" },
        "diagnostic missing",
    );
    // A run fixture that failed to compile.
    try expectJudgeFailure(
        gpa,
        .{ .any = true, .exit = 0 },
        .{ .compile_error = "[R0001]: undeclared\n" },
        "compile failed",
    );
    // A fixture with no directive at all is a mistake.
    try expectJudgeFailure(
        gpa,
        .{ .any = false },
        .{ .ran = .{ .term = .{ .exited = 0 }, .stdout = "" } },
        "no # expect: directive",
    );
    // A signal death when an exit code was wanted is a mismatch, not a crash (the
    // `.exited`-on-`.signal` landmine): assert it is REPORTED, not panicked on.
    try expectJudgeFailure(
        gpa,
        .{ .any = true, .exit = 0 },
        .{ .ran = .{ .term = .{ .signal = @enumFromInt(11) }, .stdout = "" } },
        "signal",
    );
}

test "expect harness: judge PASSES exact exit, trailing-newline-insensitive stdout, and a matched compile-error" {
    const gpa = testing.allocator;

    // Exact exit + exact stdout.
    try testing.expect((try judge(gpa, "t.toy", .{ .any = true, .exit = 7, .stdout = "hi\n" }, .{ .ran = .{ .term = .{ .exited = 7 }, .stdout = "hi\n" } })) == null);
    // Trailing-newline parity: "x\n" vs "x" compare equal (bash `$()` strips them).
    try testing.expect((try judge(gpa, "t.toy", .{ .any = true, .stdout = "x" }, .{ .ran = .{ .term = .{ .exited = 0 }, .stdout = "x\n" } })) == null);
    // compile-error substring present in the rendered diagnostic.
    try testing.expect((try judge(gpa, "t.toy", .{ .any = true, .cerr_expected = true, .cerr = "undeclared" }, .{ .compile_error = "[R0001]: undeclared identifier 'x'\n" })) == null);
    // A clean front-end off-target counts as covered.
    try testing.expect((try judge(gpa, "t.toy", .{ .any = true, .exit = 0 }, .skipped_exec)) == null);
}

test "expect harness: parseExpect decodes escapes, passes raw UTF-8, and strips quotes" {
    const gpa = testing.allocator;

    {
        var e = try parseExpect(gpa, "# expect: exit 42\n# expect: stdout \"a\\tb\\n\"\n");
        defer e.deinit(gpa);
        try testing.expect(e.any);
        try testing.expectEqual(@as(?u8, 42), e.exit);
        try testing.expectEqualStrings("a\tb\n", e.stdout.?);
    }
    // Raw UTF-8 in a stdout body passes through verbatim; only \n \t \" \\ are escapes.
    {
        var e = try parseExpect(gpa, "# expect: stdout \"héllo \\q\"\n");
        defer e.deinit(gpa);
        try testing.expectEqualStrings("héllo \\q", e.stdout.?);
    }
    // compile-error with a quoted body strips the surrounding quotes.
    {
        var e = try parseExpect(gpa, "# expect: compile-error \"undeclared identifier 'y'\"\n");
        defer e.deinit(gpa);
        try testing.expect(e.cerr_expected);
        try testing.expectEqualStrings("undeclared identifier 'y'", e.cerr);
    }
    // A fixture with no directive sets nothing.
    {
        var e = try parseExpect(gpa, "fn main() -> int {\n return 0\n}\n");
        defer e.deinit(gpa);
        try testing.expect(!e.any);
    }
}
