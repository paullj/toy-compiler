//! Manifest-driven corpus harness over `tests/corpora/**` (integration).
//!
//! A single `.zon` manifest (`corpus_manifest.zon`) enumerates every corpus and the
//! MODE it runs in, so adding a corpus is a data edit, not a new harness. Two modes:
//!
//! `.diagnostics` — the exhaustive two-way `#~ ERROR [CODE] <substr>` matcher.
//!   Each fixture is a real toy program carrying `#~ ERROR [CODE] <substr>` (or
//!   `#~ WARN [CODE] <substr>`) annotations. Because `#` is the toy line-comment
//!   character (the lexer skips `# ...` to end-of-line) the whole annotation is an
//!   ordinary comment the compiler ignores; the `~` marker is what THIS harness keys
//!   on. An annotation applies to the line it sits ON: it expects an error-severity
//!   (`ERROR`) or warning-severity (`WARN`) diagnostic on that line whose
//!   `code == [CODE]` and whose message CONTAINS `<substr>`. The harness runs the
//!   whole-graph front-end `toy check` runs (discover, so a fixture may import std;
//!   resolve; typecheck), maps every entry-module diagnostic to its 1-based (line,
//!   code, message) via `SourceMap`, and matches EXHAUSTIVELY in BOTH directions (every
//!   annotation claims >=1 diagnostic; every emitted error, and every warning whose
//!   code some `WARN` annotation names, is claimed by some annotation). A diagnostic
//!   that moves line, changes code, or loses its message trips the harness.
//!
//! `.expect` — the `# expect:` directive harness (ported from the old `check.sh`
//!   pair). A fixture carries `# expect: exit <N>` / `# expect: stdout "<bytes>"` /
//!   `# expect: stderr "<substr>"` / `# expect: compile-error "<substr>"`. A
//!   compile-error fixture must FAIL the front-end with a diagnostic containing the
//!   substring; every other fixture must compile, run, and match its exit code, its
//!   exact stdout and/or a substring of its stderr. The run gets `# stdin: "<bytes>"`
//!   on a pipe (empty by default, never inherited), `# args: "a" "b c"` after argv[0],
//!   only the `# env: K=V` variables, and its worker's scratch dir as cwd. The build+run is the same
//!   in-process path the real `toy build` uses (`Graph.discover` -> `resolveGraph`
//!   -> `checkGraph` -> `lowerGraphProgram` -> `buildImage`), so it follows imports
//!   and multi-file module graphs. Codegen+exec is gated on macOS/aarch64; off that
//!   target a clean front-end counts as covered but does not run the binary. It lowers
//!   in `--verify` mode: each unit twice on a cache miss, or against its cached blob on
//!   a hit, so a nondeterministic or unsound unit fails its fixture. That makes this
//!   harness the determinism gate the retired `diff.sh` was.
//!
//! Each corpus states its exact fixture `count`, so a moved, renamed or removed
//! fixture (or a silently inert walk) fails the suite, and adding one means editing
//! the manifest. The fixtures run in parallel, each worker with its own cache
//! directory, and every failing fixture is reported, not only the first.
//!
//! This file is its own test binary, `toy-corpus-test`. Run it under `zig build test`,
//! or `zig build test-bin` then `./zig-out/bin/toy-corpus-test`. `TOY_CORPUS=<substr>`
//! runs only the fixtures whose path contains it:
//! `TOY_CORPUS=closures/capture_loop ./zig-out/bin/toy-corpus-test`.

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

/// One parsed `#~ ERROR|WARN [CODE] <substr>` annotation. `line` is 1-based (the line
/// the comment sits on). `matched` is flipped true when a diagnostic satisfies it.
const Annotation = struct {
    line: usize,
    code: []const u8,
    substr: []const u8,
    warn: bool = false,
    matched: bool = false,
};

/// A diagnostic reduced to what the matcher compares against. `matched` flips true
/// when an annotation claims it; an unmatched one at the end is a failure.
const Emitted = struct {
    line: usize,
    code: codes.Code,
    message: []const u8,
    warn: bool = false,
    matched: bool = false,
};

/// Parse the `#~ ERROR [CODE] <substr>` or `#~ WARN [CODE] <substr>` annotation out of
/// one source line, or null if the line carries none. The marker is exactly `#~` (a toy
/// comment `#` plus a `~`); a plain `#` comment, or a mistyped marker (`#-`, `# ~`,
/// `#~~`), yields no annotation so a typo can't silently disable an expectation. After
/// the marker the keyword `ERROR` or `WARN`, then `[CODE]`, then the rest of the line
/// (trimmed) is the substring the message must contain.
fn parseAnnotation(line_text: []const u8, line_1based: usize) ?Annotation {
    const marker = std.mem.indexOf(u8, line_text, "#~") orelse return null;
    var rest = std.mem.trimStart(u8, line_text[marker + 2 ..], " \t");
    // Require an exact keyword (a `~` immediately followed by more `~`, e.g. `#~~`,
    // leaves `rest` starting with neither -> null, rejecting it).
    const warn = std.mem.startsWith(u8, rest, "WARN");
    const kw: []const u8 = if (warn) "WARN" else "ERROR";
    if (!std.mem.startsWith(u8, rest, kw)) return null;
    rest = std.mem.trimStart(u8, rest[kw.len..], " \t");
    // `[CODE]`.
    if (rest.len == 0 or rest[0] != '[') return null;
    const close = std.mem.indexOfScalar(u8, rest, ']') orelse return null;
    const code = rest[1..close];
    const substr = std.mem.trim(u8, rest[close + 1 ..], " \t\r");
    return .{ .line = line_1based, .code = code, .substr = substr, .warn = warn };
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

/// Run the whole-graph front-end `toy check` runs over `path` and reduce every error and
/// warning to an `Emitted` (1-based line via `SourceMap`). A tainted entry parse stops at
/// its parse diagnostics; a clean graph yields resolve then (if resolve has no error)
/// typecheck diagnostics. A structural graph error, or a diagnostic located in an imported
/// module, fails the fixture: the matcher only sees the entry's lines. Caller owns the
/// returned list.
fn collectDiags(
    gpa: std.mem.Allocator,
    io: Io,
    cache: Cache,
    path: []const u8,
) !std.ArrayList(Emitted) {
    var graph = try Graph.discover(gpa, io, cache, Driver.default_target, path, .{});
    defer graph.deinit(gpa);
    const entry = graph.entry();

    var sm = try SourceMap.init(gpa, path, entry.source);
    defer sm.deinit(gpa);

    var out: std.ArrayList(Emitted) = .empty;
    errdefer freeEmitted(gpa, &out);

    const Push = struct {
        list: *std.ArrayList(Emitted),
        a: std.mem.Allocator,
        sm: *const SourceMap,
        tokens: []const Token,
        entry_index: u32,
        bad_spans: usize = 0,
        foreign: usize = 0,

        fn go(self: *@This(), diags: []const Diagnostic) !void {
            for (diags) |d| {
                if (d.scope != toyc.DiagnosticSink.NO_SCOPE and d.scope != self.entry_index) {
                    std.debug.print("{s}: diagnostic in imported module {d}: {s}\n", .{ self.sm.name, d.scope, d.message });
                    self.foreign += 1;
                    continue;
                }
                if (!checkSpans(self.sm, self.tokens, d)) self.bad_spans += 1;
                // The stored registry-default severity: the render-time config never
                // rewrites it.
                if (d.severity != .err and d.severity != .warning) continue;
                // Duplicate the borrowed message: the graph is freed on return, but the
                // matcher (and any failure message) reads it afterwards.
                const msg = try self.a.dupe(u8, d.message);
                errdefer self.a.free(msg);
                try self.list.append(self.a, .{
                    .line = self.sm.lineIndex(d.byte_offset) + 1, // 0-based -> 1-based
                    .code = d.code,
                    .message = msg,
                    .warn = d.severity == .warning,
                });
            }
        }
    };
    var push: Push = .{ .list = &out, .a = gpa, .sm = &sm, .tokens = entry.tokens, .entry_index = graph.entry_index };

    if (graph.err) |ge| {
        const in_entry = ge.module == null or ge.module == graph.entry_index;
        // A structural error at an import of the entry (a missing module, a cycle) is one
        // coded diagnostic there, as `toy check` renders it.
        if (ge.kind != .parse and in_entry and ge.span != null) {
            const msg = try gpa.dupe(u8, ge.message);
            errdefer gpa.free(msg);
            try out.append(gpa, .{ .line = sm.lineIndex(ge.span.?.start) + 1, .code = ge.code(), .message = msg, .warn = false });
            return out;
        }
        if (ge.kind != .parse or !in_entry) {
            std.debug.print("{s}: {s}\n", .{ path, ge.message });
            return error.GraphError;
        }
        try push.go(ge.parse_diags);
    } else {
        var res = try ResolveGraph.resolveGraph(gpa, &graph);
        defer res.deinit(gpa);
        try push.go(res.diags);
        if (countErr(res.diags) == 0) {
            var tc = try TypecheckGraph.checkGraph(gpa, &graph, &res, io, 0);
            defer tc.deinit(gpa);
            try push.go(tc.diags);
        }
    }
    if (push.bad_spans > 0) return error.SpanInvariant;
    if (push.foreign > 0) return error.ImportedModuleDiagnostic;

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

fn freeEmitted(gpa: std.mem.Allocator, list: *std.ArrayList(Emitted)) void {
    for (list.items) |e| gpa.free(@constCast(e.message));
    list.deinit(gpa);
}

/// The core exhaustive matcher, factored out so a NEGATIVE unit test can drive it
/// directly with a mismatched annotation set and assert rejection. Mutates the
/// `matched` flags in place; returns the first failure message (allocated with
/// `gpa`) or null on a full two-way match. An unclaimed warning fails only when some
/// `WARN` annotation names its code: error fixtures also trip incidental warnings
/// (an unused local, say) that they do not test.
fn matchExhaustive(
    gpa: std.mem.Allocator,
    file: []const u8,
    annotations: []Annotation,
    emitted: []Emitted,
) !?[]const u8 {
    // Forward: every annotation must match >=1 emitted diagnostic of its severity on its line.
    for (annotations) |*a| {
        for (emitted) |*e| {
            if (e.line != a.line or e.warn != a.warn) continue;
            const code_str = codes.str(e.code) orelse continue; // uncoded can't match a `[CODE]`
            if (!std.mem.eql(u8, code_str, a.code)) continue;
            if (std.mem.indexOf(u8, e.message, a.substr) == null) continue;
            a.matched = true;
            e.matched = true;
        }
        if (!a.matched) {
            return try std.fmt.allocPrint(
                gpa,
                "{s}:{d}: annotation [{s}] \"{s}\" matched no {s} diagnostic on that line",
                .{ file, a.line, a.code, a.substr, if (a.warn) "warning" else "error" },
            );
        }
    }
    // Reverse: every emitted error, and every warning of a code some WARN annotation
    // names, must have been claimed by some annotation.
    for (emitted) |e| {
        if (e.matched) continue;
        const code_str = codes.str(e.code) orelse "(none)";
        if (e.warn and !warnAnnotated(annotations, code_str)) continue;
        return try std.fmt.allocPrint(
            gpa,
            "{s}:{d}: emitted {s} [{s}] \"{s}\" has no matching #~ {s} annotation",
            .{ file, e.line, if (e.warn) "warning" else "error", code_str, e.message, if (e.warn) "WARN" else "ERROR" },
        );
    }
    return null;
}

fn warnAnnotated(annotations: []const Annotation, code: []const u8) bool {
    for (annotations) |a| if (a.warn and std.mem.eql(u8, a.code, code)) return true;
    return false;
}

/// Check one fixture end-to-end: collect its annotations + emitted diagnostics and run
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
        return try std.fmt.allocPrint(gpa, "{s}: fixture has no #~ ERROR or #~ WARN annotations", .{path});
    }

    var emitted = try collectDiags(gpa, io, cache, path);
    defer freeEmitted(gpa, &emitted);

    return matchExhaustive(gpa, path, anns.items, emitted.items);
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
    /// How many fixtures the walk must find. Exact rather than a floor: a floor goes
    /// stale as fixtures are added, and then losing one no longer fails.
    count: usize,
    /// First-path-component names to skip during a `single_file` walk (e.g. the
    /// `modules` subtree lives under `language-features` but is its own corpus).
    prune: []const []const u8 = &.{},
};

const manifest: []const Corpus = @import("corpus_manifest.zon");

/// The parsed directives of one fixture: the `# expect:` checks, and the `# stdin:`,
/// `# args:` and `# env:` inputs the run gets. `any` gates the "a fixture with no
/// expect directive is a mistake" check (matching the old bash). Every slice is owned
/// (see `deinit`).
const Expect = struct {
    any: bool = false,
    exit: ?u8 = null,
    stdout: ?[]const u8 = null,
    /// Substrings stderr must contain. Not an exact match: a panic's backtrace ends in
    /// unsymbolized frames whose addresses move with ASLR.
    stderr: std.ArrayList([]const u8) = .empty,
    cerr_expected: bool = false,
    cerr: []const u8 = "",
    stdin: std.ArrayList(u8) = .empty,
    args: std.ArrayList([]const u8) = .empty,
    /// `K=V` entries; the run's environment holds these and nothing else.
    env: std.ArrayList([]const u8) = .empty,

    fn deinit(self: *Expect, gpa: std.mem.Allocator) void {
        if (self.stdout) |s| gpa.free(s);
        if (self.cerr.len != 0) gpa.free(self.cerr);
        for (self.stderr.items) |s| gpa.free(s);
        self.stderr.deinit(gpa);
        self.stdin.deinit(gpa);
        for (self.args.items) |s| gpa.free(s);
        self.args.deinit(gpa);
        for (self.env.items) |s| gpa.free(s);
        self.env.deinit(gpa);
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

/// Split `# args:` into its quoted words (`"a" "b c"`), each escape-decoded. A word
/// that is not quoted is an error rather than a guess at shell splitting.
fn parseArgs(gpa: std.mem.Allocator, body: []const u8, out: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (true) {
        while (i < body.len and (body[i] == ' ' or body[i] == '\t' or body[i] == '\r')) i += 1;
        if (i == body.len) return;
        if (body[i] != '"') return error.BadArgsDirective;
        const start = i + 1;
        var j = start;
        while (j < body.len and body[j] != '"') : (j += 1) {
            if (body[j] == '\\') j += 1;
        }
        if (j >= body.len) return error.BadArgsDirective;
        const word = try decodeEscapes(gpa, body[start..j]);
        errdefer gpa.free(word);
        try out.append(gpa, word);
        i = j + 1;
    }
}

/// Parse every directive out of `source`. The `# expect:` checks are `exit <N>`,
/// `stdout "<bytes>"`, `stderr "<substr>"` and `compile-error "<substr>"`; the run
/// inputs are `# stdin: "<bytes>"` (repeats concatenate), `# args: "a" "b c"` (repeats
/// append) and `# env: K=V`. Caller `deinit`s.
fn parseExpect(gpa: std.mem.Allocator, source: []const u8) !Expect {
    var e: Expect = .{};
    errdefer e.deinit(gpa);
    var it = std.mem.splitScalar(u8, source, '\n');
    while (it.next()) |line| {
        if (std.mem.startsWith(u8, line, "# stdin:")) {
            const raw = betweenQuotes(line["# stdin:".len..]) orelse return error.BadStdinDirective;
            const bytes = try decodeEscapes(gpa, raw);
            defer gpa.free(bytes);
            try e.stdin.appendSlice(gpa, bytes);
            continue;
        }
        if (std.mem.startsWith(u8, line, "# args:")) {
            try parseArgs(gpa, line["# args:".len..], &e.args);
            continue;
        }
        if (std.mem.startsWith(u8, line, "# env:")) {
            const kv = std.mem.trim(u8, line["# env:".len..], " \t\r");
            const eq = std.mem.indexOfScalar(u8, kv, '=') orelse return error.BadEnvDirective;
            if (eq == 0) return error.BadEnvDirective;
            const owned = try gpa.dupe(u8, kv);
            errdefer gpa.free(owned);
            try e.env.append(gpa, owned);
            continue;
        }
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
        } else if (std.mem.startsWith(u8, body, "stderr")) {
            const raw = betweenQuotes(body) orelse return error.BadStderrDirective;
            const sub = try decodeEscapes(gpa, raw);
            errdefer gpa.free(sub);
            try e.stderr.append(gpa, sub);
            e.any = true;
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
    /// The program compiled and ran.
    ran: Ran,

    fn deinit(self: *RunResult, gpa: std.mem.Allocator) void {
        switch (self.*) {
            .compile_error => |s| gpa.free(s),
            .ran => |r| r.deinit(gpa),
            .skipped_exec => {},
        }
    }
};

/// One finished run of a binary; `stdout` and `stderr` are owned.
const Ran = struct {
    term: std.process.Child.Term,
    stdout: []const u8,
    stderr: []const u8 = "",

    fn deinit(r: Ran, gpa: std.mem.Allocator) void {
        gpa.free(r.stdout);
        gpa.free(r.stderr);
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
    e: *const Expect,
    cwd: []const u8,
) !RunResult {
    var graph = try Graph.discover(gpa, io, cache, Driver.default_target, entry, .{});
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

    var lowered = try DriverCodegen.lowerGraphProgram(gpa, io, cache, Driver.default_target, &graph, &res, &tc, .verify, .O0, true, null, null, 0);
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
    return .{ .ran = try runBinary(gpa, io, abs, e, cwd) };
}

/// Run `abs` with `e`'s args, stdin and environment, in `cwd`, capturing both output
/// streams. Every stdio fd is piped: under `zig build test` the test binary's stdin and
/// stderr are the build runner's `--listen=-` protocol stream, so a child that read
/// the inherited stdin would eat protocol bytes, and one that wrote a backtrace to the
/// inherited stderr would corrupt it. The environment starts empty so a fixture cannot
/// pass or fail on whatever the developer's shell exports.
fn runBinary(gpa: std.mem.Allocator, io: Io, abs: []const u8, e: *const Expect, cwd: []const u8) !Ran {
    const argv = try gpa.alloc([]const u8, 1 + e.args.items.len);
    defer gpa.free(argv);
    argv[0] = abs;
    @memcpy(argv[1..], e.args.items);

    var env: std.process.Environ.Map = .init(gpa);
    defer env.deinit();
    for (e.env.items) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=').?;
        try env.put(kv[0..eq], kv[eq + 1 ..]);
    }

    var child = try std.process.spawn(io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = &env,
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .pipe,
    });
    errdefer child.kill(io);
    // The fixtures' stdin is far below a pipe buffer, so writing it all before reading
    // any output cannot deadlock. A child that exits without reading it closes the pipe
    // first; that is the child's business, not a harness failure.
    child.stdin.?.writeStreamingAll(io, e.stdin.items) catch |err| switch (err) {
        error.BrokenPipe => {},
        else => return err,
    };
    child.stdin.?.close(io);
    child.stdin = null;

    var rdr = child.stdout.?.readerStreaming(io, &.{});
    const got = try rdr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(got);
    var errr = child.stderr.?.readerStreaming(io, &.{});
    const err_bytes = try errr.interface.allocRemaining(gpa, .limited(1 << 16));
    errdefer gpa.free(err_bytes);
    const term = try child.wait(io);
    return .{ .term = term, .stdout = got, .stderr = err_bytes };
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
            for (e.stderr.items) |want| {
                if (std.mem.indexOf(u8, run.stderr, want) == null) {
                    return try std.fmt.allocPrint(gpa, "{s}: stderr lacks \"{s}\" (got \"{s}\")", .{ name, want, run.stderr });
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
    cwd: []const u8,
) !?[]const u8 {
    const source = try Io.Dir.cwd().readFileAlloc(io, entry, gpa, .unlimited);
    defer gpa.free(source);

    var e = try parseExpect(gpa, source);
    defer e.deinit(gpa);

    var r = try compileAndRun(gpa, io, cache, entry, out_name, &e, cwd);
    defer r.deinit(gpa);

    return judge(gpa, entry, e, r);
}

/// One fixture: its cwd-relative path (owned) and how its corpus checks it.
const Fixture = struct { path: []const u8, mode: Mode };

fn freeFixtures(gpa: std.mem.Allocator, list: *std.ArrayList(Fixture)) void {
    for (list.items) |f| gpa.free(f.path);
    list.deinit(gpa);
}

/// Every fixture of every manifest corpus, sorted by path so the report reads the same on
/// every file system. A corpus whose walk finds other than its manifest `count` fails here.
fn collectFixtures(gpa: std.mem.Allocator, io: Io) !std.ArrayList(Fixture) {
    var list: std.ArrayList(Fixture) = .empty;
    errdefer freeFixtures(gpa, &list);
    var miscounted = false;
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
            // The in-process front-end reads every fixture through `Io.Dir.cwd()`, which
            // rejects an absolute path, and module imports resolve against the entry's
            // relative dir, so the path stays cwd-relative.
            const rel = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ c.path, entry.path });
            errdefer gpa.free(rel);
            try list.append(gpa, .{ .path = rel, .mode = c.mode });
            count += 1;
        }
        if (count != c.count) {
            std.debug.print("corpus [{s}]: found {d} fixtures, corpus_manifest.zon says {d}; update its count if the change is intended\n", .{ c.path, count, c.count });
            miscounted = true;
        }
    }
    if (miscounted) return error.CorpusMiscount;
    std.mem.sort(Fixture, list.items, {}, struct {
        fn lessThan(_: void, a: Fixture, b: Fixture) bool {
            return std.mem.lessThan(u8, a.path, b.path);
        }
    }.lessThan);
    return list;
}

/// The fixtures, a shared cursor into them, and one result slot per fixture. Workers
/// claim fixtures in order, so a slow one never holds back the rest.
const Run = struct {
    gpa: std.mem.Allocator,
    io: Io,
    fixtures: []const Fixture,
    /// Per fixture, its failure message (owned) or null.
    failures: []?[]const u8,
    /// Where `.expect` fixtures write their binaries, one name per fixture index.
    out_dir: []const u8,
    next: std.atomic.Value(usize) = .init(0),

    /// `cwd` is the worker's own scratch dir, so a fixture that touches files never
    /// races another worker's fixture.
    fn work(run: *Run, cache: Cache, cwd: []const u8) void {
        while (true) {
            const i = run.next.fetchAdd(1, .monotonic);
            if (i >= run.fixtures.len) return;
            const f = run.fixtures[i];
            run.failures[i] = run.check(cache, cwd, i) catch |e| blk: {
                const msg: []const u8 = std.fmt.allocPrint(run.gpa, "{s}: {t}", .{ f.path, e }) catch @panic("OOM");
                break :blk msg;
            };
        }
    }

    fn check(run: *Run, cache: Cache, cwd: []const u8, i: usize) !?[]const u8 {
        const f = run.fixtures[i];
        return switch (f.mode) {
            .diagnostics => checkFixture(run.gpa, run.io, cache, f.path),
            .expect => {
                const out_name = try std.fmt.allocPrint(run.gpa, "{s}/f{d}", .{ run.out_dir, i });
                defer run.gpa.free(out_name);
                return checkExpectFixture(run.gpa, run.io, cache, f.path, out_name, cwd);
            },
        };
    }
};

test "corpus: every manifest corpus matches its expectations" {
    var leak_check: LeakCheck = .init;
    const result = runCorpus(leak_check.allocator());
    const leaked = leak_check.deinit() == .leak;
    try result;
    try testing.expect(!leaked);
}

/// `std.testing.allocator` captures a stack trace on every allocation, which costs more
/// than the compiles; this one still fails the test on a leak, without its site.
const LeakCheck = std.heap.DebugAllocator(.{ .stack_trace_frames = 0 });

fn runCorpus(gpa: std.mem.Allocator) !void {
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var all = try collectFixtures(gpa, io);
    defer freeFixtures(gpa, &all);
    try testing.expect(all.items.len > 0);

    var picked: std.ArrayList(Fixture) = .empty;
    defer picked.deinit(gpa);
    const filter: ?[]const u8 = if (testing.environ.getPosix("TOY_CORPUS")) |sub| (if (sub.len == 0) null else sub) else null;
    for (all.items) |f| {
        if (filter == null or std.mem.indexOf(u8, f.path, filter.?) != null) try picked.append(gpa, f);
    }
    if (filter) |sub| {
        std.debug.print("corpus: {d} of {d} fixtures match TOY_CORPUS={s}\n", .{ picked.items.len, all.items.len, sub });
        if (picked.items.len == 0) return error.NoFixtureMatches;
    }

    // Emitted binaries and the per-worker caches land here, relative to cwd, so duplicate
    // fixture basenames can't clobber each other.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var out_dir_buf: [64]u8 = undefined;
    const out_dir = std.fmt.bufPrint(&out_dir_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;

    const failures = try gpa.alloc(?[]const u8, picked.items.len);
    defer gpa.free(failures);
    @memset(failures, null);
    var run: Run = .{ .gpa = gpa, .io = io, .fixtures = picked.items, .failures = failures, .out_dir = out_dir };

    // A cache per worker: two workers writing one entry at once would share its temp file.
    const workers = @min(picked.items.len, std.Thread.getCpuCount() catch 1);
    const dirs = try gpa.alloc([80]u8, workers);
    defer gpa.free(dirs);
    const cwds = try gpa.alloc([80]u8, workers);
    defer gpa.free(cwds);
    const threads = try gpa.alloc(std.Thread, workers);
    defer gpa.free(threads);
    {
        var spawned: usize = 0;
        defer for (threads[0..spawned]) |t| t.join();
        for (dirs, 0..) |*buf, w| {
            const cache = try Cache.init(io, std.fmt.bufPrint(buf, "{s}/cache{d}", .{ out_dir, w }) catch unreachable);
            const cwd = std.fmt.bufPrint(&cwds[w], "{s}/cwd{d}", .{ out_dir, w }) catch unreachable;
            try Io.Dir.cwd().createDirPath(io, cwd);
            threads[w] = try std.Thread.spawn(.{}, Run.work, .{ &run, cache, cwd });
            spawned += 1;
        }
    }

    var failed: usize = 0;
    for (failures) |msg| if (msg) |m| {
        failed += 1;
        std.debug.print("corpus mismatch: {s}\n", .{m});
        gpa.free(m);
    };
    if (failed > 0) {
        std.debug.print("corpus: {d} of {d} fixtures failed\n", .{ failed, picked.items.len });
        return error.CorpusMismatch;
    }
}

test "ui harness: parseAnnotation accepts the exact marker and rejects near-misses" {
    // Accepted: the canonical form.
    {
        const a = parseAnnotation("  aaa  #~ ERROR [R0001] undeclared identifier 'aaa'", 2).?;
        try testing.expectEqual(@as(usize, 2), a.line);
        try testing.expectEqualStrings("R0001", a.code);
        try testing.expectEqualStrings("undeclared identifier 'aaa'", a.substr);
    }
    // Accepted: a warning annotation.
    {
        const a = parseAnnotation("fn each() {  #~ WARN [W0011] shadows", 3).?;
        try testing.expect(a.warn);
        try testing.expectEqualStrings("W0011", a.code);
        try testing.expectEqualStrings("shadows", a.substr);
        try testing.expect(!parseAnnotation("  #~ ERROR [R0001] x", 1).?.warn);
    }
    // Rejected: a plain comment (no `~`).
    try testing.expect(parseAnnotation("  # ERROR [R0001] x", 1) == null);
    try testing.expect(parseAnnotation("  #~ WARNING [W0011] x", 1) == null);
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
        var errs = [_]Emitted{.{ .line = 2, .code = .R0001, .message = "undeclared identifier 'x'" }};
        const failure = (try matchExhaustive(gpa, "t.toy", &anns, &errs)).?;
        defer gpa.free(failure);
        try testing.expect(std.mem.indexOf(u8, failure, "matched no error diagnostic") != null);
    }
    // 2) Wrong substring: code matches, message does not contain the substring.
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "R0001", .substr = "TOTALLY WRONG" }};
        var errs = [_]Emitted{.{ .line = 2, .code = .R0001, .message = "undeclared identifier 'x'" }};
        const failure = (try matchExhaustive(gpa, "t.toy", &anns, &errs)).?;
        defer gpa.free(failure);
        try testing.expect(std.mem.indexOf(u8, failure, "matched no error diagnostic") != null);
    }
    // 3) Unannotated error: an emitted error with NO covering annotation -> reverse
    //    match fails (unmatched diagnostic).
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "R0001", .substr = "aaa" }};
        var errs = [_]Emitted{
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
        var errs = [_]Emitted{.{ .line = 2, .code = .R0001, .message = "undeclared identifier 'aaa'" }};
        try testing.expect((try matchExhaustive(gpa, "t.toy", &anns, &errs)) == null);
    }
}

test "ui harness: WARN annotations match warnings only, and claim every warning of their code" {
    const gpa = testing.allocator;
    const shadow = "method 'each' on 'Vec' shadows the one std/vec declares";
    // A WARN annotation is not satisfied by an error of the same code and line.
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "W0011", .substr = "shadows", .warn = true }};
        var got = [_]Emitted{.{ .line = 2, .code = .W0011, .message = shadow }};
        const failure = (try matchExhaustive(gpa, "t.toy", &anns, &got)).?;
        defer gpa.free(failure);
        try testing.expect(std.mem.indexOf(u8, failure, "matched no warning diagnostic") != null);
    }
    // A second warning of an annotated code must be annotated too.
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "W0011", .substr = "shadows", .warn = true }};
        var got = [_]Emitted{
            .{ .line = 2, .code = .W0011, .message = shadow, .warn = true },
            .{ .line = 5, .code = .W0011, .message = shadow, .warn = true },
        };
        const failure = (try matchExhaustive(gpa, "t.toy", &anns, &got)).?;
        defer gpa.free(failure);
        try testing.expect(std.mem.indexOf(u8, failure, "no matching #~ WARN annotation") != null);
    }
    // A warning of a code no WARN annotation names is incidental: ignored.
    {
        var anns = [_]Annotation{.{ .line = 2, .code = "R0001", .substr = "aaa" }};
        var got = [_]Emitted{
            .{ .line = 2, .code = .R0001, .message = "undeclared identifier 'aaa'" },
            .{ .line = 3, .code = .W0001, .message = "unused variable 'b'", .warn = true },
        };
        try testing.expect((try matchExhaustive(gpa, "t.toy", &anns, &got)) == null);
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

test "expect harness: parseExpect reads stderr, stdin, args and env directives" {
    const gpa = testing.allocator;
    var e = try parseExpect(gpa,
        \\# expect: stderr "boom\n"
        \\# expect: stderr "main"
        \\# stdin: "a\n"
        \\# stdin: "b"
        \\# args: "x" "y z"  "q\"r"
        \\# args: "w"
        \\# env: K=V=W
        \\# env: EMPTY=
        \\
    );
    defer e.deinit(gpa);
    try testing.expect(e.any);
    try testing.expectEqual(@as(usize, 2), e.stderr.items.len);
    try testing.expectEqualStrings("boom\n", e.stderr.items[0]);
    try testing.expectEqualStrings("a\nb", e.stdin.items);
    try testing.expectEqual(@as(usize, 4), e.args.items.len);
    try testing.expectEqualStrings("y z", e.args.items[1]);
    try testing.expectEqualStrings("q\"r", e.args.items[2]);
    try testing.expectEqualStrings("w", e.args.items[3]);
    try testing.expectEqualStrings("K=V=W", e.env.items[0]);
    try testing.expectEqualStrings("EMPTY=", e.env.items[1]);

    // Input directives alone are not a check.
    var only_in = try parseExpect(gpa, "# stdin: \"x\"\n# args: \"a\"\n");
    defer only_in.deinit(gpa);
    try testing.expect(!only_in.any);

    try testing.expectError(error.BadArgsDirective, parseExpect(gpa, "# args: bare\n"));
    try testing.expectError(error.BadArgsDirective, parseExpect(gpa, "# args: \"open\n"));
    try testing.expectError(error.BadEnvDirective, parseExpect(gpa, "# env: NOEQ\n"));
    try testing.expectError(error.BadEnvDirective, parseExpect(gpa, "# env: =V\n"));
    try testing.expectError(error.BadStdinDirective, parseExpect(gpa, "# stdin: x\n"));
    try testing.expectError(error.BadStderrDirective, parseExpect(gpa, "# expect: stderr x\n"));
}

test "expect harness: judge matches stderr by substring" {
    const gpa = testing.allocator;
    var e: Expect = .{ .any = true };
    defer e.stderr.deinit(gpa);
    try e.stderr.append(gpa, "boom\n");
    const backtrace = "boom\n0x0000000000000054 main\n0x000000008f4fbbbc \n";
    try testing.expect((try judge(gpa, "t.toy", e, .{ .ran = .{ .term = .{ .exited = 1 }, .stdout = "", .stderr = backtrace } })) == null);
    try expectJudgeFailure(gpa, e, .{ .ran = .{ .term = .{ .exited = 1 }, .stdout = "", .stderr = "bang\n" } }, "stderr lacks");
}

test "expect harness: runBinary pipes stdin, passes args, empties env, and sets cwd" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var cwd_buf: [64]u8 = undefined;
    const cwd = try std.fmt.bufPrint(&cwd_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path});
    try tmp.dir.writeFile(io, .{ .sub_path = "marker", .data = "" });

    var e = try parseExpect(gpa,
        \\# stdin: "in\n"
        \\# args: "a" "b c"
        \\# env: K=V
        \\
    );
    defer e.deinit(gpa);
    // `$HOME` is set in every developer shell, so its absence proves the env is not inherited.
    const script = "cat; printf '%s|' \"$#\" \"$1\" \"$2\" \"$K\" \"${HOME-unset}\"; ls; echo err >&2; exit 3";
    var sh_e: Expect = .{};
    defer sh_e.args.deinit(gpa);
    try sh_e.args.appendSlice(gpa, &.{ "-c", script, "sh" });
    try sh_e.args.appendSlice(gpa, e.args.items);
    sh_e.stdin = e.stdin;
    sh_e.env = e.env;

    const r = try runBinary(gpa, io, "/bin/sh", &sh_e, cwd);
    defer r.deinit(gpa);
    try testing.expectEqual(@as(?u8, 3), exitCode(r.term));
    try testing.expectEqualStrings("in\n2|a|b c|V|unset|marker\n", r.stdout);
    try testing.expectEqualStrings("err\n", r.stderr);

    // With no `# stdin:` the child reads EOF at once instead of blocking on an inherited fd.
    var empty: Expect = .{};
    defer empty.args.deinit(gpa);
    try empty.args.appendSlice(gpa, &.{ "-c", "cat; echo done" });
    const r2 = try runBinary(gpa, io, "/bin/sh", &empty, cwd);
    defer r2.deinit(gpa);
    try testing.expectEqualStrings("done\n", r2.stdout);
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
