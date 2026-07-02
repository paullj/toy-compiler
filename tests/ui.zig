//! UI-test harness over `tests/ui/**/*.toy` fixtures with inline expectation
//! annotations (integration).
//!
//! Each fixture is a real toy program carrying `#~ ERROR [CODE] <substr>`
//! annotations. Because `#` is the toy line-comment character (the lexer skips
//! `# ...` to end-of-line) the whole annotation is an ordinary comment the
//! compiler ignores; the `~` marker is what THIS harness keys on. An annotation
//! applies to the line it sits ON: it expects an error-severity diagnostic on that
//! line whose `code == [CODE]` and whose message CONTAINS `<substr>`.
//!
//! The harness runs the in-process front-end (`Driver.pipeline` in `.check` mode:
//! lex -> parse -> resolve -> typecheck, collecting ALL diagnostics), maps every
//! diagnostic to its 1-based (line, code, message) via `SourceMap`, and matches
//! EXHAUSTIVELY in BOTH directions:
//!   - every annotation must match >=1 diagnostic on its line with that code +
//!     substring, and
//!   - every emitted error diagnostic must have a matching annotation.
//! Any unmatched annotation OR unmatched diagnostic fails the test with a message
//! naming the file + line. The two-way match is what makes the corpus a genuine
//! regression net: a diagnostic that moves line, changes code, or loses its
//! message trips the harness.
//!
//! Run: `zig build test-bin` then `./zig-out/bin/toy-integration-test`.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");

const Driver = toyc.Driver;
const Cache = toyc.Cache;
const version = toyc.version;
const codes = toyc.diagnostics.codes;
const SourceMap = toyc.term.render.SourceMap;
const Diagnostic = toyc.diagnostics.Diagnostic.Diagnostic;

const testing = std.testing;

const ui_dir = "tests/ui";

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
        fn go(list: *std.ArrayList(EmittedError), a: std.mem.Allocator, s: *const SourceMap, diags: []const Diagnostic) !void {
            for (diags) |d| {
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

    if (r.diags.len > 0) try push(&out, gpa, &sm, r.diags);
    if (r.resolve) |res| try push(&out, gpa, &sm, res.diags);
    if (r.typecheck) |tc| try push(&out, gpa, &sm, tc.diags);

    return out;
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

test "ui fixtures: every tests/ui/**/*.toy matches its inline #~ ERROR annotations exhaustively" {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    var dir_buf: [Driver.cache_root.len + 1 + version.stamp_max + "/cache".len]u8 = undefined;
    var stamp_buf: [version.stamp_max]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/{s}/cache", .{ Driver.cache_root, version.stamp(&stamp_buf) }) catch unreachable;
    const cache = try Cache.init(io, dir);

    var root = try Io.Dir.cwd().openDir(io, ui_dir, .{ .iterate = true });
    defer root.close(io);

    var walker = try root.walk(gpa);
    defer walker.deinit();

    var count: usize = 0;
    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".toy")) continue;
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ ui_dir, entry.path });
        count += 1;
        if (try checkFixture(gpa, io, cache, path)) |failure| {
            defer gpa.free(failure);
            std.debug.print("UI fixture mismatch: {s}\n", .{failure});
            return error.UiFixtureMismatch;
        }
    }
    // The corpus must be non-empty, else the harness is silently inert.
    try testing.expect(count >= 5);
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
