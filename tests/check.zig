//! `toy check` emitter integration tests (black box via the published surface).
//!
//! These drive the SAME front-end the `check` subcommand runs — `Driver.run(.check)`
//! over a real temp source file — then exercise the `Check` emitters (`emitNdjson`,
//! `tallyAll`) that back the subcommand. They pin the NDJSON wire shape, the
//! severity-config-aware error/warning tallies (the exit gate), and the "collect ALL
//! diagnostics" contract (a tainted parse still surfaces its diagnostics under check).

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");

const Driver = toyc.Driver;
const Check = toyc.Check;
const SevCfg = toyc.diagnostics.severity_config;
const testing = std.testing;

/// Run `Driver.run(.check)` over one temp source file and hand the results to `body`.
/// The temp dir is unique + torn down, so tests never disturb the repo.
fn withCheck(
    comptime dir_name: []const u8,
    src: []const u8,
    body: fn (results: []Driver.FileResult) anyerror!void,
) !void {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const path = dir_name ++ "/c.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = src });

    const results = try Driver.run(gpa, io, .check, "native", &.{path});
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }
    try body(results);
}

test "check: a resolve error is tallied as one error and emitted as NDJSON" {
    const src = "fn main() -> int {\n  return nope_zzq\n}\n";
    try withCheck(".toy-test-check-err", src, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            const gpa = testing.allocator;
            // tallyAll (the exit gate): exactly one error, no warnings.
            const counts = Check.tallyAll(results, .{});
            try testing.expectEqual(@as(usize, 1), counts.errors);
            try testing.expectEqual(@as(usize, 0), counts.warnings);
            try testing.expect(counts.hasErrors());

            // NDJSON: exactly one line, valid JSON, level "error", R0001 code.
            var aw: std.Io.Writer.Allocating = .init(gpa);
            defer aw.deinit();
            const nd = try Check.emitNdjson(&aw.writer, gpa, results, .{});
            try testing.expectEqual(@as(usize, 1), nd.errors);
            const out = aw.written();
            try testing.expectEqual(@as(usize, 1), std.mem.count(u8, out, "\n"));
            try testing.expect(std.mem.endsWith(u8, out, "\n"));
            try testing.expect(std.mem.indexOf(u8, out, "\"level\":\"error\"") != null);
            try testing.expect(std.mem.indexOf(u8, out, "\"code\":\"R0001\"") != null);
            // The message carries the offending identifier verbatim in the JSON string.
            try testing.expect(std.mem.indexOf(u8, out, "nope_zzq") != null);
            // Parses as strict JSON per line (drop the trailing newline).
            const line = out[0 .. out.len - 1];
            var parsed = try std.json.parseFromSlice(std.json.Value, gpa, line, .{});
            defer parsed.deinit();
            // D2 enrichment: every line carries `file`, `rendered`, and a `labels` array.
            const obj = parsed.value.object;
            try testing.expect(obj.get("file") != null);
            try testing.expect(obj.get("file").?.string.len > 0);
            const rendered = obj.get("rendered").?.string;
            // `rendered` is the PLAIN single-diagnostic render: the coded header + a caret
            // line, with the embedded newline preserved through the JSON round-trip.
            try testing.expect(std.mem.indexOf(u8, rendered, "error[R0001]") != null);
            try testing.expect(std.mem.indexOf(u8, rendered, "-->") != null);
            try testing.expect(std.mem.indexOf(u8, rendered, "\n") != null);
            // No secondary labels on a plain undeclared-identifier error.
            try testing.expectEqual(@as(usize, 0), obj.get("labels").?.array.items.len);
        }
    }.body);
}

test "check: a duplicate-definition NDJSON line carries a `previously defined here` label" {
    const src =
        \\fn f() -> int {
        \\  return 1
        \\}
        \\fn f() -> int {
        \\  return 2
        \\}
        \\fn main() -> int {
        \\  return f()
        \\}
    ;
    try withCheck(".toy-test-check-dup-label", src, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            const gpa = testing.allocator;
            var aw: std.Io.Writer.Allocating = .init(gpa);
            defer aw.deinit();
            _ = try Check.emitNdjson(&aw.writer, gpa, results, .{});
            const out = aw.written();
            // Find the R0002 line and parse it as strict JSON.
            var it = std.mem.tokenizeScalar(u8, out, '\n');
            var checked = false;
            while (it.next()) |line| {
                if (std.mem.indexOf(u8, line, "\"code\":\"R0002\"") == null) continue;
                checked = true;
                var parsed = try std.json.parseFromSlice(std.json.Value, gpa, line, .{});
                defer parsed.deinit();
                const labels = parsed.value.object.get("labels").?.array;
                try testing.expectEqual(@as(usize, 1), labels.items.len);
                const lbl = labels.items[0].object;
                try testing.expectEqualStrings("previously defined here", lbl.get("message").?.string);
                try testing.expectEqual(false, lbl.get("is_primary").?.bool);
                // The label carries its own location + byte span (the first definition).
                try testing.expect(lbl.get("line").?.integer >= 1);
                try testing.expect(lbl.get("col").?.integer >= 1);
                try testing.expect(lbl.get("byte_start") != null);
                try testing.expect(lbl.get("byte_end") != null);
                try testing.expect(lbl.get("file") != null);
            }
            try testing.expect(checked);
        }
    }.body);
}

test "check: --warn demotes a code so the error tally drops to a warning (exit gate)" {
    const src = "fn main() -> int {\n  return nope_zzq\n}\n";
    try withCheck(".toy-test-check-warn", src, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            var rules = [_]SevCfg.Rule{.{ .match = "R0001", .action = .warning }};
            const cfg = SevCfg.SeverityConfig{ .rules = &rules };
            const counts = Check.tallyAll(results, cfg);
            try testing.expectEqual(@as(usize, 0), counts.errors);
            try testing.expectEqual(@as(usize, 1), counts.warnings);
            try testing.expect(!counts.hasErrors()); // exit 0 under --warn
        }
    }.body);
}

test "check: --ignore suppresses a code from both the tally and the NDJSON stream" {
    const src = "fn main() -> int {\n  return nope_zzq\n}\n";
    try withCheck(".toy-test-check-ignore", src, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            const gpa = testing.allocator;
            var rules = [_]SevCfg.Rule{.{ .match = "R0001", .action = .ignore }};
            const cfg = SevCfg.SeverityConfig{ .rules = &rules };
            const counts = Check.tallyAll(results, cfg);
            try testing.expectEqual(@as(usize, 0), counts.errors);
            try testing.expectEqual(@as(usize, 0), counts.warnings);

            var aw: std.Io.Writer.Allocating = .init(gpa);
            defer aw.deinit();
            _ = try Check.emitNdjson(&aw.writer, gpa, results, cfg);
            try testing.expectEqual(@as(usize, 0), aw.written().len); // nothing emitted
        }
    }.body);
}

test "check: a clean program yields zero diagnostics and no NDJSON output" {
    const src = "fn main() -> int {\n  return 0\n}\n";
    try withCheck(".toy-test-check-clean", src, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            const gpa = testing.allocator;
            const counts = Check.tallyAll(results, .{});
            try testing.expectEqual(@as(usize, 0), counts.errors);
            try testing.expectEqual(@as(usize, 0), counts.warnings);
            try testing.expect(!counts.hasErrors());

            var aw: std.Io.Writer.Allocating = .init(gpa);
            defer aw.deinit();
            _ = try Check.emitNdjson(&aw.writer, gpa, results, .{});
            try testing.expectEqual(@as(usize, 0), aw.written().len);
        }
    }.body);
}

test "check: a tainted parse still surfaces its diagnostics (no early-bail under check)" {
    // A syntactically broken body: the parser taints but keeps a partial tree; check
    // must still report the parse diagnostic(s) rather than bailing silently.
    const src = "fn main() -> int {\n  return\n";
    try withCheck(".toy-test-check-tainted", src, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            const counts = Check.tallyAll(results, .{});
            try testing.expect(counts.errors >= 1); // at least one parse diagnostic
        }
    }.body);
}

/// Run `Driver.run(.check)` over the given paths (no temp file written) and hand the
/// results to `body`. Used for hard-error paths where a path deliberately does NOT
/// exist, so the FileResult carries `err` with zero diagnostics.
fn withCheckPaths(
    paths: []const []const u8,
    body: fn (results: []Driver.FileResult) anyerror!void,
) !void {
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const results = try Driver.run(gpa, io, .check, "native", paths);
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }
    try body(results);
}

test "check: a missing file is a hard error in BOTH forms (exit 2, not clean)" {
    // Regression: the NDJSON branch used to skip the hard-error gate the human branch
    // applied, so `check --format ndjson <missing>` printed nothing and exited 0 — a
    // missing file looked "checked clean" to a machine consumer. Both forms must treat
    // an `err`-with-no-diagnostics FileResult as a hard error (exit 2). We assert the
    // SHARED predicate `Check.anyHardError` (the exact gate both branches now consult)
    // plus the "no diagnostics were produced" invariant that makes the human form's
    // tally 0 yet the exit code 2.
    try withCheckPaths(&.{"does_not_exist_zzq.toy"}, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            const gpa = testing.allocator;
            try testing.expectEqual(@as(usize, 1), results.len);
            // The missing file surfaces as a hard error: `err` set, zero diagnostics.
            try testing.expect(results[0].err != null);
            try testing.expectEqual(@as(usize, 0), results[0].diags.len);

            // The shared gate both runCheck branches use fires -> exit 2.
            try testing.expect(Check.anyHardError(results));

            // Human form: the diagnostic tally is 0 (a hard error is not a compile
            // diagnostic), so exit code must come from the hard-error gate, not the tally.
            const counts = Check.tallyAll(results, .{});
            try testing.expectEqual(@as(usize, 0), counts.errors);
            try testing.expectEqual(@as(usize, 0), counts.warnings);

            // NDJSON form: a missing file has no diagnostics, so emitNdjson writes
            // nothing and reports zero counts — WITHOUT the anyHardError gate this
            // would map to exit 0. The gate above is what forces exit 2.
            var aw: std.Io.Writer.Allocating = .init(gpa);
            defer aw.deinit();
            const nd = try Check.emitNdjson(&aw.writer, gpa, results, .{});
            try testing.expectEqual(@as(usize, 0), aw.written().len);
            try testing.expectEqual(@as(usize, 0), nd.errors);
            try testing.expectEqual(@as(usize, 0), nd.warnings);
        }
    }.body);
}

test "check: a clean file mixed with a missing file still trips the hard-error gate" {
    // Multi-file divergence: one file checks clean, one is missing. The human branch
    // returned 2; the NDJSON branch used to emit the clean file's (empty) stream and
    // exit 0. anyHardError must fire because ANY file is a hard error.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const dir_name = ".toy-test-check-mixed";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const ok_path = dir_name ++ "/ok.toy";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = ok_path, .data = "fn main() -> int {\n  return 0\n}\n" });
    const missing_path = dir_name ++ "/missing_zzq.toy";

    const results = try Driver.run(gpa, io, .check, "native", &.{ ok_path, missing_path });
    defer {
        for (results) |*r| r.deinit(gpa);
        gpa.free(results);
    }

    // No visible diagnostics (the clean file has none, the missing file has none), yet
    // the hard-error gate fires because one file failed to read -> exit 2.
    const counts = Check.tallyAll(results, .{});
    try testing.expectEqual(@as(usize, 0), counts.errors);
    try testing.expect(Check.anyHardError(results));
}

test "check: a compile-diagnostic file is NOT a hard error (exit 1, not 2)" {
    // A resolve error sets `err` to a stage sentinel AND carries diagnostics. That is a
    // compile failure (exit 1), so anyHardError must be FALSE — the diagnostic-count
    // guard is what separates it from a genuine IO hard error.
    const src = "fn main() -> int {\n  return nope_zzq\n}\n";
    try withCheck(".toy-test-check-not-hard", src, struct {
        fn body(results: []Driver.FileResult) anyerror!void {
            try testing.expect(results[0].err != null); // stage sentinel set
            try testing.expect(!Check.anyHardError(results)); // but has diags -> not hard
        }
    }.body);
}
