const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Compiler identity for the on-disk cache (see src/version.zig). Hashing the
    // whole `src/` tree at build time means every source file counts toward the
    // stamp automatically — no hand-maintained list to forget to update.
    const options = b.addOptions();
    options.addOption([]const u8, "semver", "0.0.0");
    options.addOption(u64, "source_digest", sourceDigest(b));
    // DEV-only pipeline inspection. When true, the CLI registers `--emit
    // lex|parse|ir` (front-end dump modes). Defaults to on in Debug (so the default
    // build + all test binaries expose `--emit`), off in a release build — a
    // ReleaseFast `toy` ships no `--emit` flag at all. Overridable with
    // `-Ddev-inspect=true/false`. Threaded through `build_options` exactly like
    // `source_digest`, so `Cli.zig` reads it at comptime to gate the option's presence.
    const dev_inspect = b.option(bool, "dev-inspect", "Register the --emit lex|parse|ir pipeline-inspection flag (default: on in Debug)") orelse (optimize == .Debug);
    options.addOption(bool, "dev_inspect", dev_inspect);

    const mod = b.addModule("toy_compiler", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    mod.addOptions("build_options", options);

    // The embedded standard library: `core/` and `std/` `.toy` source generated into
    // a single Zig module (`bundled_std`) at configure time, so `std`/`core` imports
    // resolve with no files on disk. Bundled source folds into `source_digest` above,
    // so a stdlib edit busts the compiler-identity cache stamp.
    const wf = b.addWriteFiles();
    const gen = wf.add("bundled_std.zig", bundledModulesSource(b));
    const bundled_mod = b.createModule(.{
        .root_source_file = gen,
        .target = target,
    });
    mod.addImport("bundled_std", bundled_mod);

    const exe = b.addExecutable(.{
        .name = "toy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/driver/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "toy_compiler", .module = mod },
            },
        }),
    });
    exe.root_module.addOptions("build_options", options);

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .name = "toy-test",
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    // Test executables cover one module at a time, so the exe's root module
    // needs its own test build separate from `mod_tests`.
    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    // NOTE: `zig build test` currently DEADLOCKS under the `zig build --listen`
    // runner because the integration tests in `mod_tests` spawn subprocesses
    // (codesign, the emitted ./prog) — a Zig 0.16 build-runner/IPC interaction;
    // the tests themselves all pass. `zig build test-bin` instead COMPILES and
    // installs the test binary to zig-out/bin/toy-test, which can then be run
    // DIRECTLY (`./zig-out/bin/toy-test`) to get a normal result without the
    // runner. This is the reliable way to run the suite until the runner issue
    // is resolved.
    const install_mod_tests = b.addInstallArtifact(mod_tests, .{});
    const test_bin_step = b.step("test-bin", "Build the test binary; run ./zig-out/bin/toy-test directly (avoids the runner hang)");
    test_bin_step.dependOn(&install_mod_tests.step);

    // Integration tests live in the repo-root tests/ and consume the compiler as a
    // BLACK BOX through the published `toy_compiler` module (src/root.zig's pub
    // exports) — `@import("toy_compiler")`, never `../` into src/ (a Zig module can't
    // import above its root). Their own test artifact keeps them out of the unit-test
    // (toy-test) binary; src/ stays library + inline unit tests, tests/ is integration.
    const integration_mod = b.createModule(.{
        .root_source_file = b.path("tests/integration.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "toy_compiler", .module = mod },
        },
    });
    const integration_tests = b.addTest(.{
        .name = "toy-integration-test",
        .root_module = integration_mod,
    });
    test_step.dependOn(&b.addRunArtifact(integration_tests).step);
    // Same runner-hang workaround: install the integration binary so it runs directly
    // (`./zig-out/bin/toy-integration-test`) alongside toy-test under `test-bin`.
    test_bin_step.dependOn(&b.addInstallArtifact(integration_tests, .{}).step);

    // `zig build fuzz`: an in-process front-end fuzzer (tests/fuzz.zig). It feeds
    // mutated seed-corpus bytes + grammar-generated programs through lex → parse →
    // (on a clean parse) resolve + typecheck, enforcing the robustness contract
    // (no panic / no hang / always a tree, and the parser invariants that fire under
    // runtime_safety). It is FORCED to Debug regardless of `-Doptimize`: those
    // invariant asserts only exist when `std.debug.runtime_safety` is on, so a
    // ReleaseFast build would silently turn the whole run into a no-op. Seeded +
    // bounded (FUZZ_SEED / FUZZ_ITERS envs), so it terminates fast and reproducibly.
    const fuzz_exe = b.addExecutable(.{
        .name = "toy-fuzz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/fuzz.zig"),
            .target = target,
            .optimize = .Debug,
            .imports = &.{
                .{ .name = "toy_compiler", .module = mod },
            },
        }),
    });
    const run_fuzz = b.addRunArtifact(fuzz_exe);
    // The fuzzer reads its seed corpus from tests/ui + examples relative to cwd, so
    // it must run from the build root (the default cwd for `zig build` run steps).
    if (b.args) |args| run_fuzz.addArgs(args);
    const fuzz_step = b.step("fuzz", "Build + run the bounded, seeded front-end fuzzer (FUZZ_SEED / FUZZ_ITERS envs)");
    fuzz_step.dependOn(&run_fuzz.step);
    // Also install the fuzz binary so it can be run directly for long soak runs.
    b.installArtifact(fuzz_exe);
}

/// Hash the compiler's source into a single 64-bit identity: every `.zig` under
/// `src/` PLUS every bundled-stdlib `.toy` under `core/` and `std/` (by path +
/// contents, sorted for determinism). Folding the bundled `.toy` in means a stdlib
/// edit busts the on-disk cache stamp exactly like a `src/` edit. Done at configure
/// time with native I/O, so only the digest — not the source bytes — ends up in the
/// binary. Returns 0 only if src/ can't be read (degrades to one shared cache
/// namespace); a missing bundled tree is folded as nothing, not a hard degrade.
fn sourceDigest(b: *std.Build) u64 {
    var hasher = std.hash.Wyhash.init(0);
    if (!hashDir(b, &hasher, "src", ".zig")) return 0;
    // Bundled stdlib is best-effort: a source tree that ships without core/std still
    // gets a meaningful src/-derived identity instead of collapsing to one shared
    // cache namespace (which would disable cross-version invalidation entirely).
    _ = hashDir(b, &hasher, "core", ".toy");
    _ = hashDir(b, &hasher, "std", ".toy");
    return hasher.final();
}

/// Fold every file under `sub` with extension `ext` (path + contents, sorted) into
/// `hasher`. Returns false if the directory can't be read.
fn hashDir(b: *std.Build, hasher: *std.hash.Wyhash, sub: []const u8, ext: []const u8) bool {
    const io = b.graph.io;
    var dir = b.build_root.handle.openDir(io, sub, .{ .iterate = true }) catch return false;
    defer dir.close(io);

    var paths: std.ArrayList([]const u8) = .empty;
    var walker = dir.walk(b.allocator) catch return false;
    defer walker.deinit();
    while (walker.next(io) catch return false) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ext)) continue;
        paths.append(b.allocator, b.allocator.dupe(u8, entry.path) catch return false) catch return false;
    }

    // Walk order is filesystem-dependent; sort so the digest is stable.
    std.mem.sort([]const u8, paths.items, {}, struct {
        fn lessThan(_: void, a: []const u8, c: []const u8) bool {
            return std.mem.lessThan(u8, a, c);
        }
    }.lessThan);

    // Fold the subdir name too so identical files under different roots can't alias.
    hasher.update(sub);
    for (paths.items) |p| {
        hasher.update(p);
        const bytes = dir.readFileAlloc(io, p, b.allocator, .unlimited) catch return false;
        hasher.update(bytes);
    }
    return true;
}

/// Generate the `bundled_std.zig` module source: one `Entry{ path, source }` per
/// `.toy` file under `core/` and `std/`, sorted by import path so the emitted array
/// (and thus the module's identity) is deterministic regardless of FS walk order.
/// The import path is `<root>/<rel>` with the trailing `.toy` stripped and OS
/// separators normalized to `/` (e.g. `core/ffi`).
fn bundledModulesSource(b: *std.Build) []const u8 {
    const Entry = struct { path: []const u8, source: []const u8 };
    var entries: std.ArrayList(Entry) = .empty;

    for ([_][]const u8{ "core", "std" }) |root| {
        var dir = b.build_root.handle.openDir(b.graph.io, root, .{ .iterate = true }) catch continue;
        defer dir.close(b.graph.io);
        var walker = dir.walk(b.allocator) catch continue;
        defer walker.deinit();
        while (walker.next(b.graph.io) catch break) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".toy")) continue;
            const bytes = dir.readFileAlloc(b.graph.io, entry.path, b.allocator, .unlimited) catch continue;
            // `<root>/<rel-without-.toy>`, separators normalized to `/`.
            const rel = entry.path[0 .. entry.path.len - ".toy".len];
            const joined = std.fmt.allocPrint(b.allocator, "{s}/{s}", .{ root, rel }) catch continue;
            const norm = b.allocator.dupe(u8, joined) catch continue;
            std.mem.replaceScalar(u8, norm, std.fs.path.sep, '/');
            entries.append(b.allocator, .{ .path = norm, .source = bytes }) catch continue;
        }
    }

    std.mem.sort(Entry, entries.items, {}, struct {
        fn lessThan(_: void, a: Entry, c: Entry) bool {
            return std.mem.lessThan(u8, a.path, c.path);
        }
    }.lessThan);

    var out: std.ArrayList(u8) = .empty;
    const gpa = b.allocator;
    out.appendSlice(gpa, "pub const Entry = struct { path: []const u8, source: []const u8 };\n") catch {};
    out.appendSlice(gpa, "pub const modules = [_]Entry{\n") catch {};
    for (entries.items) |e| {
        out.appendSlice(gpa, "    .{ .path = \"") catch {};
        appendZigStringBody(gpa, &out, e.path);
        out.appendSlice(gpa, "\", .source = \"") catch {};
        appendZigStringBody(gpa, &out, e.source);
        out.appendSlice(gpa, "\" },\n") catch {};
    }
    out.appendSlice(gpa, "};\n") catch {};
    return out.items;
}

/// Escape `s` into the body of a Zig double-quoted string literal (no surrounding
/// quotes), appending to `out`. A manual escaper (rather than a std helper whose
/// exact 0.16 API varies) so the generated file is always valid Zig.
fn appendZigStringBody(gpa: std.mem.Allocator, out: *std.ArrayList(u8), s: []const u8) void {
    for (s) |c| switch (c) {
        '"' => out.appendSlice(gpa, "\\\"") catch {},
        '\\' => out.appendSlice(gpa, "\\\\") catch {},
        '\n' => out.appendSlice(gpa, "\\n") catch {},
        '\r' => out.appendSlice(gpa, "\\r") catch {},
        '\t' => out.appendSlice(gpa, "\\t") catch {},
        else => if (c < 0x20 or c == 0x7f) {
            var buf: [4]u8 = undefined;
            const hex = std.fmt.bufPrint(&buf, "\\x{x:0>2}", .{c}) catch unreachable;
            out.appendSlice(gpa, hex) catch {};
        } else out.append(gpa, c) catch {},
    };
}
