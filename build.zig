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

    const mod = b.addModule("toy_compiler", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
    });
    mod.addOptions("build_options", options);

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
}

/// Hash every `.zig` file under `src/` (by path + contents, sorted for
/// determinism) into a single 64-bit compiler identity. Done at configure time
/// with native I/O, so only the digest — not the source bytes — ends up in the
/// binary, and adding a file never needs a code change. Returns 0 if `src/`
/// can't be read (degrades to a single shared cache namespace).
fn sourceDigest(b: *std.Build) u64 {
    const io = b.graph.io;
    var dir = b.build_root.handle.openDir(io, "src", .{ .iterate = true }) catch return 0;
    defer dir.close(io);

    var paths: std.ArrayList([]const u8) = .empty;
    var walker = dir.walk(b.allocator) catch return 0;
    defer walker.deinit();
    while (walker.next(io) catch return 0) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".zig")) continue;
        paths.append(b.allocator, b.allocator.dupe(u8, entry.path) catch return 0) catch return 0;
    }

    // Walk order is filesystem-dependent; sort so the digest is stable.
    std.mem.sort([]const u8, paths.items, {}, struct {
        fn lessThan(_: void, a: []const u8, c: []const u8) bool {
            return std.mem.lessThan(u8, a, c);
        }
    }.lessThan);

    var hasher = std.hash.Wyhash.init(0);
    for (paths.items) |p| {
        hasher.update(p);
        const bytes = dir.readFileAlloc(io, p, b.allocator, .unlimited) catch return 0;
        hasher.update(bytes);
    }
    return hasher.final();
}
