//! Proof that `place op= v` is `place = place op v`: every `compound_*` run case in
//! `tests/corpora/language-features/operators/` compiles, at -O0 and at -O1, to the same
//! binary as its hand-expanded twin in `compound_twins/`. The corpus harness checks that
//! each case runs correctly; this checks that it is the SAME program as the expanded form.

const std = @import("std");
const harness = @import("harness.zig");

test "compound: every compound run case is byte-identical to its hand-expanded twin" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try harness.skipUnlessBackend(io);
    try harness.expectTwins(gpa, io, .{
        .cases_dir = "tests/corpora/language-features/operators",
        .twins_dir = "packages/compiler/tests/compound_twins",
        .prefix = "compound_",
        .tmp = ".toy-test-compound-twins",
        .levels = &.{ "-O0", "-O1" },
    });
}
