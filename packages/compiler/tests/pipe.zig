//! Proof that `|>` is pure syntax: every run case in
//! `tests/corpora/language-features/pipe/` compiles to the same binary as its
//! hand-desugared twin in `pipe_twins/`. The corpus harness already checks that each
//! case runs correctly; this checks that it is the SAME program as the call form.
//!
//! One shape is kept out of the cases: a lambda in a hole pipe's lhs AND in an argument
//! left of its `_` (`fn.. |> f(fn.., _)`). The lhs lambda's nodes are created first but it
//! sits later in the call, and lambda units are emitted in node order, so the twin's two
//! units swap places: the same program, different bytes (packages/docs/pipe.md).

const std = @import("std");
const harness = @import("harness.zig");

const cases_dir = "tests/corpora/language-features/pipe";
const twins_dir = "packages/compiler/tests/pipe_twins";

const skipUnlessBackend = harness.skipUnlessBackend;

test "pipe: every pipe run case is byte-identical to its hand-desugared twin" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    try harness.expectTwins(gpa, io, .{ .cases_dir = cases_dir, .twins_dir = twins_dir, .tmp = ".toy-test-pipe-twins" });
}
