//! End-to-end coverage for `std/math`: the pure-integer helpers (min/max/clamp/sign/pow,
//! plus `abs` over `core/ffi`'s `labs`) and the libm float ops (sqrt/floor/ceil) reached
//! through `core/ffi`'s safe wrappers — the float extern round-trips an f64 across the
//! AAPCS64 V-register boundary. Each check surfaces as the child's EXIT CODE (0/computed).

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const harness = @import("harness.zig");

const skipUnlessBackend = harness.skipUnlessBackend;

const compile = harness.compile;

const runExit = harness.runExit;

const buildAndRun = harness.buildAndRun;

test "math: integer helpers min/max/clamp/sign/pow/abs" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // 3 + 7 + 3 + 32 + 4 == 49, then + sign(-2)+sign(0)+sign(9) == 49 + 0 == 49.
    const code = try buildAndRun(gpa, io, ".toy-test-math-int",
        \\import std/math
        \\fn main() -> int {
        \\    base := math.min(3, 7) + math.max(3, 7) + math.clamp(5, 0, 3) + math.pow(2, 5) + math.abs(-4)
        \\    return base + math.sign(-2) + math.sign(0) + math.sign(9)
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 49), code);
}

test "math: libm sqrt over the float extern (16.0 -> 4)" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // The f64 arg rides a V-register into libSystem `sqrt`; the f64 result rides back and
    // truncates to 4. Proves float marshals correctly across the extern boundary.
    const code = try buildAndRun(gpa, io, ".toy-test-math-sqrt",
        \\import std/math
        \\fn main() -> int {
        \\    return math.sqrt(16.0).try_into().unwrap()
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 4), code);
}

test "math: libm floor/ceil round toward -inf/+inf" {
    const gpa = std.testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    try skipUnlessBackend(io);
    // floor(3.7)=3, ceil(3.2)=4 -> 3 + 4 == 7.
    const code = try buildAndRun(gpa, io, ".toy-test-math-floorceil",
        \\import std/math
        \\fn main() -> int {
        \\    f: int = math.floor(3.7).try_into().unwrap()
        \\    c: int = math.ceil(3.2).try_into().unwrap()
        \\    return f + c
        \\}
        \\
    );
    try std.testing.expectEqual(@as(u8, 7), code);
}
