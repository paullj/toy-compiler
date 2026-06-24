//! Tests for cache-key semantics. The e2e cold/warm run exercises the on-disk
//! round-trip; these pin down the digest logic in isolation.

const std = @import("std");
const testing = std.testing;
const Cache = @import("Cache.zig");

test "same phase + content => same digest" {
    const a = Cache.Key.fromSource(.lex, "native", "fn main() {}");
    const b = Cache.Key.fromSource(.lex, "native", "fn main() {}");
    try testing.expectEqual(a.digest(), b.digest());
}

test "different content => different digest" {
    const a = Cache.Key.fromSource(.lex, "native", "fn a() {}");
    const b = Cache.Key.fromSource(.lex, "native", "fn b() {}");
    try testing.expect(a.digest() != b.digest());
}

test "lexing is target-independent: target does not change the key" {
    // Lexing doesn't depend on the target, so the same source must reuse one
    // cache entry across targets rather than re-lex per target.
    const native = Cache.Key.fromSource(.lex, "native", "fn main() {}");
    const arm = Cache.Key.fromSource(.lex, "aarch64-macos", "fn main() {}");
    try testing.expectEqual(native.digest(), arm.digest());
}
