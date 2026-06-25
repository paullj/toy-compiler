//! Exact constant-fold arithmetic for the opt stage.
//!
//! These helpers MUST replicate what `CodegenIr.genArith` + `Aarch64` emit
//! bit-for-bit, otherwise opt-on diverges from opt-off (the differential test
//! is the backstop, but correctness lives here). Verified against
//! CodegenIr.zig:362-429 + Aarch64.zig:
//!   add/sub/mul : 64-bit two's-complement WRAPPING (`+%`/`-%`/`*%`).
//!   sdiv        : SIGNED. aarch64 hardware does NOT trap: /0 yields 0,
//!                 INT_MIN / -1 yields INT_MIN. Zig `@divTrunc` TRAPS on both
//!                 in safe builds, so they are special-cased BEFORE it.
//!   neg         : `0 -% v` (so neg(INT_MIN) == INT_MIN, matching hardware).
//!   bnot        : logical "== 0" on a BOOL (cmp #0 + cset eq), i.e. `!b`.
//!   icmp        : SIGNED i64 comparison → bool.
//! `smod` is DEAD (CodegenIr treats it as unreachable) — never folded here.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");

pub const BinKind = enum { add, sub, mul, sdiv };

/// Fold a binary integer op on two constant i64 operands, matching aarch64.
pub fn foldBin(kind: BinKind, l: i64, r: i64) i64 {
    return switch (kind) {
        .add => l +% r,
        .sub => l -% r,
        .mul => l *% r,
        .sdiv => sdiv(l, r),
    };
}

/// SIGNED division matching aarch64 `sdiv`: /0 → 0, INT_MIN/-1 → INT_MIN, else
/// truncate-toward-zero. The two special cases MUST come before `@divTrunc`
/// (which traps on both in safe builds).
pub fn sdiv(l: i64, r: i64) i64 {
    if (r == 0) return 0;
    if (l == std.math.minInt(i64) and r == -1) return std.math.minInt(i64);
    return @divTrunc(l, r);
}

/// Unary negate matching aarch64 `sub xd, xzr, xm` (wrapping).
pub fn neg(v: i64) i64 {
    return 0 -% v;
}

/// Logical not on a bool (matches `cmp #0; cset eq`).
pub fn bnot(b: bool) bool {
    return !b;
}

/// SIGNED comparison → bool, matching `cmp` + `cset cc`.
pub fn icmp(cc: Ir.Cond, l: i64, r: i64) bool {
    return switch (cc) {
        .eq => l == r,
        .ne => l != r,
        .lt => l < r,
        .le => l <= r,
        .gt => l > r,
        .ge => l >= r,
    };
}

// ---------------------------------------------------------------------------
// Tests — the fold-divergence edge cases (run under std.testing.allocator).
// ---------------------------------------------------------------------------

test "sdiv by zero is 0 (no trap)" {
    try std.testing.expectEqual(@as(i64, 0), sdiv(5, 0));
    try std.testing.expectEqual(@as(i64, 0), sdiv(-7, 0));
    try std.testing.expectEqual(@as(i64, 0), sdiv(0, 0));
}

test "INT_MIN / -1 is INT_MIN (no overflow trap)" {
    const min = std.math.minInt(i64);
    try std.testing.expectEqual(min, sdiv(min, -1));
}

test "sdiv truncates toward zero" {
    try std.testing.expectEqual(@as(i64, -2), sdiv(-7, 3));
    try std.testing.expectEqual(@as(i64, 2), sdiv(7, 3));
    try std.testing.expectEqual(@as(i64, -2), sdiv(7, -3));
}

test "neg(INT_MIN) == INT_MIN" {
    const min = std.math.minInt(i64);
    try std.testing.expectEqual(min, neg(min));
    try std.testing.expectEqual(@as(i64, -5), neg(5));
}

test "wrapping add/sub/mul overflow" {
    const max = std.math.maxInt(i64);
    const min = std.math.minInt(i64);
    try std.testing.expectEqual(min, foldBin(.add, max, 1));
    try std.testing.expectEqual(max, foldBin(.sub, min, 1));
    try std.testing.expectEqual(@as(i64, 0), foldBin(.mul, min, 2)); // min*2 wraps to 0
}

test "signed icmp on negatives" {
    try std.testing.expect(icmp(.lt, -3, 0));
    try std.testing.expect(!icmp(.gt, -3, 0));
    try std.testing.expect(icmp(.le, -5, -5));
    try std.testing.expect(icmp(.ge, 2, -2));
    try std.testing.expect(icmp(.eq, -1, -1));
    try std.testing.expect(icmp(.ne, -1, 1));
}

test "bnot" {
    try std.testing.expect(bnot(false));
    try std.testing.expect(!bnot(true));
}
