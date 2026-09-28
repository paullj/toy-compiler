//! Exact constant-fold arithmetic for the opt stage.
//!
//! These helpers MUST replicate what `CodegenIr.genArith` + `Aarch64` emit
//! bit-for-bit, otherwise opt-on diverges from opt-off (the differential test
//! is the backstop, but correctness lives here). Verified against
//! `CodegenIr.genArith` + `Aarch64`:
//!   add/sub/mul : 64-bit two's-complement WRAPPING (`+%`/`-%`/`*%`).
//!   sdiv        : SIGNED. aarch64 hardware does NOT trap: /0 yields 0,
//!                 INT_MIN / -1 yields INT_MIN. Zig `@divTrunc` TRAPS on both
//!                 in safe builds, so they are special-cased BEFORE it.
//!   neg         : `0 -% v` (so neg(INT_MIN) == INT_MIN, matching hardware).
//!   bnot        : logical "== 0" on a BOOL (cmp #0 + cset eq), i.e. `!b`.
//!   icmp        : SIGNED i64 comparison → bool.

const std = @import("std");
const Ir = @import("../ir/Ir.zig");

pub const BinKind = enum { add, sub, mul, sdiv, udiv, smod, umod, band, bor, bxor };

/// Fold a binary integer op on two constant i64 operands, matching aarch64.
pub fn foldBin(kind: BinKind, l: i64, r: i64) i64 {
    return switch (kind) {
        .add => l +% r,
        .sub => l -% r,
        .mul => l *% r,
        .sdiv => sdiv(l, r),
        .udiv => udiv(l, r),
        .smod => smod(l, r),
        .umod => umod(l, r),
        .band => l & r,
        .bor => l | r,
        .bxor => l ^ r,
    };
}

pub const ShiftKind = enum { shl, lshr, ashr };

/// Go-semantics shift fold — the value-domain mirror of `CodegenIr.genShift`.
/// The amount is compared UNSIGNED against `width` (matching the `csel …,hs`
/// guard): amt>=width → 0 (shl/lshr) or sign-fill (ashr, = `l >> 63`). Below width
/// the amount is < 64, so a plain native shift in the u64/i64 domain. Returns the
/// raw 64-bit value; the caller applies `wrapTo` (≡ codegen's `normalizeWidth`).
pub fn foldShift(kind: ShiftKind, l: i64, r: i64, width: u16) i64 {
    if (@as(u64, @bitCast(r)) >= width) {
        return switch (kind) {
            .shl, .lshr => 0,
            .ashr => l >> 63,
        };
    }
    const amt: u6 = @intCast(r); // r < width <= 64 ⇒ r in [0,63]
    return switch (kind) {
        .shl => @bitCast(@as(u64, @bitCast(l)) << amt),
        .lshr => @bitCast(@as(u64, @bitCast(l)) >> amt),
        .ashr => l >> amt,
    };
}

/// Bitwise complement matching aarch64 `mvn` (full 64-bit `~`); the caller
/// width-normalizes via `wrapTo`.
pub fn bcompl(v: i64) i64 {
    return ~v;
}

/// SIGNED division matching aarch64 `sdiv`: /0 → 0, INT_MIN/-1 → INT_MIN, else
/// truncate-toward-zero. The two special cases MUST come before `@divTrunc`
/// (which traps on both in safe builds).
pub fn sdiv(l: i64, r: i64) i64 {
    if (r == 0) return 0;
    if (l == std.math.minInt(i64) and r == -1) return std.math.minInt(i64);
    return @divTrunc(l, r);
}

/// UNSIGNED division matching aarch64 `udiv`: /0 → 0, else unsigned truncate. No
/// INT_MIN/-1 case (that overflow is signed-only). Operands are the raw 64-bit bit
/// patterns reinterpreted as `u64`.
pub fn udiv(l: i64, r: i64) i64 {
    if (r == 0) return 0;
    return @bitCast(@as(u64, @bitCast(l)) / @as(u64, @bitCast(r)));
}

/// SIGNED remainder matching aarch64 `sdiv`+`msub` (r = a - (a/b)*b): sign of the
/// dividend, truncated. /0 -> 0 and INT_MIN%-1 -> 0 (the msub cancels:
/// INT_MIN -% (INT_MIN *% -1) == 0), both special-cased before `@rem` (which traps
/// on them). /0 is caller-gated (fold skips it), so this stays total like `sdiv`.
pub fn smod(l: i64, r: i64) i64 {
    if (r == 0) return 0;
    if (l == std.math.minInt(i64) and r == -1) return 0;
    return @rem(l, r);
}

/// UNSIGNED remainder matching aarch64 `udiv`+`msub`. /0 -> 0 (caller-gated).
/// Operands are the raw 64-bit bit patterns reinterpreted as `u64`.
pub fn umod(l: i64, r: i64) i64 {
    if (r == 0) return 0;
    return @bitCast(@as(u64, @bitCast(l)) % @as(u64, @bitCast(r)));
}

/// Unary negate matching aarch64 `sub xd, xzr, xm` (wrapping).
pub fn neg(v: i64) i64 {
    return 0 -% v;
}

/// Width-normalize a folded 64-bit result to its result type's canonical form —
/// the value-domain mirror of `CodegenIr.normalizeWidth` (signed narrow →
/// sign-extend, unsigned narrow → zero-extend). Exhaustive over `IntWidth` (no
/// `else`) so a new width class is a build break, not a silent fold divergence.
/// Identity for non-integers and platform/w64 ints → byte-identical for existing
/// programs.
pub fn wrapTo(ty: Ir.Type, x: i64) i64 {
    if (!ty.isInteger()) return x;
    return switch (ty.int_desc.width) {
        .plat, .w64 => x,
        .w8 => if (ty.int_desc.signed) @as(i64, @as(i8, @truncate(x))) else x & 0xFF,
        .w16 => if (ty.int_desc.signed) @as(i64, @as(i16, @truncate(x))) else x & 0xFFFF,
        .w32 => if (ty.int_desc.signed) @as(i64, @as(i32, @truncate(x))) else x & 0xFFFFFFFF,
    };
}

/// Logical not on a bool (matches `cmp #0; cset eq`).
pub fn bnot(b: bool) bool {
    return !b;
}

/// Comparison → bool, matching `cmp` + `cset cc`. Signedness rides on `cc` (the
/// `u*` conds compare the operands as `u64`), so this stays correct under
/// operand-rewriting opt passes.
pub fn icmp(cc: Ir.Cond, l: i64, r: i64) bool {
    return switch (cc) {
        .eq => l == r,
        .ne => l != r,
        .lt => l < r,
        .le => l <= r,
        .gt => l > r,
        .ge => l >= r,
        .ult => @as(u64, @bitCast(l)) < @as(u64, @bitCast(r)),
        .ule => @as(u64, @bitCast(l)) <= @as(u64, @bitCast(r)),
        .ugt => @as(u64, @bitCast(l)) > @as(u64, @bitCast(r)),
        .uge => @as(u64, @bitCast(l)) >= @as(u64, @bitCast(r)),
    };
}

// Tests — the fold-divergence edge cases (run under std.testing.allocator).

test "sdiv by zero is 0 (no trap)" {
    try std.testing.expectEqual(@as(i64, 0), sdiv(5, 0));
    try std.testing.expectEqual(@as(i64, 0), sdiv(-7, 0));
    try std.testing.expectEqual(@as(i64, 0), sdiv(0, 0));
}

test "INT_MIN / -1 is INT_MIN (no overflow trap)" {
    const min = std.math.minInt(i64);
    try std.testing.expectEqual(min, sdiv(min, -1));
}

test "smod: sign of the dividend, truncated (matches sdiv+msub)" {
    try std.testing.expectEqual(@as(i64, -2), smod(-17, 5));
    try std.testing.expectEqual(@as(i64, 2), smod(17, -5));
    try std.testing.expectEqual(@as(i64, 2), smod(17, 5));
    try std.testing.expectEqual(@as(i64, 0), smod(std.math.minInt(i64), -1)); // msub cancels
    try std.testing.expectEqual(@as(i64, 0), smod(5, 0)); // caller-gated; total fn
}

test "umod: unsigned remainder (raw bit patterns)" {
    try std.testing.expectEqual(@as(i64, 1), umod(7, 2));
    try std.testing.expectEqual(@as(i64, 0), umod(8, 2));
    // top-bit-set operand: as u64, 0x8000_0000_0000_0000 % 3 == 2.
    try std.testing.expectEqual(@as(i64, 2), umod(std.math.minInt(i64), 3));
    try std.testing.expectEqual(@as(i64, 0), umod(5, 0)); // caller-gated
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

test "udiv is unsigned: /0 is 0; 2^63/2 == 2^62 (bit63 set)" {
    try std.testing.expectEqual(@as(i64, 0), udiv(5, 0));
    const two_63 = std.math.minInt(i64); // 0x8000_0000_0000_0000 bit pattern
    try std.testing.expectEqual(@as(i64, 0x4000000000000000), udiv(two_63, 2));
    try std.testing.expectEqual(@as(i64, 3), udiv(7, 2));
}

test "wrapTo normalizes narrow widths (mirrors codegen)" {
    // int8 200 (as -56 after wrap) and -56 both canonicalize to -56.
    try std.testing.expectEqual(@as(i64, -56), wrapTo(Ir.Type.int8, 200));
    try std.testing.expectEqual(@as(i64, 44), wrapTo(Ir.Type.uint8, 300));
    try std.testing.expectEqual(@as(i64, -1), wrapTo(Ir.Type.int16, 0xFFFF));
    try std.testing.expectEqual(@as(i64, 0xFFFF), wrapTo(Ir.Type.uint16, 0xFFFF));
    try std.testing.expectEqual(@as(i64, -1), wrapTo(Ir.Type.int32, 0xFFFFFFFF));
    try std.testing.expectEqual(@as(i64, 0xFFFFFFFF), wrapTo(Ir.Type.uint32, -1));
    // plat / w64 / bool are identity.
    try std.testing.expectEqual(@as(i64, 200), wrapTo(Ir.Type.int, 200));
    try std.testing.expectEqual(@as(i64, -56), wrapTo(Ir.Type.int64, -56));
    try std.testing.expectEqual(@as(i64, 5), wrapTo(Ir.Type.bool, 5));
}

test "unsigned icmp contrasts signed on bit63-set operands" {
    const min = std.math.minInt(i64); // top-bit-set: huge unsigned, negative signed
    try std.testing.expect(icmp(.ugt, min, 1)); // unsigned: 2^63 > 1
    try std.testing.expect(!icmp(.gt, min, 1)); // signed: -2^63 > 1 is false
    try std.testing.expect(icmp(.ult, 1, min));
    try std.testing.expect(icmp(.uge, min, min));
    try std.testing.expect(icmp(.ule, 1, min));
}

test "foldShift Go semantics — below width, signed vs unsigned, and >= width" {
    // Below width: logical vs arithmetic right shift on a bit7-set value.
    try std.testing.expectEqual(@as(i64, 64), foldShift(.lshr, 0x80, 1, 8));
    try std.testing.expectEqual(@as(i64, -64), foldShift(.ashr, -128, 1, 8));
    // bit63-set value: lshr != ashr.
    const bit63 = std.math.minInt(i64);
    try std.testing.expectEqual(@as(i64, 0x4000000000000000), foldShift(.lshr, bit63, 1, 64));
    try std.testing.expectEqual(@as(i64, @bitCast(@as(u64, 0xC000000000000000))), foldShift(.ashr, bit63, 1, 64));
    // amt >= width: shl/lshr -> 0, ashr -> sign-fill.
    try std.testing.expectEqual(@as(i64, 0), foldShift(.lshr, 0xFF, 8, 8));
    try std.testing.expectEqual(@as(i64, 0), foldShift(.shl, 0xFF, 8, 8));
    try std.testing.expectEqual(@as(i64, -1), foldShift(.ashr, -1, 8, 8));
    try std.testing.expectEqual(@as(i64, 0), foldShift(.ashr, 5, 8, 8));
    // amt >= 64 (the mask-wrap region — the guard's raison d'être).
    try std.testing.expectEqual(@as(i64, 0), foldShift(.lshr, 0xFF, 64, 8));
    try std.testing.expectEqual(@as(i64, 0), foldShift(.shl, 0xFF, 64, 8));
    try std.testing.expectEqual(@as(i64, 0), foldShift(.lshr, -1, 64, 64));
    try std.testing.expectEqual(@as(i64, -1), foldShift(.ashr, -1, 64, 64));
    // negative amount (huge unsigned) -> >= width.
    try std.testing.expectEqual(@as(i64, 0), foldShift(.lshr, 0xFF, -1, 8));
    try std.testing.expectEqual(@as(i64, -1), foldShift(.ashr, -1, -1, 64));
}

test "foldBin bitwise and bcompl, then narrow wrapTo" {
    try std.testing.expectEqual(@as(i64, 0x0A), foldBin(.band, 0xAA, 0x0F));
    try std.testing.expectEqual(@as(i64, 0xAF), foldBin(.bor, 0xAA, 0x0F));
    try std.testing.expectEqual(@as(i64, 0xA5), foldBin(.bxor, 0xAA, 0x0F));
    try std.testing.expectEqual(~@as(i64, 48), bcompl(48));
    try std.testing.expectEqual(@as(i64, 207), wrapTo(Ir.Type.uint8, bcompl(48))); // demo's ~c
    try std.testing.expectEqual(@as(i64, 65280), wrapTo(Ir.Type.uint16, bcompl(0x00FF)));
}
