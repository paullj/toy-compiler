//! Pure aarch64 (AArch64 / ARM64) instruction-word emitter.
//!
//! WHY: all the bit-twiddling needed to turn a mnemonic into a 32-bit machine
//! word lives here, behind named functions, so `Codegen` and the unit tests
//! never hand-encode an instruction. Every helper is a pure function (no IO, no
//! allocation) returning the little-endian-on-disk `u32` word; callers serialize
//! with `std.mem.writeInt(u32, buf, word, .little)`.
//!
//! Each expected word in the tests below was cross-checked against the real
//! assembler on this host: write the mnemonics to a `.s` file, `as -arch arm64`,
//! then `objdump -d` — the disassembly's hex column is the ground truth the
//! tests assert (see the comments next to each constant/function).
//!
//! Scope is the initial subset only: movz/movk (+ an i64 materializer), reg/reg moves,
//! add/sub (reg and imm12), mul, sdiv, neg, the frame stp/ldp pair, ldr/str
//! unsigned-offset against sp, sp add/sub, bl, and ret.

const std = @import("std");

pub const XZR: u32 = 31;
pub const SP: u32 = 31;
pub const FP: u32 = 29;
pub const LR: u32 = 30;

/// movz rd, #imm16, lsl #(hw*16) — load a 16-bit immediate into a lane, zeroing
/// the rest. `hw` selects the lane (0..3). movz x0,#40 → 0xD2800500.
pub fn movz(rd: u32, imm16: u16, hw: u2) u32 {
    return 0xD2800000 | (@as(u32, hw) << 21) | (@as(u32, imm16) << 5) | rd;
}

/// movk rd, #imm16, lsl #(hw*16) — overwrite a 16-bit lane, keeping the others.
/// movk x0,#0x1234,lsl#16 → 0xF2A24680.
pub fn movk(rd: u32, imm16: u16, hw: u2) u32 {
    return 0xF2800000 | (@as(u32, hw) << 21) | (@as(u32, imm16) << 5) | rd;
}

/// add rd, rn, #imm12. With rd/rn = SP this is the sp adjust. add sp,sp,#16 →
/// 0x910043FF.
pub fn addImm(rd: u32, rn: u32, imm12: u12) u32 {
    return 0x91000000 | (@as(u32, imm12) << 10) | (rn << 5) | rd;
}

/// sub rd, rn, #imm12. sub sp,sp,#16 → 0xD10043FF.
pub fn subImm(rd: u32, rn: u32, imm12: u12) u32 {
    return 0xD1000000 | (@as(u32, imm12) << 10) | (rn << 5) | rd;
}

/// add rd, rn, rm. add x0,x0,x1 → 0x8B010000.
pub fn addReg(rd: u32, rn: u32, rm: u32) u32 {
    return 0x8B000000 | (rm << 16) | (rn << 5) | rd;
}

/// sub rd, rn, rm. sub x5,x0,x1 → 0xCB010005.
pub fn subReg(rd: u32, rn: u32, rm: u32) u32 {
    return 0xCB000000 | (rm << 16) | (rn << 5) | rd;
}

/// mul rd, rn, rm (alias of madd with the accumulator = XZR). mul x2,x0,x1 →
/// 0x9B017C02.
pub fn mul(rd: u32, rn: u32, rm: u32) u32 {
    return 0x9B007C00 | (rm << 16) | (rn << 5) | rd;
}

/// msub rd, rn, rm, ra -> rd = ra - rn*rm (MADD family, bit15 set). Used by smod/
/// umod for r = a - (a/b)*b. Words verified on-host (`as -arch arm64` + `otool`):
/// msub x0,x1,x2,x3 -> 0x9B028C20, msub x3,x0,x1,x2 -> 0x9B018803.
pub fn msub(rd: u32, rn: u32, rm: u32, ra: u32) u32 {
    return 0x9B008000 | (rm << 16) | (ra << 10) | (rn << 5) | rd;
}

/// sdiv rd, rn, rm (signed division; arm64 sdiv by 0 yields 0, no trap). sdiv
/// x3,x0,x1 → 0x9AC10C03.
pub fn sdiv(rd: u32, rn: u32, rm: u32) u32 {
    return 0x9AC00C00 | (rm << 16) | (rn << 5) | rd;
}

/// udiv rd, rn, rm (unsigned division; arm64 udiv by 0 yields 0, no trap). Mirrors
/// `sdiv` with bit10 cleared. udiv x0,x0,x0 → 0x9AC00800.
pub fn udiv(rd: u32, rn: u32, rm: u32) u32 {
    return 0x9AC00800 | (rm << 16) | (rn << 5) | rd;
}

/// neg rd, rm — alias of `sub rd, xzr, rm`. neg x4,x0 → 0xCB0003E4.
pub fn neg(rd: u32, rm: u32) u32 {
    return subReg(rd, XZR, rm);
}

/// sxtb rd, rn — sign-extend low byte to 64-bit (SBFM alias). sxtb x0,x0 →
/// 0x93401C00. Used to re-canonicalize a signed narrow (int8) result after a
/// full-width op wraps it.
pub fn sxtb(rd: u32, rn: u32) u32 {
    return 0x93401C00 | (rn << 5) | rd;
}

/// sxth rd, rn — sign-extend low halfword to 64-bit. sxth x0,x0 → 0x93403C00.
pub fn sxth(rd: u32, rn: u32) u32 {
    return 0x93403C00 | (rn << 5) | rd;
}

/// sxtw rd, rn — sign-extend low word to 64-bit. sxtw x0,x0 → 0x93407C00.
pub fn sxtw(rd: u32, rn: u32) u32 {
    return 0x93407C00 | (rn << 5) | rd;
}

/// and rd, rn, #mask — AND with a low-bits bitmask immediate (N=1, immr=0; `imms`
/// = mask width − 1). imms 7/15/31 encode #0xff / #0xffff / #0xffffffff, the
/// zero-extend masks for an unsigned narrow (uint8/16/32) result. and x0,x0,#0xff
/// → 0x92401C00; and x0,x0,#0xffff → 0x92403C00; and x0,x0,#0xffffffff → 0x92407C00.
pub fn andLowBits(rd: u32, rn: u32, imms: u6) u32 {
    return 0x92400000 | (@as(u32, imms) << 10) | (rn << 5) | rd;
}

/// and rd, rn, rm — bitwise AND (logical-shifted-reg, LSL#0). and x0,x0,x1 → 0x8A010000.
pub fn andReg(rd: u32, rn: u32, rm: u32) u32 {
    return 0x8A000000 | (rm << 16) | (rn << 5) | rd;
}

/// orr rd, rn, rm — bitwise OR. orr x0,x0,x1 → 0xAA010000.
pub fn orrReg(rd: u32, rn: u32, rm: u32) u32 {
    return 0xAA000000 | (rm << 16) | (rn << 5) | rd;
}

/// eor rd, rn, rm — bitwise XOR. eor x0,x0,x1 → 0xCA010000.
pub fn eorReg(rd: u32, rn: u32, rm: u32) u32 {
    return 0xCA000000 | (rm << 16) | (rn << 5) | rd;
}

/// orn rd, rn, rm — OR-NOT (ORR with N=1). orn x0,xzr,x1 → 0xAA2103E0.
pub fn orn(rd: u32, rn: u32, rm: u32) u32 {
    return 0xAA200000 | (rm << 16) | (rn << 5) | rd;
}

/// mvn rd, rm — bitwise complement, alias of `orn rd, xzr, rm` (mirrors `neg`=`sub
/// rd,xzr,rm`). mvn x0,x0 → 0xAA2003E0.
pub fn mvn(rd: u32, rm: u32) u32 {
    return orn(rd, XZR, rm);
}

/// lslv rd, rn, rm — logical shift left by register (masks amt mod 64). lslv
/// x0,x0,x1 → 0x9AC12000.
pub fn lslv(rd: u32, rn: u32, rm: u32) u32 {
    return 0x9AC02000 | (rm << 16) | (rn << 5) | rd;
}

/// lsrv rd, rn, rm — logical shift right by register. lsrv x0,x0,x1 → 0x9AC12400.
pub fn lsrv(rd: u32, rn: u32, rm: u32) u32 {
    return 0x9AC02400 | (rm << 16) | (rn << 5) | rd;
}

/// asrv rd, rn, rm — arithmetic shift right by register. asrv x0,x0,x1 → 0x9AC12800.
pub fn asrv(rd: u32, rn: u32, rm: u32) u32 {
    return 0x9AC02800 | (rm << 16) | (rn << 5) | rd;
}

/// asr rd, rn, #shift — arithmetic shift right by immediate (SBFM Xd,Xn,#shift,#63).
/// Used with #63 to broadcast the sign bit for the Go ashr amt>=width fill.
/// asr x0,x0,#63 → 0x937FFC00.
pub fn asrImm(rd: u32, rn: u32, shift: u6) u32 {
    return 0x93400000 | (@as(u32, shift) << 16) | (@as(u32, 63) << 10) | (rn << 5) | rd;
}

/// csel rd, rn, rm, cond — rd = cond ? rn : rm; cond in bits[15:12].
/// csel x0,xzr,x0,hs → 0x9A8023E0.
pub fn csel(rd: u32, rn: u32, rm: u32, c: Cond) u32 {
    return 0x9A800000 | (rm << 16) | (@as(u32, @intFromEnum(c)) << 12) | (rn << 5) | rd;
}

/// str rt, [sp, #byteOff] — 64-bit store; byteOff must be a multiple of 8 (the
/// encoded imm12 is the scaled word index). str x0,[sp,#8] → 0xF90007E0.
pub fn strSp(rt: u32, byteOff: u32) u32 {
    std.debug.assert(byteOff % 8 == 0);
    const scaled: u32 = byteOff / 8;
    return 0xF9000000 | (scaled << 10) | (SP << 5) | rt;
}

/// ldr rt, [sp, #byteOff] — 64-bit load; byteOff multiple of 8. ldr x1,[sp,#8]
/// → 0xF94007E1.
pub fn ldrSp(rt: u32, byteOff: u32) u32 {
    std.debug.assert(byteOff % 8 == 0);
    const scaled: u32 = byteOff / 8;
    return 0xF9400000 | (scaled << 10) | (SP << 5) | rt;
}

/// ldr rt, [x29, #byteOff] — 64-bit load; base is FP(29). byteOff multiple of 8.
/// Used to read incoming stack arguments (the (8+k)th param at [x29,#16+k*8]).
/// ldr x9,[x29,#16] → 0xF9400BA9.
pub fn ldrFp(rt: u32, byteOff: u32) u32 {
    std.debug.assert(byteOff % 8 == 0);
    const scaled: u32 = byteOff / 8;
    return 0xF9400000 | (scaled << 10) | (FP << 5) | rt;
}

/// bl #(imm26 words) — branch-and-link, PC-relative by a signed *word* offset
/// (the raw imm26 field). The linker patches the placeholder `bl #0` with
/// imm26 = (target_off − site_off) / 4 for each `.call26` relocation.
/// bl #0 → 0x94000000.
pub fn bl(imm26: i26) u32 {
    const bits: u32 = @as(u26, @bitCast(imm26));
    return 0x94000000 | bits;
}

// These four encoders are how codegen reaches data the linker only sizes at the very
// end: a string in `__cstring` and the `_write` slot in `__got`. `adrp` forms a
// 4 KiB-page-relative base, `addImm` adds the in-page byte offset (for a cstring
// literal address), `ldrRegUoff` loads the GOT slot, and `blr` calls it. The
// `page_delta`/offset placeholders are 0 at emit time; the post-vmaddr pass
// (`Link.applyDataRelocs`, via the `patch*` helpers below) rewrites them once
// MachO assigns the final segment vmaddrs. Each word was assembler-verified on
// this host (see the tests).

/// adrp rd, #(page_delta pages) — form the address of a 4 KiB *page* PC-relative
/// by a signed 21-bit page count (units of 4096, NOT 0x4000). The placeholder
/// emit uses page_delta = 0. adrp x8,0 → 0x90000008; adrp x16,4 → 0x90000030.
pub fn adrp(rd: u32, page_delta: i21) u32 {
    const u: u21 = @bitCast(page_delta);
    return 0x90000000 | ((@as(u32, u) & 3) << 29) | (((@as(u32, u) >> 2) & 0x7FFFF) << 5) | rd;
}

/// ldr rt, [rn, #byteOff] — 64-bit load, unsigned scaled offset, arbitrary base
/// register `rn`. byteOff must be a multiple of 8. ldr x16,[x16] → 0xF9400210;
/// ldr x16,[x16,#0x18] → 0xF9400E10.
pub fn ldrRegUoff(rt: u32, rn: u32, byteOff: u32) u32 {
    std.debug.assert(byteOff % 8 == 0);
    const scaled: u32 = byteOff / 8;
    return 0xF9400000 | (scaled << 10) | (rn << 5) | rt;
}

/// ldrb wt, [rn, #byteOff] — zero-extended single-byte load, unsigned offset (the
/// imm12 is UNSCALED for a byte access, so `byteOff` is a raw byte displacement,
/// unlike `ldrRegUoff`'s /8 scaling). ldrb w0,[x0] → 0x39400000; ldrb w0,[x0,#1] →
/// 0x39400400. The str byte-compare reads `ptr[i]` with this.
pub fn ldrbRegUoff(rt: u32, rn: u32, byteOff: u12) u32 {
    return 0x39400000 | (@as(u32, byteOff) << 10) | (rn << 5) | rt;
}

/// str rt, [rn, #byteOff] — 64-bit store, unsigned scaled offset, arbitrary base
/// register `rn`. byteOff must be a multiple of 8. str x0,[x8] → 0xF9000100;
/// str x1,[x8,#8] → 0xF9000501. Symmetric to `ldrRegUoff`; used for the
/// indirect/x8 sret store-through-pointer and struct byte-copies.
pub fn strRegUoff(rt: u32, rn: u32, byteOff: u32) u32 {
    std.debug.assert(byteOff % 8 == 0);
    const scaled: u32 = byteOff / 8;
    return 0xF9000000 | (scaled << 10) | (rn << 5) | rt;
}

/// blr rn — indirect branch-and-link through a register. blr x16 → 0xD63F0200.
pub fn blr(rn: u32) u32 {
    return 0xD63F0000 | (rn << 5);
}

/// strb wt, [rn, #byteOff] — single-byte store, unsigned offset (the imm12 is UNSCALED
/// for a byte access, mirroring `ldrbRegUoff`). strb w13,[x10] → 0x3900014D;
/// strb w0,[x0,#1] → 0x39000400. The int-renderer writes each ASCII digit this way.
pub fn strb(rt: u32, rn: u32, byteOff: u12) u32 {
    return 0x39000000 | (@as(u32, byteOff) << 10) | (rn << 5) | rt;
}

/// cbz rt, #(imm19 words) — branch if rt == 0, PC-relative signed word offset. The
/// renderer never uses it directly but it completes the cbz/cbnz pair; symmetric to
/// `cbnz`. cbz x0,#0 → 0xB4000000; cbz x0,#-4 → 0xB4FFFF80.
pub fn cbz(rt: u32, imm19: i19) u32 {
    return 0xB4000000 | (@as(u32, @as(u19, @bitCast(imm19))) << 5) | rt;
}

/// b.cond #(imm19 words) — conditional branch, PC-relative signed word offset in
/// bits[23:5], cond in bits[3:0]. Emits the RESOLVED word directly (compile-time-known
/// local offset), unlike the `#0`-placeholder + `patchBCond` codegen path. b.ge #+2 →
/// 0x5400004A; b.ge #+4 → 0x5400008A.
pub fn bCond(cond: Cond, imm19: i19) u32 {
    return 0x54000000 | (@as(u32, @as(u19, @bitCast(imm19))) << 5) | @as(u32, @intFromEnum(cond));
}

/// mov rd, rm — register move (alias of `orr rd, xzr, rm`). mov x2,x1 →
/// 0xAA0103E2; mov x1,x0 → 0xAA0003E1.
pub fn movReg(rd: u32, rm: u32) u32 {
    return 0xAA0003E0 | (rm << 16) | rd;
}

// The placeholder words carry the destination register(s) in their low fields;
// these recover those and re-encode with the resolved page delta / offset, so
// the patch needs no separate record of which register the emitter chose.

/// Re-encode an `adrp` placeholder with the resolved page delta, keeping its rd.
pub fn patchAdrp(word: u32, page_delta: i21) u32 {
    const rd: u32 = word & 0x1F;
    return adrp(rd, page_delta);
}

/// Re-encode an `add (imm12)` placeholder with the resolved low-12 offset,
/// keeping its rd/rn.
pub fn patchAddImm12(word: u32, imm12: u12) u32 {
    const rd: u32 = word & 0x1F;
    const rn: u32 = (word >> 5) & 0x1F;
    return addImm(rd, rn, imm12);
}

/// Re-encode an `ldr (unsigned-offset)` placeholder with the resolved byte
/// offset, keeping its rt/rn.
pub fn patchLdrUoff(word: u32, byteOff: u32) u32 {
    const rt: u32 = word & 0x1F;
    const rn: u32 = (word >> 5) & 0x1F;
    return ldrRegUoff(rt, rn, byteOff);
}

// WHY: codegen needs to test values and jump. Comparisons set NZCV via SUBS-to-XZR
// (`cmp`), a condition turns NZCV into 0/1 (`cset`) in VALUE context or steers a
// `b.cond`/`cbz`/`cbnz` in CONTROL context, and `b` is the unconditional jump.
// Branch targets are intra-function byte offsets resolved by Codegen backpatch
// (NOT relocations): the imm field is (target_byte − site_byte)/4, two's
// complement, so a `#0` placeholder is emitted then rewritten by the patchers.
// Every word below was assembler-verified on this host (see the tests).

/// AArch64 condition codes. Equality (eq/ne), the SIGNED magnitude comparisons
/// (ge/lt/gt/le), and — since codegen dispatches unsigned integer compares by operand
/// signedness — the UNSIGNED magnitude comparisons (hs/lo/hi/ls). Values are the
/// 4-bit cond field encoding (e.g. b.cond carries cond in bits[3:0]).
pub const Cond = enum(u4) {
    eq = 0x0, // equal (Z==1)
    ne = 0x1, // not equal (Z==0)
    hs = 0x2, // unsigned >=
    lo = 0x3, // unsigned <
    mi = 0x4, // N==1 (float <: unordered→false, NOT `lt` which is unordered→true)
    pl = 0x5, // N==0 (the inverse of `mi`)
    hi = 0x8, // unsigned >
    ls = 0x9, // unsigned <=
    ge = 0xA, // signed >=
    lt = 0xB, // signed <
    gt = 0xC, // signed >
    le = 0xD, // signed <=
};

/// Logical negation of a condition (swaps the branch sense). Used by genCond to
/// branch on the inverse, and by `cset` which physically encodes the inverted
/// condition (CSINC reads the inverse).
pub fn invert(c: Cond) Cond {
    return switch (c) {
        .eq => .ne,
        .ne => .eq,
        .lt => .ge,
        .ge => .lt,
        .gt => .le,
        .le => .gt,
        .lo => .hs,
        .hs => .lo,
        .hi => .ls,
        .ls => .hi,
        .mi => .pl,
        .pl => .mi,
    };
}

/// cmp rn, rm — alias of SUBS XZR, Xn, Xm (subtract, set flags, discard result).
/// cmp x1,x0 → 0xEB00003F; cmp x0,x1 → 0xEB01001F.
pub fn cmpReg(rn: u32, rm: u32) u32 {
    return 0xEB000000 | (rm << 16) | (rn << 5) | XZR;
}

/// cmp rn, #imm12 — alias of SUBS XZR, Xn, #imm12 (unshifted). cmp x0,#5 →
/// 0xF100141F; cmp x0,#0 → 0xF100001F; cmp x1,#0 → 0xF100003F.
pub fn cmpImm(rn: u32, imm12: u12) u32 {
    return 0xF1000000 | (@as(u32, imm12) << 10) | (rn << 5) | XZR;
}

/// cset rd, cond — alias of CSINC Xd, XZR, XZR, invert(cond): set rd to 1 if the
/// condition holds, else 0. The encoding stores the INVERTED condition in
/// bits[15:12] (CSINC increments the false-source when the inverse holds).
/// cset x0,eq → 0x9A9F17E0; cset x0,lt → 0x9A9FA7E0; cset x2,lt → 0x9A9FA7E2.
pub fn cset(rd: u32, c: Cond) u32 {
    return 0x9A9F07E0 | (@as(u32, @intFromEnum(invert(c))) << 12) | rd;
}

/// cbnz rt, #(imm19 words) — branch if rt != 0. cbnz x0,#0 → 0xB5000000;
/// cbnz x0,#-4 → 0xB5FFFF80.
pub fn cbnz(rt: u32, imm19: i19) u32 {
    return 0xB5000000 | (@as(u32, @as(u19, @bitCast(imm19))) << 5) | rt;
}

/// b #(imm26 words) — unconditional branch, PC-relative signed word offset in
/// bits[25:0]. b #0 → 0x14000000; b #+2 → 0x14000002; b #-2 → 0x17FFFFFE.
pub fn b(imm26: i26) u32 {
    return 0x14000000 | @as(u32, @as(u26, @bitCast(imm26)));
}

// Emit a branch with a #0 placeholder, record its site + target label, then
// rewrite only the imm field once the label's byte offset is known. The opcode
// + identity (cond for b.cond, rt for cbz/cbnz) is preserved. For b.cond and
// cbz/cbnz the mask 0xFF00001F keeps the top byte (0x54/0xB4/0xB5) and the low 5
// bits (cond ⊂ low4, or rt = low5) while clearing the imm19 field at bits[23:5].

/// Rewrite a b.cond placeholder's imm19, keeping opcode + cond. Round-trips:
/// patchBCond(0x54000000, 2) → 0x54000040.
pub fn patchBCond(word: u32, imm19: i19) u32 {
    return (word & 0xFF00001F) | (@as(u32, @as(u19, @bitCast(imm19))) << 5);
}

/// Rewrite a cbz/cbnz placeholder's imm19, keeping opcode + rt. Round-trips:
/// patchCbz(0xB4000000, -4) → 0xB4FFFF80.
pub fn patchCbz(word: u32, imm19: i19) u32 {
    return (word & 0xFF00001F) | (@as(u32, @as(u19, @bitCast(imm19))) << 5);
}

/// Rewrite a `b` placeholder's imm26 (full opcode is fixed). Round-trips:
/// patchB(b(0), -2) → 0x17FFFFFE.
pub fn patchB(word: u32, imm26: i26) u32 {
    _ = word;
    return b(imm26);
}

/// stp x29, x30, [sp, #-16]! — pre-index store of the FP/LR pair; opens the
/// frame in one instruction. → 0xA9BF7BFD.
pub const stpFpLrPre: u32 = 0xA9BF7BFD;

/// mov x29, sp (add x29, sp, #0) — set the frame pointer. → 0x910003FD.
pub const movFpSp: u32 = 0x910003FD;

/// ldp x29, x30, [sp], #16 — post-index load of the FP/LR pair; closes the
/// frame. → 0xA8C17BFD.
pub const ldpFpLrPost: u32 = 0xA8C17BFD;

/// ret (returns to x30). → 0xD65F03C0.
pub const ret: u32 = 0xD65F03C0;

/// brk #0 — a software breakpoint that aborts with SIGILL. The single instruction
/// emitted for the `.trap` IR terminator (`Option`/`Result` `unwrap`'s failure arm); no
/// other trap primitive exists on this backend. → 0xD4200000.
pub const brk0: u32 = 0xD4200000;

/// Materialize an arbitrary i64 `value` into register `rd` using the minimal
/// movz + movk sequence, writing the words (little-endian) into `out` starting
/// at `*len` and advancing it. At most 4 words. value 0 → a single
/// `movz rd, #0`. Negative values work via their two's-complement lanes (e.g.
/// -1 = 0xFFFF_FFFF_FFFF_FFFF → movz #0xffff then movk #0xffff in lanes 1..3).
pub fn movImm64(out: []u8, len: *usize, rd: u32, value: i64) void {
    const u: u64 = @bitCast(value);

    // First lane via movz (zeroes the upper lanes), then movk the rest.
    const lane0: u16 = @truncate(u);
    writeWord(out, len, movz(rd, lane0, 0));

    var hw: u2 = 1;
    while (true) : (hw += 1) {
        const lane: u16 = @truncate(u >> (@as(u6, hw) * 16));
        if (lane != 0) writeWord(out, len, movk(rd, lane, hw));
        if (hw == 3) break;
    }
}

inline fn writeWord(out: []u8, len: *usize, word: u32) void {
    std.mem.writeInt(u32, out[len.*..][0..4], word, .little);
    len.* += 4;
}

// Scalar double-precision FP. D-registers are the SIMD&FP file (0..31); codegen
// uses D16/D17 (caller-saved scratch). fadd/fsub/fmul/fdiv/fcmp are the
// double-precision floating-point data-processing forms (type field = 01). Each
// word assembler-verified on this host (`as -arch arm64` + `objdump -d`), same
// protocol as the integer encoders above.

/// fadd dd, dn, dm — double add. fadd d16,d16,d17 → 0x1E712A10.
pub fn fadd(dd: u32, dn: u32, dm: u32) u32 {
    return 0x1E602800 | (dm << 16) | (dn << 5) | dd;
}

/// fsub dd, dn, dm — double subtract. fsub d16,d16,d17 → 0x1E713A10.
pub fn fsub(dd: u32, dn: u32, dm: u32) u32 {
    return 0x1E603800 | (dm << 16) | (dn << 5) | dd;
}

/// fmul dd, dn, dm — double multiply. fmul d16,d16,d17 → 0x1E710A10.
pub fn fmul(dd: u32, dn: u32, dm: u32) u32 {
    return 0x1E600800 | (dm << 16) | (dn << 5) | dd;
}

/// fdiv dd, dn, dm — double divide (IEEE: /0 → ±inf/NaN, no trap). fdiv
/// d16,d16,d17 → 0x1E711A10.
pub fn fdiv(dd: u32, dn: u32, dm: u32) u32 {
    return 0x1E601800 | (dm << 16) | (dn << 5) | dd;
}

/// fcmp dn, dm — double compare, sets NZCV (unordered → C=1,V=1). fcmp d16,d17 →
/// 0x1E712200.
pub fn fcmp(dn: u32, dm: u32) u32 {
    return 0x1E602000 | (dm << 16) | (dn << 5);
}

/// ldr dt, [sp, #byteOff] — 64-bit FP load; byteOff a multiple of 8 (scaled
/// imm12). ldr d16,[sp,#8] → 0xFD4007F0.
pub fn ldrFpSp(dt: u32, byteOff: u32) u32 {
    std.debug.assert(byteOff % 8 == 0);
    return 0xFD400000 | ((byteOff / 8) << 10) | (SP << 5) | dt;
}

/// str dt, [sp, #byteOff] — 64-bit FP store; byteOff a multiple of 8. str
/// d16,[sp,#8] → 0xFD0007F0.
pub fn strFpSp(dt: u32, byteOff: u32) u32 {
    std.debug.assert(byteOff % 8 == 0);
    return 0xFD000000 | ((byteOff / 8) << 10) | (SP << 5) | dt;
}

// Tests — each expected word is the objdump hex for the matching mnemonic,
// assembled on this host with `as -arch arm64` (see the file doc comment).

const testing = std.testing;

test "fp encoders" {
    // Each expected word is the objdump hex for the matching mnemonic assembled on
    // this host: write to `/tmp/x.s`, `as -arch arm64 -o /tmp/x.o /tmp/x.s`,
    // `objdump -d /tmp/x.o` — the disassembly's hex column is the ground truth.
    try testing.expectEqual(@as(u32, 0x1E712A10), fadd(16, 16, 17)); // fadd d16,d16,d17
    try testing.expectEqual(@as(u32, 0x1E713A10), fsub(16, 16, 17)); // fsub d16,d16,d17
    try testing.expectEqual(@as(u32, 0x1E710A10), fmul(16, 16, 17)); // fmul d16,d16,d17
    try testing.expectEqual(@as(u32, 0x1E711A10), fdiv(16, 16, 17)); // fdiv d16,d16,d17
    try testing.expectEqual(@as(u32, 0x1E712200), fcmp(16, 17)); // fcmp d16,d17
    try testing.expectEqual(@as(u32, 0xFD4007F0), ldrFpSp(16, 8)); // ldr d16,[sp,#8]
    try testing.expectEqual(@as(u32, 0xFD0007F0), strFpSp(16, 8)); // str d16,[sp,#8]
    try testing.expectEqual(@as(u32, 0x9A9F57E0), cset(0, .mi)); // cset x0,mi
    try testing.expectEqual(@as(u32, 0x9A9F47E0), cset(0, .pl)); // cset x0,pl
    try testing.expectEqual(Cond.pl, invert(.mi));
    try testing.expectEqual(Cond.mi, invert(.pl));
}

test "move-wide immediates" {
    try testing.expectEqual(@as(u32, 0xD2800500), movz(0, 40, 0)); // movz x0,#40
    try testing.expectEqual(@as(u32, 0xD2800030), movz(16, 1, 0)); // movz x16,#1
    try testing.expectEqual(@as(u32, 0xF2A24680), movk(0, 0x1234, 1)); // movk x0,#0x1234,lsl#16
    try testing.expectEqual(@as(u32, 0xF2BFFFE0), movk(0, 0xFFFF, 1));
    try testing.expectEqual(@as(u32, 0xF2DFFFE0), movk(0, 0xFFFF, 2));
    try testing.expectEqual(@as(u32, 0xF2FFFFE0), movk(0, 0xFFFF, 3));
}

test "add/sub immediate" {
    try testing.expectEqual(@as(u32, 0x910043FF), addImm(SP, SP, 16)); // add sp,sp,#16
    try testing.expectEqual(@as(u32, 0xD10043FF), subImm(SP, SP, 16)); // sub sp,sp,#16
}

test "data-processing register" {
    try testing.expectEqual(@as(u32, 0x8B010000), addReg(0, 0, 1)); // add x0,x0,x1
    try testing.expectEqual(@as(u32, 0xCB010005), subReg(5, 0, 1)); // sub x5,x0,x1
    try testing.expectEqual(@as(u32, 0xCB010000), subReg(0, 0, 1)); // sub x0,x0,x1
    try testing.expectEqual(@as(u32, 0x9B017C02), mul(2, 0, 1)); // mul x2,x0,x1
    try testing.expectEqual(@as(u32, 0x9B017C00), mul(0, 0, 1)); // mul x0,x0,x1
    try testing.expectEqual(@as(u32, 0x9AC10C03), sdiv(3, 0, 1)); // sdiv x3,x0,x1
    try testing.expectEqual(@as(u32, 0x9AC10C00), sdiv(0, 0, 1)); // sdiv x0,x0,x1
    try testing.expectEqual(@as(u32, 0xCB0003E4), neg(4, 0)); // neg x4,x0
    try testing.expectEqual(@as(u32, 0xCB0003E0), neg(0, 0)); // neg x0,x0
    try testing.expectEqual(@as(u32, 0x9B028C20), msub(0, 1, 2, 3)); // msub x0,x1,x2,x3
    try testing.expectEqual(@as(u32, 0x9B018803), msub(3, 0, 1, 2)); // msub x3,x0,x1,x2
    try testing.expectEqual(@as(u32, 0x9B018040), msub(0, 2, 1, 0)); // msub x0,x2,x1,x0 (the genRem shape)
}

test "width-correct encoders (udiv + sign-extend + low-bits mask)" {
    try testing.expectEqual(@as(u32, 0x9AC00800), udiv(0, 0, 0)); // udiv x0,x0,x0
    try testing.expectEqual(@as(u32, 0x9AC10803), udiv(3, 0, 1)); // udiv x3,x0,x1
    try testing.expectEqual(@as(u32, 0x93401C00), sxtb(0, 0)); // sxtb x0,x0
    try testing.expectEqual(@as(u32, 0x93403C00), sxth(0, 0)); // sxth x0,x0
    try testing.expectEqual(@as(u32, 0x93407C00), sxtw(0, 0)); // sxtw x0,x0
    try testing.expectEqual(@as(u32, 0x92401C00), andLowBits(0, 0, 7)); // and x0,x0,#0xff
    try testing.expectEqual(@as(u32, 0x92403C00), andLowBits(0, 0, 15)); // and x0,x0,#0xffff
    try testing.expectEqual(@as(u32, 0x92407C00), andLowBits(0, 0, 31)); // and x0,x0,#0xffffffff
}

test "bitwise, shift, and select encoders" {
    try testing.expectEqual(@as(u32, 0x8A010000), andReg(0, 0, 1)); // and x0,x0,x1
    try testing.expectEqual(@as(u32, 0xAA010000), orrReg(0, 0, 1)); // orr x0,x0,x1
    try testing.expectEqual(@as(u32, 0xCA010000), eorReg(0, 0, 1)); // eor x0,x0,x1
    try testing.expectEqual(@as(u32, 0xAA2103E0), orn(0, XZR, 1)); // orn x0,xzr,x1
    try testing.expectEqual(@as(u32, 0xAA2003E0), mvn(0, 0)); // mvn x0,x0
    try testing.expectEqual(@as(u32, 0x9AC02000), lslv(0, 0, 0)); // lslv x0,x0,x0
    try testing.expectEqual(@as(u32, 0x9AC12000), lslv(0, 0, 1)); // lslv x0,x0,x1
    try testing.expectEqual(@as(u32, 0x9AC12400), lsrv(0, 0, 1)); // lsrv x0,x0,x1
    try testing.expectEqual(@as(u32, 0x9AC12800), asrv(0, 0, 1)); // asrv x0,x0,x1
    try testing.expectEqual(@as(u32, 0x937FFC00), asrImm(0, 0, 63)); // asr x0,x0,#63
    try testing.expectEqual(@as(u32, 0x937FFC02), asrImm(2, 0, 63)); // asr x2,x0,#63
    try testing.expectEqual(@as(u32, 0x9A8023E0), csel(0, XZR, 0, .hs)); // csel x0,xzr,x0,hs
    try testing.expectEqual(@as(u32, 0x9A8123E0), csel(0, XZR, 1, .hs)); // csel x0,xzr,x1,hs
}

test "unsigned condition codes invert and cset" {
    try testing.expectEqual(Cond.lo, invert(.hs));
    try testing.expectEqual(Cond.hs, invert(.lo));
    try testing.expectEqual(Cond.ls, invert(.hi));
    try testing.expectEqual(Cond.hi, invert(.ls));
    try testing.expectEqual(@as(u32, 0x9A9F97E0), cset(0, .hi)); // cset x0,hi
}

test "load/store sp-relative" {
    try testing.expectEqual(@as(u32, 0xF90007E0), strSp(0, 8)); // str x0,[sp,#8]
    try testing.expectEqual(@as(u32, 0xF90003E0), strSp(0, 0)); // str x0,[sp,#0]
    try testing.expectEqual(@as(u32, 0xF94007E0), ldrSp(0, 8)); // ldr x0,[sp,#8]
    try testing.expectEqual(@as(u32, 0xF94007E1), ldrSp(1, 8)); // ldr x1,[sp,#8]
}

test "load/store fp-relative" {
    try testing.expectEqual(@as(u32, 0xF9400BA9), ldrFp(9, 16)); // ldr x9,[x29,#16]
    try testing.expectEqual(@as(u32, 0xF9400FA9), ldrFp(9, 24)); // ldr x9,[x29,#24]
}

test "branch/system and frame constants" {
    try testing.expectEqual(@as(u32, 0x94000000), bl(0)); // bl #0
    try testing.expectEqual(@as(u32, 0xA9BF7BFD), stpFpLrPre);
    try testing.expectEqual(@as(u32, 0x910003FD), movFpSp);
    try testing.expectEqual(@as(u32, 0xA8C17BFD), ldpFpLrPost);
    try testing.expectEqual(@as(u32, 0xD65F03C0), ret);
    try testing.expectEqual(@as(u32, 0xD4200000), brk0); // brk #0
}

test "pc-relative data addressing + indirect call" {
    try testing.expectEqual(@as(u32, 0x90000008), adrp(8, 0)); // adrp x8, 0
    try testing.expectEqual(@as(u32, 0x90000010), adrp(16, 0)); // adrp x16, 0
    try testing.expectEqual(@as(u32, 0x90000030), adrp(16, 4)); // adrp x16, +4 pages
    try testing.expectEqual(@as(u32, 0xF9400210), ldrRegUoff(16, 16, 0)); // ldr x16,[x16]
    try testing.expectEqual(@as(u32, 0xF9400E10), ldrRegUoff(16, 16, 0x18)); // ldr x16,[x16,#0x18]
    try testing.expectEqual(@as(u32, 0xD63F0200), blr(16)); // blr x16
    try testing.expectEqual(@as(u32, 0xAA0103E2), movReg(2, 1)); // mov x2, x1
    try testing.expectEqual(@as(u32, 0xAA0003E1), movReg(1, 0)); // mov x1, x0
}

test "str via arbitrary base register (indirect/x8 sret + struct copy)" {
    try testing.expectEqual(@as(u32, 0xF9000100), strRegUoff(0, 8, 0)); // str x0, [x8]
    try testing.expectEqual(@as(u32, 0xF9000501), strRegUoff(1, 8, 8)); // str x1, [x8, #8]
    try testing.expectEqual(@as(u32, 0xF9000909), strRegUoff(9, 8, 16)); // str x9, [x8, #16]
}

test "ldrb zero-extended byte load (str byte-compare)" {
    try testing.expectEqual(@as(u32, 0x39400000), ldrbRegUoff(0, 0, 0)); // ldrb w0,[x0]
    try testing.expectEqual(@as(u32, 0x39400400), ldrbRegUoff(0, 0, 1)); // ldrb w0,[x0,#1]
    try testing.expectEqual(@as(u32, 0x39401441), ldrbRegUoff(1, 2, 5)); // ldrb w1,[x2,#5]
    try testing.expectEqual(@as(u32, 0x397FFC00), ldrbRegUoff(0, 0, 4095)); // ldrb w0,[x0,#4095]
}

test "in-place patchers preserve register fields and resolve immediates" {
    // adrp x8,0 placeholder → adrp x8,0 (delta 0) stays 0x90000008.
    try testing.expectEqual(@as(u32, 0x90000008), patchAdrp(adrp(8, 0), 0));
    // adrp x16,0 placeholder patched to +4 pages → 0x90000030 (rd preserved).
    try testing.expectEqual(@as(u32, 0x90000030), patchAdrp(adrp(16, 0), 4));
    // add x8,x8,#0 placeholder patched to #0x4b0 → 0x9112c108 (rd/rn preserved).
    try testing.expectEqual(@as(u32, 0x9112C108), patchAddImm12(addImm(8, 8, 0), 0x4b0));
    // ldr x16,[x16,#0] placeholder patched to offset 0 → 0xf9400210 (rt/rn kept).
    try testing.expectEqual(@as(u32, 0xF9400210), patchLdrUoff(ldrRegUoff(16, 16, 0), 0));
    try testing.expectEqual(@as(u32, 0xF9400E10), patchLdrUoff(ldrRegUoff(16, 16, 0), 0x18));
}

test "compare + condition codes" {
    try testing.expectEqual(@as(u32, 0xEB00003F), cmpReg(1, 0)); // cmp x1,x0
    try testing.expectEqual(@as(u32, 0xEB01001F), cmpReg(0, 1)); // cmp x0,x1
    try testing.expectEqual(@as(u32, 0xF100141F), cmpImm(0, 5)); // cmp x0,#5
    try testing.expectEqual(@as(u32, 0xF100001F), cmpImm(0, 0)); // cmp x0,#0
    try testing.expectEqual(@as(u32, 0xF100003F), cmpImm(1, 0)); // cmp x1,#0

    try testing.expectEqual(@as(u32, 0x9A9F17E0), cset(0, .eq)); // cset x0,eq
    try testing.expectEqual(@as(u32, 0x9A9F07E0), cset(0, .ne)); // cset x0,ne
    try testing.expectEqual(@as(u32, 0x9A9FA7E0), cset(0, .lt)); // cset x0,lt
    try testing.expectEqual(@as(u32, 0x9A9FC7E0), cset(0, .le)); // cset x0,le
    try testing.expectEqual(@as(u32, 0x9A9FD7E0), cset(0, .gt)); // cset x0,gt
    try testing.expectEqual(@as(u32, 0x9A9FB7E0), cset(0, .ge)); // cset x0,ge
    try testing.expectEqual(@as(u32, 0x9A9FA7E2), cset(2, .lt)); // cset x2,lt

    try testing.expectEqual(Cond.ge, invert(.lt));
    try testing.expectEqual(Cond.ne, invert(.eq));
    try testing.expectEqual(Cond.le, invert(.gt));
    try testing.expectEqual(Cond.eq, invert(.ne));
    try testing.expectEqual(Cond.lt, invert(.ge));
    try testing.expectEqual(Cond.gt, invert(.le));
}

test "conditional + unconditional branches" {
    try testing.expectEqual(@as(u32, 0xB5000000), cbnz(0, 0)); // cbnz x0,.+0
    try testing.expectEqual(@as(u32, 0xB5FFFF80), cbnz(0, -4)); // cbnz x0,.-16

    try testing.expectEqual(@as(u32, 0x14000000), b(0)); // b .+0
    try testing.expectEqual(@as(u32, 0x14000002), b(2)); // b .+8
    try testing.expectEqual(@as(u32, 0x17FFFFFE), b(-2)); // b .-8
    try testing.expectEqual(@as(u32, 0x14000004), b(4)); // b .+16
    try testing.expectEqual(@as(u32, 0x17FFFFFC), b(-4)); // b .-16
}

test "byte-store + conditional-branch encoders" {
    // strb wt,[xn,#imm] — byte store, unscaled imm12 (mirrors ldrb).
    try testing.expectEqual(@as(u32, 0x3900014D), strb(13, 10, 0)); // strb w13,[x10]
    try testing.expectEqual(@as(u32, 0x39000400), strb(0, 0, 1)); // strb w0,[x0,#1]
    try testing.expectEqual(@as(u32, 0x3900002D), strb(13, 1, 0)); // strb w13,[x1]
    // cbz — symmetric to cbnz (opcode 0xB4 vs 0xB5).
    try testing.expectEqual(@as(u32, 0xB4000000), cbz(0, 0)); // cbz x0,.+0
    try testing.expectEqual(@as(u32, 0xB4FFFF80), cbz(0, -4)); // cbz x0,.-16
    // b.cond — resolved-offset conditional branch (cond in low 4 bits).
    try testing.expectEqual(@as(u32, 0x5400004A), bCond(.ge, 2)); // b.ge .+8
    try testing.expectEqual(@as(u32, 0x5400008A), bCond(.ge, 4)); // b.ge .+16
    try testing.expectEqual(@as(u32, 0x54000000), bCond(.eq, 0)); // b.eq .+0
    // A resolved b.cond round-trips through the placeholder patcher (same imm field).
    try testing.expectEqual(bCond(.lt, -2), patchBCond(bCond(.lt, 0), -2));
}

test "branch patchers preserve opcode + identity" {
    try testing.expectEqual(@as(u32, 0x54000040), patchBCond(0x54000000, 2)); // keep .eq, set +2
    try testing.expectEqual(@as(u32, 0x54FFFFCB), patchBCond(0x5400000B, -2)); // keep .lt, set -2
    try testing.expectEqual(@as(u32, 0xB4FFFF80), patchCbz(0xB4000000, -4)); // keep cbz x0, set -4
    try testing.expectEqual(@as(u32, 0xB5FFFF80), patchCbz(cbnz(0, 0), -4)); // keep cbnz x0, set -4
    try testing.expectEqual(@as(u32, 0x17FFFFFE), patchB(b(0), -2)); // b -2
    try testing.expectEqual(@as(u32, 0x14000004), patchB(b(0), 4)); // b +4
}

test "movImm64 round-trips" {
    const cases = [_]i64{ 0, 40, -1, std.math.minInt(i64), @bitCast(@as(u64, 0xFFFFFFFFFFFFFFFF)) };
    for (cases) |value| {
        var buf: [16]u8 = undefined;
        var len: usize = 0;
        movImm64(&buf, &len, 3, value);
        try testing.expect(len % 4 == 0);
        try testing.expect(len >= 4 and len <= 16);

        // Decode the emitted movz/movk lanes back into a u64 and compare.
        var reg: u64 = 0;
        var i: usize = 0;
        while (i < len) : (i += 4) {
            const w = std.mem.readInt(u32, buf[i..][0..4], .little);
            try testing.expectEqual(@as(u32, 3), w & 0x1F); // rd == x3
            const hw: u6 = @intCast((w >> 21) & 0x3);
            const imm16: u64 = (w >> 5) & 0xFFFF;
            const is_movz = (w & 0xFF800000) == 0xD2800000;
            const is_movk = (w & 0xFF800000) == 0xF2800000;
            try testing.expect(is_movz or is_movk);
            reg |= imm16 << (hw * 16);
        }
        try testing.expectEqual(@as(u64, @bitCast(value)), reg);
    }
}
