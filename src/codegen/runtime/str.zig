//! The hand-emitted `str` builtins: `__int_to_str` and `__str_concat`, both
//! `gc_alloc`-backed packed-byte-buffer producers appended at link time.

const std = @import("std");
const Aarch64 = @import("../Aarch64.zig");
const Link = @import("../../link/Link.zig");
const testing = std.testing;
const em = @import("emit.zig");
const emitWord = em.emitWord;
const emitFramePrologue = em.emitFramePrologue;
const emitBuiltinCall = em.emitBuiltinCall;
const finishBuiltin = em.finishBuiltin;
const deinitBuiltinRelocs = em.deinitBuiltinRelocs;

// __int_to_str(n) builtin body — hand-written, AST/IR-independent. Renders the i64 in x0
// into a fresh `gc_alloc`'d packed byte buffer and returns the str reg-pair
// (x0 = ptr@buffer-base, x1 = len). Appended at link time (emit.zig) when any fn
// references `__int_to_str`; it allocates, so its used-scan pulls in `gc_alloc` (and
// `gc_collect` by closure).
//
// Digit extraction avoids i64::MIN overflow by NEVER negating the running value: each step
// computes q = n/10 (sdiv, truncates toward zero) and the single-digit remainder
// r = n - q*10 (range -9..9), takes |r| (safe — |r| <= 9), and stores '0'+|r|. The original
// sign (saved in x9) prepends a '-'. n==0 emits one '0' via the do-while shape. Format
// backward into a 32-byte STACK buffer, giving the digit start (cursor) and length. Two
// extra reserved stack words hold cursor/len across the
// `gc_alloc` call (which clobbers x0-x18); the buffer is sp-addressed and survives. Then
// copy exactly `len` bytes forward into the cell (ptr@0, no header), matching the uniform
// owned-str model, and return.

/// Build the `__int_to_str` builtin's FnCode directly. Caller owns the result.
pub fn lowerIntToStr(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    const A = Aarch64;
    const emit = emitWord;

    // Frame + 48-byte scratch: [sp+0]=saved cursor, [sp+8]=saved len (survive the
    // gc_alloc call), [sp+16 .. sp+48) = 32-byte digit buffer (buf_end = sp+48).
    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 48));
    try emit(&code, gpa, A.movReg(9, 0)); // x9 = n (saved for the sign test)
    try emit(&code, gpa, A.movz(11, 10, 0)); // x11 = 10 (divisor)
    try emit(&code, gpa, A.movz(14, 0x30, 0)); // x14 = '0'
    try emit(&code, gpa, A.addImm(10, A.SP, 48)); // x10 = &buf_end (write cursor, grows down)
    // do-while digit loop (LOOP = this sdiv) — the never-negate render (i64::MIN-correct).
    try emit(&code, gpa, A.sdiv(12, 0, 11)); // x12 = n / 10
    try emit(&code, gpa, A.mul(13, 12, 11)); // x13 = q * 10
    try emit(&code, gpa, A.subReg(13, 0, 13)); // x13 = n - q*10 (signed remainder)
    try emit(&code, gpa, A.cmpImm(13, 0)); // cmp r, #0
    try emit(&code, gpa, A.bCond(.ge, 2)); // b.ge +2 (skip the neg if r >= 0)
    try emit(&code, gpa, A.neg(13, 13)); // x13 = -r (|r|, safe: |r| <= 9)
    try emit(&code, gpa, A.addReg(13, 13, 14)); // x13 = '0' + digit
    try emit(&code, gpa, A.subImm(10, 10, 1)); // --cursor
    try emit(&code, gpa, A.strb(13, 10, 0)); // *cursor = digit char
    try emit(&code, gpa, A.movReg(0, 12)); // n = q
    try emit(&code, gpa, A.cbnz(0, -10)); // cbnz n, LOOP (back 10 words to the sdiv)
    // Sign: if the original value was negative, prepend '-'.
    try emit(&code, gpa, A.cmpImm(9, 0)); // cmp orig, #0
    try emit(&code, gpa, A.bCond(.ge, 4)); // b.ge +4 (skip the 3 sign instrs if >= 0)
    try emit(&code, gpa, A.movz(13, 0x2D, 0)); // x13 = '-'
    try emit(&code, gpa, A.subImm(10, 10, 1)); // --cursor
    try emit(&code, gpa, A.strb(13, 10, 0)); // *cursor = '-'
    // len = buf_end - cursor; save cursor + len across the alloc.
    try emit(&code, gpa, A.addImm(1, A.SP, 48)); // x1 = &buf_end
    try emit(&code, gpa, A.subReg(2, 1, 10)); // x2 = len = end - cursor
    try emit(&code, gpa, A.strSp(10, 0)); // [sp+0] = cursor
    try emit(&code, gpa, A.strSp(2, 8)); // [sp+8] = len
    try emit(&code, gpa, A.movReg(0, 2)); // x0 = len (gc_alloc arg)
    try emitBuiltinCall(&code, &relocs, gpa, "gc_alloc"); // x0 = cell
    // Reload cursor + len; copy len bytes forward cell[0..len] <- stack[cursor..].
    try emit(&code, gpa, A.movReg(3, 0)); // x3 = cell (dst base)
    try emit(&code, gpa, A.ldrSp(10, 0)); // x10 = cursor (src base)
    try emit(&code, gpa, A.ldrSp(1, 8)); // x1 = len
    try emit(&code, gpa, A.movz(4, 0, 0)); // x4 = i = 0
    // COPY loop (8 instrs); a fixed forward/back branch pair (no backpatch needed).
    try emit(&code, gpa, A.cmpReg(4, 1)); // i vs len
    try emit(&code, gpa, A.bCond(.hs, 7)); // i >= len -> DONE (skip 6 body + the back-branch)
    try emit(&code, gpa, A.addReg(6, 10, 4)); // x6 = src + i
    try emit(&code, gpa, A.ldrbRegUoff(5, 6, 0)); // w5 = *(src+i)
    try emit(&code, gpa, A.addReg(7, 3, 4)); // x7 = dst + i
    try emit(&code, gpa, A.strb(5, 7, 0)); // *(dst+i) = byte
    try emit(&code, gpa, A.addImm(4, 4, 1)); // ++i
    try emit(&code, gpa, A.b(-7)); // -> COPY
    // DONE: return x0 = cell, x1 = len.
    try emit(&code, gpa, A.movReg(0, 3)); // x0 = cell
    // x1 already holds len.
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 48)); // scratch teardown
    try emit(&code, gpa, A.ldpFpLrPost); // ldp x29,x30,[sp],#16
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "__int_to_str");
}

// __str_concat(lp, ll, rp, rl) builtin body — hand-written, AST/IR-independent. Joins two
// str values into a fresh `gc_alloc`'d packed byte buffer and returns the str reg-pair
// (x0 = ptr@buffer-base, x1 = len = ll+rl). Args arrive as (x0=ptr_a, x1=len_a,
// x2=ptr_b, x3=len_b). Appended at link time (emit.zig) when any fn references
// `__str_concat`; it allocates, so its used-scan pulls in `gc_alloc` (and `gc_collect`).
//
// Out-of-lining the copy (vs inlining two IR byte-copy loops per call site) keeps a
// concat's frame cost off the caller: the spill-all frame model gives every SSA value its
// own slot, so an inlined copy loop would let a handful of concats overflow `FrameTooLarge`.
//
// The four inputs are saved to the 48-byte scratch BEFORE `gc_alloc` (which clobbers
// x0-x18); a collection the alloc triggers conservatively scans this frame, so the saved
// source `ptr`s keep both operand buffers marked. After the alloc the bytes are copied
// forward — `[buf .. buf+ll)` from ptr_a, `[buf+ll .. buf+ll+rl)` from ptr_b — reusing
// `__int_to_str`'s verbatim 8-instruction COPY loop (fixed +7/-7 branch pair).

/// Build the `__str_concat` builtin's FnCode directly. Caller owns the result.
pub fn lowerStrConcat(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    const A = Aarch64;
    const emit = emitWord;

    // Frame + 48-byte scratch: [sp+0]=ptr_a, [sp+8]=len_a, [sp+16]=ptr_b, [sp+24]=len_b,
    // [sp+32]=buf (all survive the gc_alloc call).
    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 48));
    try emit(&code, gpa, A.strSp(0, 0)); // [sp+0] = ptr_a
    try emit(&code, gpa, A.strSp(1, 8)); // [sp+8] = len_a
    try emit(&code, gpa, A.strSp(2, 16)); // [sp+16] = ptr_b
    try emit(&code, gpa, A.strSp(3, 24)); // [sp+24] = len_b
    try emit(&code, gpa, A.addReg(0, 1, 3)); // x0 = len_a + len_b (gc_alloc arg)
    try emitBuiltinCall(&code, &relocs, gpa, "gc_alloc"); // x0 = buf
    try emit(&code, gpa, A.strSp(0, 32)); // [sp+32] = buf

    // COPY loop 1: dst=buf, src=ptr_a, len=len_a.
    try emit(&code, gpa, A.movReg(3, 0)); // x3 = dst = buf
    try emit(&code, gpa, A.ldrSp(10, 0)); // x10 = src = ptr_a
    try emit(&code, gpa, A.ldrSp(1, 8)); // x1 = len = len_a
    try emit(&code, gpa, A.movz(4, 0, 0)); // x4 = i = 0
    try emit(&code, gpa, A.cmpReg(4, 1)); // i vs len
    try emit(&code, gpa, A.bCond(.hs, 7)); // i >= len -> next block
    try emit(&code, gpa, A.addReg(6, 10, 4)); // x6 = src + i
    try emit(&code, gpa, A.ldrbRegUoff(5, 6, 0)); // w5 = *(src+i)
    try emit(&code, gpa, A.addReg(7, 3, 4)); // x7 = dst + i
    try emit(&code, gpa, A.strb(5, 7, 0)); // *(dst+i) = byte
    try emit(&code, gpa, A.addImm(4, 4, 1)); // ++i
    try emit(&code, gpa, A.b(-7)); // -> cmp

    // COPY loop 2: dst=buf+len_a, src=ptr_b, len=len_b.
    try emit(&code, gpa, A.ldrSp(3, 32)); // x3 = buf
    try emit(&code, gpa, A.ldrSp(8, 8)); // x8 = len_a
    try emit(&code, gpa, A.addReg(3, 3, 8)); // x3 = dst = buf + len_a
    try emit(&code, gpa, A.ldrSp(10, 16)); // x10 = src = ptr_b
    try emit(&code, gpa, A.ldrSp(1, 24)); // x1 = len = len_b
    try emit(&code, gpa, A.movz(4, 0, 0)); // x4 = i = 0
    try emit(&code, gpa, A.cmpReg(4, 1));
    try emit(&code, gpa, A.bCond(.hs, 7));
    try emit(&code, gpa, A.addReg(6, 10, 4));
    try emit(&code, gpa, A.ldrbRegUoff(5, 6, 0));
    try emit(&code, gpa, A.addReg(7, 3, 4));
    try emit(&code, gpa, A.strb(5, 7, 0));
    try emit(&code, gpa, A.addImm(4, 4, 1));
    try emit(&code, gpa, A.b(-7));

    // Return x0 = buf, x1 = len_a + len_b.
    try emit(&code, gpa, A.ldrSp(0, 32)); // x0 = buf (ptr)
    try emit(&code, gpa, A.ldrSp(8, 8)); // x8 = len_a
    try emit(&code, gpa, A.ldrSp(9, 24)); // x9 = len_b
    try emit(&code, gpa, A.addReg(1, 8, 9)); // x1 = total len
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 48)); // scratch teardown
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "__str_concat");
}

test "ir-codegen: __int_to_str builtin byte shape (prologue, buffer, digit loop, gc_alloc, ret)" {
    const gpa = testing.allocator;
    var fc = try lowerIntToStr(gpa);
    defer fc.deinit(gpa);

    // A hand-asm builtin named `__int_to_str`.
    try testing.expectEqualStrings("__int_to_str", fc.sym.name);
    try testing.expectEqual(Link.SymKind.builtin, fc.sym.kind);
    // 4-byte aligned instruction stream; opens with the frame prologue.
    try testing.expect(fc.code.len % 4 == 0);
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, fc.code[0..4], .little));
    // Ends with `ret`; the two instructions before it are the frame teardown.
    const n = fc.code.len;
    try testing.expectEqual(Aarch64.ret, std.mem.readInt(u32, fc.code[n - 4 ..][0..4], .little));
    try testing.expectEqual(Aarch64.ldpFpLrPost, std.mem.readInt(u32, fc.code[n - 8 ..][0..4], .little));
    // The body contains the backward digit-loop back-edge (`cbnz x0, -10`) and at least one
    // byte store (`strb w13,[x10]`) — the shared never-negate digit core.
    var saw_cbnz = false;
    var saw_strb = false;
    var i: usize = 0;
    while (i < fc.code.len) : (i += 4) {
        const w = std.mem.readInt(u32, fc.code[i..][0..4], .little);
        if (w == Aarch64.cbnz(0, -10)) saw_cbnz = true;
        if (w == Aarch64.strb(13, 10, 0)) saw_strb = true;
    }
    try testing.expect(saw_cbnz);
    try testing.expect(saw_strb);
    // It allocates (unlike a bare fd write): a `.call26` reloc to the `gc_alloc` builtin.
    var saw_gc_alloc = false;
    for (fc.relocs) |r| {
        if (r.kind == .call26 and r.target == .func and std.mem.eql(u8, r.target.func.name, "gc_alloc")) saw_gc_alloc = true;
    }
    try testing.expect(saw_gc_alloc);
}
