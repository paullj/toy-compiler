//! The hand-emitted `panic` builtin (write the message + a symbolized backtrace to
//! STDERR, then SYS_exit(1)) and the `text_base()` self-locate builtin.

const std = @import("std");
const Aarch64 = @import("../Aarch64.zig");
const Link = @import("../../link/Link.zig");
const testing = std.testing;
const em = @import("emit.zig");
const emitWord = em.emitWord;
const emitFramePrologue = em.emitFramePrologue;
const emitWriteImport = em.emitWriteImport;
const emitTextBase = em.emitTextBase;
const emitFpWalk = em.emitFpWalk;
const patchCbzTo = em.patchCbzTo;
const patchBCondTo = em.patchBCondTo;
const patchBTo = em.patchBTo;
const finishBuiltin = em.finishBuiltin;
const deinitBuiltinRelocs = em.deinitBuiltinRelocs;

// panic(str) builtin body — hand-written, AST/IR-independent. Appended once at link
// time (emit.zig) when any fn references `panic`. Writes the message {ptr,len} to fd 2
// (STDERR), then a symbolized-backtrace dump — one `0x<offset> <name>` line per frame
// walked off the x29 FP chain — then exits NONZERO via SYS_exit(1) (raw `svc`, no libc):
// a deterministic, uncatchable abort. Never returns; the trailing brk #0 is an
// unreachable backstop.
//
// Each frame line is `ra - text_base - 4` in hex — the CALL SITE's slide-independent
// __text offset (post-processable by `atos`). `text_base` is self-located at runtime:
// `adr x9,.` yields this instruction's runtime address and the linker bakes that
// instruction's static __text offset into the following movz/movk (self-relative
// `.movw_g0`/`.movw_g1` relocs), so `text_base = adr_addr - baked_offset` holds under
// PIE/ASLR. Loop invariants (text_base, fp cursor, counter, buffer base, shift consts)
// live in x19-x24/x26 — callee-saved, so they survive each `write` call.
/// Emit `panic`'s per-frame backtrace work for the frame whose fp is in x20:
/// read the return address, resolve it to a `0x<call-site-offset> <name>` line
/// against the symbol table in x27, and write that line to STDERR. Reads the
/// callee-saved state `lowerPanic` seeds (x19 text_base, x22 buffer, x23/x24 hex
/// shifts, x26 &write, x27 &symtab); x4-x15/x25/x28 are scratch. Runs inside
/// `emitFpWalk`'s per-frame slot, so it does its own intra-body backpatching only.
fn emitPanicFrame(code: *std.ArrayList(u8), gpa: std.mem.Allocator) error{OutOfMemory}!void {
    const A = Aarch64;
    const emit = emitWord;

    try emit(code, gpa, A.ldrRegUoff(25, 20, 8)); // x25 = ra = *(fp+8)
    try emit(code, gpa, A.subReg(25, 25, 19)); // ra - text_base
    try emit(code, gpa, A.subImm(25, 25, 4)); // -> call site (x25 = frame offset)

    // Symbol lookup: linear-scan the sorted table for the greatest entry off <= x25;
    // x28 = its name ptr (0 = none). Entries ascend by off, so the first off > target
    // ends the scan; x28 then holds the enclosing fn's name (or the `{text_size,""}`
    // sentinel for a frame past __text). x4-x10 are scratch (no write() runs here).
    try emit(code, gpa, A.ldrRegUoff(4, 27, 0)); // x4 = count
    try emit(code, gpa, A.addImm(5, 27, 8)); // x5 = &entry[0]
    try emit(code, gpa, A.movz(6, 0, 0)); // x6 = i
    try emit(code, gpa, A.movz(28, 0, 0)); // x28 = best name ptr (none)
    const scan_top: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cmpReg(6, 4)); // i vs count
    const scan_bhs: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.hs, 0)); // i >= count -> scan_done
    try emit(code, gpa, A.ldrRegUoff(9, 5, 0)); // x9 = entry.off
    try emit(code, gpa, A.cmpReg(9, 25)); // off vs target
    const scan_bhi: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.hi, 0)); // off > target -> scan_done
    try emit(code, gpa, A.ldrRegUoff(10, 5, 8)); // x10 = entry.name_off
    try emit(code, gpa, A.addReg(28, 27, 10)); // x28 = base + name_off
    try emit(code, gpa, A.addImm(5, 5, 16)); // ++entry
    try emit(code, gpa, A.addImm(6, 6, 1)); // ++i
    const scan_b: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.b(0)); // -> scan_top
    const scan_done: u32 = @intCast(code.items.len);

    // Prefix "0x" into the buffer.
    try emit(code, gpa, A.movz(9, '0', 0));
    try emit(code, gpa, A.strb(9, 22, 0));
    try emit(code, gpa, A.movz(9, 'x', 0));
    try emit(code, gpa, A.strb(9, 22, 1));
    try emit(code, gpa, A.addImm(11, 22, 2)); // x11 = write cursor
    try emit(code, gpa, A.movz(12, 16, 0)); // x12 = 16 nibbles
    const hex_top: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.lsrv(13, 25, 24)); // top nibble = val >> 60
    try emit(code, gpa, A.lslv(25, 25, 23)); // val <<= 4
    try emit(code, gpa, A.andLowBits(13, 13, 3)); // & 0xF
    try emit(code, gpa, A.cmpImm(13, 10));
    try emit(code, gpa, A.addImm(14, 13, '0')); // '0' + n
    try emit(code, gpa, A.addImm(15, 13, 'a' - 10)); // 'a'-10 + n
    try emit(code, gpa, A.csel(14, 14, 15, .lo)); // n < 10 ? digit : letter
    try emit(code, gpa, A.strb(14, 11, 0));
    try emit(code, gpa, A.addImm(11, 11, 1));
    try emit(code, gpa, A.subImm(12, 12, 1));
    const cbnz_hex: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbnz(12, 0)); // more nibbles -> HEX

    // Append ' ' then FLUSH the prefix. The buffer only ever holds "0x" + 16 nibbles +
    // ' ' (19 bytes ≤ 64), so it cannot overflow whatever the (arbitrary-length) name is.
    try emit(code, gpa, A.movz(9, ' ', 0));
    try emit(code, gpa, A.strb(9, 11, 0));
    try emit(code, gpa, A.addImm(11, 11, 1));
    try emit(code, gpa, A.subReg(2, 11, 22)); // len = cursor - base
    try emit(code, gpa, A.movReg(1, 22)); // buf
    try emit(code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(code, gpa, A.blr(26)); // write "0x<hex> "

    // The symbol name, written DIRECTLY from its read-only __cstring pointer (x28):
    // strlen then write — no copy into the fixed buffer, so no length bound on names.
    // A missing entry (x28 == 0) or the sentinel's empty name writes nothing.
    const name_guard: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbz(28, 0)); // no entry -> name_done
    try emit(code, gpa, A.movReg(9, 28)); // x9 = scan ptr
    const strlen_top: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.ldrbRegUoff(10, 9, 0)); // w10 = *ptr
    const strlen_cbz: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbz(10, 0)); // NUL -> strlen_done
    try emit(code, gpa, A.addImm(9, 9, 1));
    const strlen_b: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.b(0)); // -> strlen_top
    const strlen_done: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.subReg(2, 9, 28)); // len = ptr - name
    try emit(code, gpa, A.movReg(1, 28)); // buf = name ptr
    try emit(code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(code, gpa, A.blr(26)); // write the name
    const name_done: u32 = @intCast(code.items.len);

    // Trailing newline (always, even for an unnamed frame).
    try emit(code, gpa, A.movz(9, '\n', 0));
    try emit(code, gpa, A.strb(9, 22, 0)); // buffer[0] = '\n'
    try emit(code, gpa, A.movz(2, 1, 0)); // len 1
    try emit(code, gpa, A.movReg(1, 22)); // buf
    try emit(code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(code, gpa, A.blr(26)); // write '\n'

    patchCbzTo(code.items, cbnz_hex, hex_top);
    patchBCondTo(code.items, scan_bhs, scan_done);
    patchBCondTo(code.items, scan_bhi, scan_done);
    patchBTo(code.items, scan_b, scan_top);
    patchCbzTo(code.items, name_guard, name_done);
    patchCbzTo(code.items, strlen_cbz, strlen_done);
    patchBTo(code.items, strlen_b, strlen_top);
}

/// The `text_base()` builtin: return the runtime base of `__text` in x0. No frame is
/// opened — the body makes no calls and the movw addends are site-relative, so it is
/// position-independent without a saved link register.
pub fn lowerTextBase(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    try emitTextBase(&code, &relocs, gpa, 0);
    try emitWord(&code, gpa, Aarch64.ret);
    return finishBuiltin(&code, &relocs, gpa, "text_base");
}

pub fn lowerPanic(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    const emit = emitWord;

    // Prologue + 64-byte scratch buffer for the hex line (`0x` + 16 nibbles + `\n`).
    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 64)); // scratch buffer
    try emit(&code, gpa, A.addImm(22, A.SP, 0)); // x22 = buffer base

    // write(2, msg_ptr, msg_len): incoming x0=ptr, x1=len. Move len (x1->x2) BEFORE
    // overwriting x1 with ptr (x0->x1). Load &write from its GOT slot ONCE into the
    // callee-saved x26, then `blr x26` for the message and every frame line.
    try emit(&code, gpa, A.movReg(2, 1)); // len -> x2
    try emit(&code, gpa, A.movReg(1, 0)); // ptr -> x1
    try emit(&code, gpa, A.movz(0, 2, 0)); // fd = 2 (STDERR)
    try emitWriteImport(&code, &relocs, gpa, 26); // x26 = &write
    try emit(&code, gpa, A.blr(26)); // write the message

    // One '\n' separator: messages carry no trailing newline, so this keeps the
    // first frame line off the message regardless of the message's contents.
    try emit(&code, gpa, A.movz(9, '\n', 0));
    try emit(&code, gpa, A.strb(9, 22, 0)); // buffer[0] = '\n'
    try emit(&code, gpa, A.movz(2, 1, 0)); // len 1
    try emit(&code, gpa, A.movReg(1, 22)); // buf
    try emit(&code, gpa, A.movz(0, 2, 0)); // fd 2
    try emit(&code, gpa, A.blr(26)); // write '\n'

    // Self-locate text_base into x19 (see the doc comment).
    try emitTextBase(&code, &relocs, gpa, 19);
    var site: u32 = undefined;

    // Load the backtrace symbol table base into the callee-saved x27 (survives every
    // write() call). The relink tail reserved `symtab_base_hash` at the table's
    // __cstring offset, so this resolves like a string-literal pointer (adrp+add).
    site = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .{ .cstr = Link.symtab_base_hash }, .kind = .adrp_page });
    try emit(&code, gpa, A.adrp(27, 0)); // adrp x27, symtab@page
    site = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .{ .cstr = Link.symtab_base_hash }, .kind = .add_lo12 });
    try emit(&code, gpa, A.addImm(27, 27, 0)); // x27 = &symtab

    // Walk state: x20 = fp cursor (this frame), x21 = frame cap, x23/x24 = hex shifts.
    try emit(&code, gpa, A.movReg(20, A.FP)); // x20 = x29
    try emit(&code, gpa, A.movz(21, 64, 0)); // x21 = 64 (frame cap)
    try emit(&code, gpa, A.movz(23, 4, 0)); // x23 = 4
    try emit(&code, gpa, A.movz(24, 60, 0)); // x24 = 60

    // Walk the x29 chain, symbolizing each live frame; the loop falls through
    // when fp hits 0 (or the 64-frame cap trips), leaving x20 == 0.
    try emitFpWalk(&code, gpa, .each_frame, 20, 21, undefined, emitPanicFrame);

    // SYS_exit(1): x0 = status, x16 = SYS_exit, svc #0x80. Never returns.
    try emit(&code, gpa, A.movz(0, 1, 0));
    try emit(&code, gpa, A.movz(16, 1, 0));
    try emit(&code, gpa, A.svc0x80);
    try emit(&code, gpa, A.brk0); // unreachable backstop

    return finishBuiltin(&code, &relocs, gpa, "panic");
}

test "lowerPanic: prologue, self-locate movw relocs, hex frame loop, then SYS_exit(1)" {
    const gpa = testing.allocator;
    var fc = try lowerPanic(gpa);
    defer fc.deinit(gpa);
    try testing.expectEqual(Link.SymKind.builtin, fc.sym.kind);
    try testing.expectEqualStrings("panic", fc.sym.name);
    try testing.expect(fc.code.len % 4 == 0);
    // Opens with the frame prologue.
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, fc.code[0..4], .little));
    // Ends with the never-returning SYS_exit(1) + brk backstop.
    const n = fc.code.len;
    try testing.expectEqual(Aarch64.brk0, std.mem.readInt(u32, fc.code[n - 4 ..][0..4], .little));
    try testing.expectEqual(Aarch64.svc0x80, std.mem.readInt(u32, fc.code[n - 8 ..][0..4], .little));
    // The self-locate `adr x9, .` and the hex-loop `csel x14,x14,x15,lo` are present.
    var saw_adr = false;
    var saw_csel = false;
    var i: usize = 0;
    while (i < fc.code.len) : (i += 4) {
        const word = std.mem.readInt(u32, fc.code[i..][0..4], .little);
        if (word == Aarch64.adr(9, 0)) saw_adr = true;
        if (word == Aarch64.csel(14, 14, 15, .lo)) saw_csel = true;
    }
    try testing.expect(saw_adr);
    try testing.expect(saw_csel);
    // Six relocs: two `write` imports (adrp_page + ldr_lo12) reached via the GOT, the
    // two self-relative movw relocs that bake the text-base offset, then the adrp+add
    // pair that loads the backtrace symbol table base (a `.cstr` to the reserved hash).
    try testing.expectEqual(@as(usize, 6), fc.relocs.len);
    try testing.expectEqual(Link.RelocKind.adrp_page, fc.relocs[0].kind);
    try testing.expectEqualStrings("write", fc.relocs[0].target.import.name);
    try testing.expectEqual(Link.RelocKind.ldr_lo12, fc.relocs[1].kind);
    try testing.expectEqualStrings("write", fc.relocs[1].target.import.name);
    try testing.expectEqual(Link.RelocKind.movw_g0, fc.relocs[2].kind);
    try testing.expectEqual(Link.RelocKind.movw_g1, fc.relocs[3].kind);
    // The movw addends name the `adr` site relative to each mov site (−4, −8).
    try testing.expectEqual(@as(i64, -4), fc.relocs[2].addend);
    try testing.expectEqual(@as(i64, -8), fc.relocs[3].addend);
    try testing.expectEqual(Link.RelocKind.adrp_page, fc.relocs[4].kind);
    try testing.expectEqual(Link.symtab_base_hash, fc.relocs[4].target.cstr);
    try testing.expectEqual(Link.RelocKind.add_lo12, fc.relocs[5].kind);
    try testing.expectEqual(Link.symtab_base_hash, fc.relocs[5].target.cstr);
}
