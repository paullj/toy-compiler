//! The precise container traces: `ga_trace` (Vec) and `mp_trace` (Map/Set), both
//! `gc_array`-backed and descriptor-driven.

const std = @import("std");
const Aarch64 = @import("../Aarch64.zig");
const Link = @import("../../link/Link.zig");
const Abi = @import("../abi/Abi.zig");
const em = @import("emit.zig");
const emitWord = em.emitWord;
const emitFramePrologue = em.emitFramePrologue;
const emitTextBase = em.emitTextBase;
const emitBuiltinCall = em.emitBuiltinCall;
const finishBuiltin = em.finishBuiltin;
const deinitBuiltinRelocs = em.deinitBuiltinRelocs;
const patchCbzTo = em.patchCbzTo;
const patchBCondTo = em.patchBCondTo;
const patchBTo = em.patchBTo;

/// Build the `ga_trace(header)` builtin: the precise trace of a `gc_array`-backed Vec. x0 =
/// the header. Reads the self-describing stash `{len@0, elems@16, elem_desc@40,
/// elem_size@48}`: leaf-marks the `elems` backing (kept, never scanned), then — when the
/// element descriptor names a trace unit (`trace_off@40 != 0`) — walks the DENSE
/// `elems[0..len)` and dispatches each live element through `(text_base + trace_off)`. An
/// unmanaged element (int/scalar → `trace_off == 0`) skips the walk: the leaf-marked backing
/// already keeps it. Erased/non-generic (everything read from the header), so one body traces
/// every Vec. Loop state lives in callee-saved regs saved in the prologue; a `Vec[Ref[int]]`'s
/// per-element trace `bl`s `gc_mark`, which honours the callee-saved ABI, so the state
/// survives. Caller owns the result.
pub fn lowerGaTrace(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 48));
    try emit(&code, gpa, A.strSp(19, 0)); // elems
    try emit(&code, gpa, A.strSp(20, 8)); // len
    try emit(&code, gpa, A.strSp(21, 16)); // elem_size
    try emit(&code, gpa, A.strSp(22, 24)); // i
    try emit(&code, gpa, A.strSp(23, 32)); // element trace fn addr

    // x0 = header. Load the loop constants + compute the element trace fn address FIRST, so
    // nothing below reads scratch clobbered by the gc_mark_leaf call.
    try emit(&code, gpa, A.ldrRegUoff(20, 0, 0)); // len
    try emit(&code, gpa, A.ldrRegUoff(21, 0, 48)); // elem_size
    try emit(&code, gpa, A.ldrRegUoff(2, 0, 40)); // elem_desc
    try emit(&code, gpa, A.ldrRegUoff(19, 0, 16)); // elems
    try emit(&code, gpa, A.movz(23, 0, 0)); // fn addr = 0 (no element trace)
    const cbz_skip_a: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(2, 0)); // elem_desc == 0 → no trace
    try emit(&code, gpa, A.ldrRegUoff(3, 2, Abi.desc.trace_off)); // descriptor trace_off
    const cbz_skip_b: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(3, 0)); // trace_off == 0 → unmanaged element
    try emitTextBase(&code, &relocs, gpa, 4); // x4 = text_base
    try emit(&code, gpa, A.addReg(23, 4, 3)); // fn addr = text_base + trace_off
    const skip: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_skip_a, skip);
    patchCbzTo(code.items, cbz_skip_b, skip);

    // leaf-mark the backing (survives the sweep; its interior is reached only via the walk).
    try emit(&code, gpa, A.movReg(0, 19));
    try emitBuiltinCall(&code, &relocs, gpa, "gc_mark_leaf");

    const cbz_done: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(23, 0)); // no element trace → done
    try emit(&code, gpa, A.movz(22, 0, 0)); // i = 0
    const loop_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cmpReg(22, 20));
    const bhs_done: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0)); // i >= len → done
    try emit(&code, gpa, A.mul(5, 22, 21));
    try emit(&code, gpa, A.addReg(0, 19, 5)); // x0 = elems + i*elem_size
    try emit(&code, gpa, A.blr(23));
    try emit(&code, gpa, A.addImm(22, 22, 1));
    const b_loop: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_loop, loop_top);

    const done: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_done, done);
    patchBCondTo(code.items, bhs_done, done);
    try emit(&code, gpa, A.ldrSp(19, 0));
    try emit(&code, gpa, A.ldrSp(20, 8));
    try emit(&code, gpa, A.ldrSp(21, 16));
    try emit(&code, gpa, A.ldrSp(22, 24));
    try emit(&code, gpa, A.ldrSp(23, 32));
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 48));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "ga_trace");
}

/// Build the `mp_trace(header)` builtin: the precise trace of a `gc_array`-backed Map (and
/// Set = `Map[T,()]`). x0 = the header. Reads the stash `{buckets@0, entries@8, len@24,
/// key_desc@40, val_desc@48, key_size@56, stride@64}`: leaf-marks BOTH backings, then walks
/// the DENSE `entries[0..len)` (insertion order — open-addressing scatter is invisible to
/// it, so every live entry is visited regardless of which physical bucket it probed into)
/// and dispatches each entry's key (at cell+0) and value (at cell+key_size) through their
/// descriptors' trace units when present. A `()` value (Set) has `val_desc.trace_off == 0`,
/// so the value dispatch is a no-op. Because the backings are leaf-marked (not scanned), a
/// scattered collision key's buffer is reached ONLY through this dense walk — a physical or
/// truncated walk would sweep it. Erased/non-generic. Caller owns the result.
pub fn lowerMpTrace(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 80));
    try emit(&code, gpa, A.strSp(19, 0)); // entries
    try emit(&code, gpa, A.strSp(20, 8)); // len
    try emit(&code, gpa, A.strSp(21, 16)); // stride
    try emit(&code, gpa, A.strSp(22, 24)); // i
    try emit(&code, gpa, A.strSp(23, 32)); // key trace fn addr
    try emit(&code, gpa, A.strSp(24, 40)); // preserve caller x24 (reloaded per-entry, not held)
    try emit(&code, gpa, A.strSp(25, 48)); // val trace fn addr
    try emit(&code, gpa, A.strSp(26, 56)); // header
    try emit(&code, gpa, A.strSp(27, 64)); // text_base

    try emit(&code, gpa, A.movReg(26, 0)); // header

    // leaf-mark buckets and entries (both backings survive; interiors reached via the walk).
    try emit(&code, gpa, A.ldrRegUoff(0, 26, 0)); // buckets
    try emitBuiltinCall(&code, &relocs, gpa, "gc_mark_leaf");
    try emit(&code, gpa, A.ldrRegUoff(0, 26, 8)); // entries
    try emitBuiltinCall(&code, &relocs, gpa, "gc_mark_leaf");

    try emit(&code, gpa, A.ldrRegUoff(19, 26, 8)); // entries
    try emit(&code, gpa, A.ldrRegUoff(20, 26, 24)); // len
    try emit(&code, gpa, A.ldrRegUoff(21, 26, 64)); // stride
    try emitTextBase(&code, &relocs, gpa, 27); // x27 = text_base

    // key trace fn addr → x23 (0 when the key type is unmanaged).
    try emit(&code, gpa, A.movz(23, 0, 0));
    try emit(&code, gpa, A.ldrRegUoff(4, 26, 40)); // key_desc
    const cbz_nok: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(4, 0));
    try emit(&code, gpa, A.ldrRegUoff(5, 4, Abi.desc.trace_off)); // key descriptor trace_off
    const cbz_nok2: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(5, 0));
    try emit(&code, gpa, A.addReg(23, 27, 5));
    const nok: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_nok, nok);
    patchCbzTo(code.items, cbz_nok2, nok);

    // val trace fn addr → x25 (0 for a `()` value / unmanaged value).
    try emit(&code, gpa, A.movz(25, 0, 0));
    try emit(&code, gpa, A.ldrRegUoff(4, 26, 48)); // val_desc
    const cbz_nov: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(4, 0));
    try emit(&code, gpa, A.ldrRegUoff(5, 4, Abi.desc.trace_off)); // val descriptor trace_off
    const cbz_nov2: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(5, 0));
    try emit(&code, gpa, A.addReg(25, 27, 5));
    const nov: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_nov, nov);
    patchCbzTo(code.items, cbz_nov2, nov);

    try emit(&code, gpa, A.movz(22, 0, 0)); // i = 0
    const loop_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cmpReg(22, 20));
    const bhs_done: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0)); // i >= len → done

    // key dispatch: x0 = entries + i*stride.
    const cbz_dov: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(23, 0)); // no key trace → skip to value
    try emit(&code, gpa, A.mul(6, 22, 21));
    try emit(&code, gpa, A.addReg(0, 19, 6));
    try emit(&code, gpa, A.blr(23));
    const dov: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_dov, dov);

    // value dispatch: x0 = entries + i*stride + key_size (recomputed; the key blr clobbers scratch).
    const cbz_next: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(25, 0)); // no value trace → skip
    try emit(&code, gpa, A.mul(6, 22, 21));
    try emit(&code, gpa, A.addReg(0, 19, 6));
    // Reload key_size from the header: a key/value trace that routes through gc_mark resets
    // x24 (the collector's pending-push descriptor) to 0, so it cannot be kept across the loop.
    try emit(&code, gpa, A.ldrRegUoff(24, 26, 56));
    try emit(&code, gpa, A.addReg(0, 0, 24)); // + key_size
    try emit(&code, gpa, A.blr(25));
    const next: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_next, next);

    try emit(&code, gpa, A.addImm(22, 22, 1));
    const b_loop: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_loop, loop_top);

    const done: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, bhs_done, done);
    try emit(&code, gpa, A.ldrSp(19, 0));
    try emit(&code, gpa, A.ldrSp(20, 8));
    try emit(&code, gpa, A.ldrSp(21, 16));
    try emit(&code, gpa, A.ldrSp(22, 24));
    try emit(&code, gpa, A.ldrSp(23, 32));
    try emit(&code, gpa, A.ldrSp(24, 40));
    try emit(&code, gpa, A.ldrSp(25, 48));
    try emit(&code, gpa, A.ldrSp(26, 56));
    try emit(&code, gpa, A.ldrSp(27, 64));
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 80));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "mp_trace");
}
