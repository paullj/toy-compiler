//! The conservative, non-moving mark-sweep collector: the `gc_alloc` allocator, the
//! `gc_collect` stop-the-world collection, the `gc_mark`/`gc_mark_leaf` marking
//! wrappers, and the `gc_span_count`/`gc_stats` counters — plus the heap ABI (control
//! block, span prologue, and large-object layout) they all read.

const std = @import("std");
const Aarch64 = @import("../Aarch64.zig");
const Link = @import("../../link/Link.zig");
const Abi = @import("../abi/Abi.zig");
const testing = std.testing;
const em = @import("emit.zig");
const emitWord = em.emitWord;
const emitFramePrologue = em.emitFramePrologue;
const emitImportPreamble = em.emitImportPreamble;
const emitBuiltinCall = em.emitBuiltinCall;
const emitTextBase = em.emitTextBase;
const finishBuiltin = em.finishBuiltin;
const deinitBuiltinRelocs = em.deinitBuiltinRelocs;
const patchCbzTo = em.patchCbzTo;
const patchBCondTo = em.patchBCondTo;
const patchBTo = em.patchBTo;

// A conservative, non-moving MARK-SWEEP collector, hand-emitted like `print`/`panic`.
// Appended once at link time (emit.zig) when any fn references a heap intrinsic. There is
// no writable image segment (LC_MAIN gives no init hook, __DATA_CONST is read-only after
// dyld binds, and no __bss exists), so the collector's control block lives at a fixed
// high VM address, `mmap`ped MAP_FIXED on the first call. `msync(CB_ADDR, 1, MS_ASYNC)`
// is the fault-free is-it-initialized probe: it returns 0 when the page is mapped and −1
// (ENOMEM) when it is not. (macOS `mincore` cannot serve here — it returns 0 for an
// unmapped page too.) The CB page is metadata-only (span_count, collection count, a
// growth watermark, the span/large lists, the mark-stack fields, and 13 per-class
// freelist heads); a fresh zeroed page leaves every field 0, so nothing is seeded.
//
// Each 16 KiB size-class span carries a 256-byte self-describing prologue (its size
// class, cell count, cells base, class index, and a 1-bit-per-cell mark bitmap); cells
// begin at prologue end. Large objects are a direct span-aligned mmap with the same
// prologue shape. `gc_alloc` pops a per-class freelist (clearing the reused cell's
// next-pointer to keep the zero-on-alloc invariant), refills by mmapping+carving a fresh
// span when the freelist is empty, and runs a synchronous stop-the-world collection once
// an allocation watermark is crossed. The collector walks the FP chain enumerating each
// frame's roots by PRECISE stack-map lookup (a per-fn table names the managed cells;
// frames with no entry fall back to a conservative whole-frame word scan — sound because
// the spill-everything frame puts every live reference in a frame cell), traces marked
// objects to a fixpoint over a malloc-backed mark stack, and linear-sweeps every span
// (dead cells → freelists, mark bits cleared). All emitted bodies are PURE fixed AArch64
// (only the constant CB address / mmap+malloc flags / lengths are baked), so the bytes
// are byte-identical at any -jN; the runtime span/mark-stack addresses never touch the
// image.

const cb_addr_hw: u16 = 0x3000; // CB_ADDR = 0x3000 << 32 (48 TiB, image-far, empty mid-VA)
const cb_addr_lsl: u2 = 2; // lsl #32

// CB header field offsets (all 8-byte). `span_count` sits at +0 so `gc_span_count` reads
// it there. `scratch_a`/`scratch_b` hold collector temporaries live across malloc/realloc.
const off_span_count: u32 = 0;
const off_collections: u32 = 8;
const off_watermark: u32 = 16;
const off_span_head: u32 = 24;
const off_large_head: u32 = 32;
const off_ms_ptr: u32 = 40;
const off_ms_len: u32 = 48;
const off_ms_cap: u32 = 56;
const off_freelist: u32 = 64; // freelist_heads[0..12] → +64..+160
const off_scratch_a: u32 = 168;
const off_scratch_b: u32 = 176;

// Class-span prologue field offsets (size_class != 0 marks a class span).
const sp_next: u32 = 0;
const sp_class: u32 = 8;
const sp_cellcount: u32 = 16;
const sp_cellsbase: u32 = 24;
const sp_classidx: u32 = 32;
const sp_bitmap: u32 = 40; // 1 bit/cell; worst case (class 16 → 1008 cells) 126 B ≤ 216

// Large-object prologue field offsets (size_class == 0 marks a large object).
const lg_next: u32 = 0;
const lg_reqsize: u32 = 16;
const lg_objbase: u32 = 24;
const lg_maplen: u32 = 32;
const lg_mark: u32 = 40;

const prologue: u12 = 256; // per-span self-describing header; cells begin here
const span: u16 = 0x4000; // 16 KiB span
const cells_bytes: u16 = 0x3F00; // span - prologue = 16128 bytes usable for cells
const prot_rw: u16 = 3; // PROT_READ | PROT_WRITE
const map_anon_priv: u16 = 0x1002; // MAP_ANON | MAP_PRIVATE
const map_anon_priv_fixed: u16 = 0x1012; // MAP_ANON | MAP_PRIVATE | MAP_FIXED
const large_threshold: u16 = 0x2000; // 8 KiB: requests above this bypass the span arena
const ms_async: u16 = 1; // MS_ASYNC
const ms_init: u16 = 4096; // initial mark-stack capacity (entries; 16 B each)

// Size classes (index 0..12). 48/96/192 are non-pow2 → normalize via real udiv/mul.
// The taxonomy lives in `Abi` (the descriptor + the allocator share one authority).
const size_classes = Abi.size_classes;

/// Ensure the control-block page exists, leaving CB_ADDR in x9 on exit. The page is
/// metadata-only now (it no longer doubles as span 0); a fresh MAP_FIXED page is zeroed,
/// so every CB field starts 0 and nothing is seeded. Clobbers x0-x5, x9, x16; emits
/// `.import` relocs for `msync` and `mmap` (auto-joining the dedup+sorted GOT set).
fn emitLocateCb(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator) error{OutOfMemory}!void {
    const A = Aarch64;
    const emit = emitWord;

    // msync(CB_ADDR, 1, MS_ASYNC): 0 → page mapped (already initialized) → skip mmap.
    try emit(code, gpa, A.movz(0, cb_addr_hw, cb_addr_lsl));
    try emit(code, gpa, A.movz(1, 1, 0));
    try emit(code, gpa, A.movz(2, ms_async, 0));
    try emitImportPreamble(code, relocs, gpa, "msync", 16);
    try emit(code, gpa, A.blr(16));
    const cbz_site: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbz(0, 0)); // mapped → skip mmap (backpatched below)

    // mmap(CB_ADDR, SPAN, RW, ANON|PRIVATE|FIXED, -1, 0): a zeroed page at CB_ADDR.
    try emit(code, gpa, A.movz(0, cb_addr_hw, cb_addr_lsl));
    try emit(code, gpa, A.movz(1, span, 0));
    try emit(code, gpa, A.movz(2, prot_rw, 0));
    try emit(code, gpa, A.movz(3, map_anon_priv_fixed, 0));
    try emit(code, gpa, A.movn(4, 0, 0)); // fd = -1
    try emit(code, gpa, A.movz(5, 0, 0));
    try emitImportPreamble(code, relocs, gpa, "mmap", 16);
    try emit(code, gpa, A.blr(16));

    const mapped: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_site, mapped);
    try emit(code, gpa, A.movz(9, cb_addr_hw, cb_addr_lsl)); // x9 = CB_ADDR (both paths)
}

/// Build the `gc_alloc(size)` builtin: returns a fresh zeroed cell for the request.
/// Small requests round up to a size class and pop that class's freelist (clearing the
/// reused cell's next-pointer to preserve zero-on-alloc); an empty freelist past the
/// growth watermark triggers a collection, else mmaps + carves a fresh span. Requests
/// above 8 KiB get a direct span-aligned mmap (large object). Caller owns the result.
pub fn lowerGcAlloc(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    // Frame + 32-byte scratch: [sp+0]=size, [sp+8]=class, [sp+16]=idx (survive calls).
    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 32));
    try emit(&code, gpa, A.strSp(0, 0)); // save size (calls clobber x0-x18)

    try emitLocateCb(&code, &relocs, gpa); // x9 = CB
    try emit(&code, gpa, A.ldrSp(0, 0)); // x0 = size

    // Large object (> 8 KiB): a direct span-aligned mmap, bypassing the class arena.
    try emit(&code, gpa, A.movz(13, large_threshold, 0));
    try emit(&code, gpa, A.cmpReg(0, 13));
    const to_small: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.ls, 0)); // size <= 8192 → small path

    // Large objects carry no freelist, so — unlike the small path, which collects only
    // when its class freelist is also empty — the sole reclaim trigger for the large arena
    // is the growth watermark: cross it and collect first (the sweep munmaps every dead
    // large span), then map the fresh object. Without this, a pure large-object churn never
    // fires a collection and leaks unbounded. The collection precedes the mmap, so the new
    // object is never at risk. The call clobbers x0-x18, so size/CB are reloaded after.
    try emit(&code, gpa, A.ldrRegUoff(3, 9, off_watermark));
    try emit(&code, gpa, A.movz(8, 4, 1)); // WATERMARK = 0x40000
    try emit(&code, gpa, A.cmpReg(3, 8));
    const lg_skip_collect: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.lo, 0)); // not crossed → skip
    try emitBuiltinCall(&code, &relocs, gpa, "gc_collect");
    const lg_after_collect: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, lg_skip_collect, lg_after_collect);
    try emit(&code, gpa, A.movz(9, cb_addr_hw, cb_addr_lsl)); // reload CB
    try emit(&code, gpa, A.ldrSp(0, 0)); // reload size

    try emit(&code, gpa, A.addImm(1, 0, prologue)); // 256 + size
    try emit(&code, gpa, A.movz(14, 0x3FFF, 0));
    try emit(&code, gpa, A.addReg(1, 1, 14)); // + (SPAN-1)
    try emit(&code, gpa, A.movz(15, 14, 0)); // shift = 14
    try emit(&code, gpa, A.lsrv(1, 1, 15));
    try emit(&code, gpa, A.lslv(1, 1, 15)); // len = roundUp16K(256 + size)
    try emit(&code, gpa, A.strSp(1, 8)); // save map_len
    try emit(&code, gpa, A.movz(0, 0, 0)); // addr = 0 (kernel-chosen)
    try emit(&code, gpa, A.movz(2, prot_rw, 0));
    try emit(&code, gpa, A.movz(3, map_anon_priv, 0));
    try emit(&code, gpa, A.movn(4, 0, 0)); // fd = -1
    try emit(&code, gpa, A.movz(5, 0, 0));
    try emitImportPreamble(&code, &relocs, gpa, "mmap", 16);
    try emit(&code, gpa, A.blr(16)); // x0 = base
    try emit(&code, gpa, A.movz(9, cb_addr_hw, cb_addr_lsl)); // x9 = CB
    try emit(&code, gpa, A.ldrRegUoff(10, 9, off_large_head));
    try emit(&code, gpa, A.strRegUoff(10, 0, lg_next)); // base.next = large_head
    try emit(&code, gpa, A.strRegUoff(0, 9, off_large_head)); // large_head = base
    try emit(&code, gpa, A.strRegUoff(A.XZR, 0, sp_class)); // size_class = 0 (large marker)
    try emit(&code, gpa, A.ldrSp(11, 0)); // req_size
    try emit(&code, gpa, A.strRegUoff(11, 0, lg_reqsize));
    try emit(&code, gpa, A.addImm(12, 0, prologue)); // objbase = base + 256
    try emit(&code, gpa, A.strRegUoff(12, 0, lg_objbase));
    try emit(&code, gpa, A.ldrSp(13, 8)); // map_len
    try emit(&code, gpa, A.strRegUoff(13, 0, lg_maplen));
    try emit(&code, gpa, A.strRegUoff(A.XZR, 0, lg_mark));
    try emit(&code, gpa, A.ldrRegUoff(14, 9, off_span_count));
    try emit(&code, gpa, A.addImm(14, 14, 1));
    try emit(&code, gpa, A.strRegUoff(14, 9, off_span_count)); // span_count++
    try emit(&code, gpa, A.ldrRegUoff(15, 9, off_watermark));
    try emit(&code, gpa, A.addReg(15, 15, 11));
    try emit(&code, gpa, A.strRegUoff(15, 9, off_watermark)); // watermark += req_size
    try emit(&code, gpa, A.movReg(0, 12)); // return objbase
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 32));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    const small: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, to_small, small);

    // Small path: round size (x0) up to a class → x1=class, x2=idx via a forward ladder.
    try emit(&code, gpa, A.ldrSp(0, 0)); // x0 = size
    var ladder_sites: [size_classes.len]u32 = undefined;
    for (size_classes, 0..) |cls, i| {
        try emit(&code, gpa, A.movz(1, cls, 0)); // candidate class
        try emit(&code, gpa, A.movz(2, @intCast(i), 0)); // candidate idx
        try emit(&code, gpa, A.cmpReg(0, 1));
        ladder_sites[i] = @intCast(code.items.len);
        try emit(&code, gpa, A.bCond(.ls, 0)); // size <= class → keep x1/x2, done
    }
    const ladder_done: u32 = @intCast(code.items.len);
    for (ladder_sites) |s| patchBCondTo(code.items, s, ladder_done);

    try emit(&code, gpa, A.strSp(1, 8)); // save class
    try emit(&code, gpa, A.strSp(2, 16)); // save idx
    try emit(&code, gpa, A.ldrRegUoff(3, 9, off_watermark));
    try emit(&code, gpa, A.addReg(3, 3, 1));
    try emit(&code, gpa, A.strRegUoff(3, 9, off_watermark)); // watermark += class

    // freelist slot: x4 = CB + idx*8; head at [x4, #off_freelist].
    try emit(&code, gpa, A.movz(5, 3, 0));
    try emit(&code, gpa, A.lslv(6, 2, 5)); // idx*8
    try emit(&code, gpa, A.addReg(4, 9, 6));
    try emit(&code, gpa, A.ldrRegUoff(7, 4, off_freelist)); // head
    const cbnz_pop: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(7, 0)); // head != 0 → pop

    // Empty freelist: collect only if the growth watermark is crossed.
    try emit(&code, gpa, A.ldrRegUoff(3, 9, off_watermark));
    try emit(&code, gpa, A.movz(8, 4, 1)); // WATERMARK = 0x40000
    try emit(&code, gpa, A.cmpReg(3, 8));
    const to_refill_a: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.lo, 0)); // watermark not crossed → refill
    try emitBuiltinCall(&code, &relocs, gpa, "gc_collect");
    // Reload CB/idx/slot/head (the call clobbered x0-x18).
    try emit(&code, gpa, A.movz(9, cb_addr_hw, cb_addr_lsl));
    try emit(&code, gpa, A.ldrSp(2, 16)); // idx
    try emit(&code, gpa, A.movz(5, 3, 0));
    try emit(&code, gpa, A.lslv(6, 2, 5));
    try emit(&code, gpa, A.addReg(4, 9, 6));
    try emit(&code, gpa, A.ldrRegUoff(7, 4, off_freelist)); // head after collect
    const to_refill_b: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(7, 0)); // still empty → refill

    // pop: freelist_heads[idx] = *head; *head = 0 (zero-on-alloc); return head.
    const pop: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbnz_pop, pop); // (cbnz reuses the cbz patcher: opcode+rt preserved)
    try emit(&code, gpa, A.ldrRegUoff(10, 7, 0));
    try emit(&code, gpa, A.strRegUoff(10, 4, off_freelist));
    try emit(&code, gpa, A.strRegUoff(A.XZR, 7, 0));
    try emit(&code, gpa, A.movReg(0, 7));
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 32));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    // refill: mmap a fresh span, write its prologue, carve its cells onto the freelist.
    const refill: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, to_refill_a, refill);
    patchCbzTo(code.items, to_refill_b, refill);
    try emit(&code, gpa, A.movz(0, 0, 0));
    try emit(&code, gpa, A.movz(1, span, 0));
    try emit(&code, gpa, A.movz(2, prot_rw, 0));
    try emit(&code, gpa, A.movz(3, map_anon_priv, 0));
    try emit(&code, gpa, A.movn(4, 0, 0));
    try emit(&code, gpa, A.movz(5, 0, 0));
    try emitImportPreamble(&code, &relocs, gpa, "mmap", 16);
    try emit(&code, gpa, A.blr(16)); // x0 = span base
    try emit(&code, gpa, A.movz(9, cb_addr_hw, cb_addr_lsl)); // x9 = CB
    try emit(&code, gpa, A.ldrSp(1, 8)); // class
    try emit(&code, gpa, A.ldrSp(2, 16)); // idx
    try emit(&code, gpa, A.ldrRegUoff(10, 9, off_span_head));
    try emit(&code, gpa, A.strRegUoff(10, 0, sp_next)); // base.next = span_head
    try emit(&code, gpa, A.strRegUoff(0, 9, off_span_head)); // span_head = base
    try emit(&code, gpa, A.strRegUoff(1, 0, sp_class));
    try emit(&code, gpa, A.movz(11, cells_bytes, 0));
    try emit(&code, gpa, A.udiv(12, 11, 1)); // cell_count = 16128 / class
    try emit(&code, gpa, A.strRegUoff(12, 0, sp_cellcount));
    try emit(&code, gpa, A.addImm(13, 0, prologue)); // cells_base = base + 256
    try emit(&code, gpa, A.strRegUoff(13, 0, sp_cellsbase));
    try emit(&code, gpa, A.strRegUoff(2, 0, sp_classidx));
    try emit(&code, gpa, A.ldrRegUoff(14, 9, off_span_count));
    try emit(&code, gpa, A.addImm(14, 14, 1));
    try emit(&code, gpa, A.strRegUoff(14, 9, off_span_count)); // span_count++
    try emit(&code, gpa, A.movz(5, 3, 0));
    try emit(&code, gpa, A.lslv(6, 2, 5));
    try emit(&code, gpa, A.addReg(4, 9, 6)); // freelist slot base
    try emit(&code, gpa, A.movz(7, 0, 0)); // i = 0
    const carve_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cmpReg(7, 12));
    const carve_exit: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0)); // i >= cell_count → carve_done
    try emit(&code, gpa, A.mul(8, 7, 1)); // i*class
    try emit(&code, gpa, A.addReg(8, 13, 8)); // cell = cells_base + i*class
    try emit(&code, gpa, A.ldrRegUoff(10, 4, off_freelist));
    try emit(&code, gpa, A.strRegUoff(10, 8, 0)); // cell.next = head
    try emit(&code, gpa, A.strRegUoff(8, 4, off_freelist)); // head = cell
    try emit(&code, gpa, A.addImm(7, 7, 1));
    const carve_back: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → carve_top
    const carve_done: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, carve_exit, carve_done);
    patchBTo(code.items, carve_back, carve_top);

    // pop one carved cell.
    try emit(&code, gpa, A.ldrRegUoff(7, 4, off_freelist));
    try emit(&code, gpa, A.ldrRegUoff(10, 7, 0));
    try emit(&code, gpa, A.strRegUoff(10, 4, off_freelist));
    try emit(&code, gpa, A.strRegUoff(A.XZR, 7, 0));
    try emit(&code, gpa, A.movReg(0, 7));
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 32));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "gc_alloc");
}

/// Emit, inline, the conservative "mark one candidate word" step: given the candidate
/// pointer in x10 and CB in x19, round it down to a 16 KiB span base and look that base
/// up in the class-span list (then the large-object list). On a hit, normalize to the
/// enclosing cell/payload, set its mark bit, and — if newly set AND `push` — push it
/// (carrying x24 as its descriptor). A leaf mark (`push == false`) sets only the mark
/// bit: the cell survives the sweep but is never traced, so a backing array's scattered
/// interior pointers are reached only through an explicit container walk, not a rescan.
/// Clobbers x0-x16; preserves x17-x28. Every branch is a local backpatched jump.
fn emitConservativeMark(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, push: bool) error{OutOfMemory}!void {
    const A = Aarch64;
    const emit = emitWord;

    try emit(code, gpa, A.movz(11, 0x3FFF, 0));
    try emit(code, gpa, A.mvn(12, 11));
    try emit(code, gpa, A.andReg(13, 10, 12)); // cand_base = word & ~0x3FFF

    try emit(code, gpa, A.ldrRegUoff(14, 19, off_span_head)); // node
    const span_top: u32 = @intCast(code.items.len);
    const cbz_large: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbz(14, 0)); // end of class spans → try large
    try emit(code, gpa, A.cmpReg(14, 13));
    const beq_found: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.eq, 0)); // node == cand_base → span_found
    try emit(code, gpa, A.ldrRegUoff(14, 14, sp_next));
    const b_span_top: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.b(0));

    const span_found: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, beq_found, span_found);
    try emit(code, gpa, A.ldrRegUoff(15, 14, sp_cellsbase)); // cells_base
    try emit(code, gpa, A.cmpReg(10, 15));
    const blo_done_a: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.lo, 0)); // word < cells_base (prologue) → done
    try emit(code, gpa, A.subReg(0, 10, 15)); // off
    try emit(code, gpa, A.ldrRegUoff(1, 14, sp_class));
    try emit(code, gpa, A.udiv(2, 0, 1)); // idx = off / class
    try emit(code, gpa, A.ldrRegUoff(3, 14, sp_cellcount));
    try emit(code, gpa, A.cmpReg(2, 3));
    const bhs_done_a: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.hs, 0)); // idx >= cell_count → done
    try emit(code, gpa, A.mul(4, 2, 1));
    try emit(code, gpa, A.addReg(5, 15, 4)); // cell = cells_base + idx*class
    try emit(code, gpa, A.addImm(6, 14, sp_bitmap));
    try emit(code, gpa, A.movz(7, 3, 0));
    try emit(code, gpa, A.lsrv(8, 2, 7)); // idx/8
    try emit(code, gpa, A.addReg(8, 6, 8)); // byte addr
    try emit(code, gpa, A.ldrbRegUoff(9, 8, 0));
    try emit(code, gpa, A.andLowBits(0, 2, 2)); // idx & 7
    try emit(code, gpa, A.movz(1, 1, 0));
    try emit(code, gpa, A.lslv(1, 1, 0)); // mask = 1 << (idx&7)
    try emit(code, gpa, A.andReg(0, 9, 1));
    const cbnz_done_a: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbnz(0, 0)); // already marked → done
    try emit(code, gpa, A.orrReg(9, 9, 1));
    try emit(code, gpa, A.strb(9, 8, 0)); // set bit
    if (push) {
        try emit(code, gpa, A.movReg(10, 5)); // push cell
        try emitPush(code, relocs, gpa);
    }
    const b_done_a: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.b(0)); // → done

    const try_large: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_large, try_large);
    try emit(code, gpa, A.ldrRegUoff(14, 19, off_large_head));
    const large_top: u32 = @intCast(code.items.len);
    const cbz_done_b: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbz(14, 0)); // end of large list → done
    try emit(code, gpa, A.ldrRegUoff(0, 14, lg_objbase));
    try emit(code, gpa, A.cmpReg(10, 0));
    const blo_next: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.lo, 0)); // word < objbase → next
    try emit(code, gpa, A.ldrRegUoff(1, 14, lg_reqsize));
    try emit(code, gpa, A.addReg(2, 0, 1)); // end
    try emit(code, gpa, A.cmpReg(10, 2));
    const bhs_next: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.hs, 0)); // word >= end → next
    try emit(code, gpa, A.ldrRegUoff(3, 14, lg_mark));
    const cbnz_done_c: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.cbnz(3, 0)); // already marked → done
    try emit(code, gpa, A.movz(4, 1, 0));
    try emit(code, gpa, A.strRegUoff(4, 14, lg_mark));
    if (push) {
        try emit(code, gpa, A.movReg(10, 0)); // push objbase
        try emitPush(code, relocs, gpa);
    }
    const b_done_c: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.b(0)); // → done
    const large_next: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, blo_next, large_next);
    patchBCondTo(code.items, bhs_next, large_next);
    try emit(code, gpa, A.ldrRegUoff(14, 14, lg_next));
    const b_large_top: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.b(0)); // → large_top

    const done: u32 = @intCast(code.items.len);
    patchBTo(code.items, b_span_top, span_top);
    patchBCondTo(code.items, blo_done_a, done);
    patchBCondTo(code.items, bhs_done_a, done);
    patchCbzTo(code.items, cbnz_done_a, done);
    patchBTo(code.items, b_done_a, done);
    patchCbzTo(code.items, cbz_done_b, done);
    patchCbzTo(code.items, cbnz_done_c, done);
    patchBTo(code.items, b_done_c, done);
    patchBTo(code.items, b_large_top, large_top);
}

/// Emit, inline, a push of the pointer in x10 onto the malloc-backed mark stack (16-byte
/// (ptr, descriptor) pairs; the descriptor half takes x24 — the pending tag/descriptor the
/// caller set for this candidate, 0 for a conservative word-scan on pop). Doubles the
/// backing via `realloc` when full, preserving the pushed ptr + new capacity in CB scratch
/// across the call. Clobbers x0-x2, x11-x14, x16; preserves x17-x28 (and x10 across the
/// realloc).
fn emitPush(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator) error{OutOfMemory}!void {
    const A = Aarch64;
    const emit = emitWord;

    try emit(code, gpa, A.ldrRegUoff(11, 19, off_ms_len));
    try emit(code, gpa, A.ldrRegUoff(12, 19, off_ms_cap));
    try emit(code, gpa, A.cmpReg(11, 12));
    const blo_store: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.bCond(.lo, 0)); // len < cap → no grow

    try emit(code, gpa, A.strRegUoff(10, 19, off_scratch_a)); // save ptr
    try emit(code, gpa, A.movz(13, 1, 0));
    try emit(code, gpa, A.lslv(13, 12, 13)); // new_cap = cap*2
    try emit(code, gpa, A.strRegUoff(13, 19, off_scratch_b)); // save new_cap
    try emit(code, gpa, A.ldrRegUoff(0, 19, off_ms_ptr));
    try emit(code, gpa, A.movz(1, 4, 0));
    try emit(code, gpa, A.lslv(1, 13, 1)); // bytes = new_cap*16
    try emitImportPreamble(code, relocs, gpa, "realloc", 16);
    try emit(code, gpa, A.blr(16)); // x0 = new ptr
    try emit(code, gpa, A.strRegUoff(0, 19, off_ms_ptr));
    try emit(code, gpa, A.ldrRegUoff(13, 19, off_scratch_b));
    try emit(code, gpa, A.strRegUoff(13, 19, off_ms_cap));
    try emit(code, gpa, A.ldrRegUoff(10, 19, off_scratch_a)); // restore ptr
    try emit(code, gpa, A.ldrRegUoff(11, 19, off_ms_len));

    const store: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, blo_store, store);
    try emit(code, gpa, A.ldrRegUoff(0, 19, off_ms_ptr));
    try emit(code, gpa, A.movz(1, 4, 0));
    try emit(code, gpa, A.lslv(1, 11, 1)); // len*16
    try emit(code, gpa, A.addReg(2, 0, 1));
    try emit(code, gpa, A.strRegUoff(10, 2, 0)); // slot.ptr
    try emit(code, gpa, A.strRegUoff(24, 2, 8)); // slot.descriptor = x24
    try emit(code, gpa, A.addImm(11, 11, 1));
    try emit(code, gpa, A.strRegUoff(11, 19, off_ms_len)); // ms_len++
}

/// Build the `gc_mark(candidate)` builtin: mark ONE candidate pointer (x0) and push it for a
/// conservative body scan, reusing the collector's own `emitConservativeMark`/`emitPush`
/// with descriptor 0. It is the callable wrapper a trace unit `bl`s per managed reference:
/// an erased `Ref`/`gc_array` element trace marks its pointee/header through it, so the
/// pointee is scanned in turn (a sound superset — a leaf pointee scans to a no-op). Caller
/// owns the result.
pub fn lowerGcMark(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    // x19 is callee-saved (emitConservativeMark reads CB from it) and x0 is clobbered by
    // emitLocateCb, so both are stashed in a 16-byte scratch frame across the setup.
    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 16));
    try emit(&code, gpa, A.strSp(19, 0)); // preserve caller's x19
    try emit(&code, gpa, A.strSp(0, 8)); // save candidate (emitLocateCb clobbers x0)

    try emitLocateCb(&code, &relocs, gpa); // x9 = CB
    try emit(&code, gpa, A.movReg(19, 9)); // x19 = CB (the reg emitConservativeMark reads)
    try emit(&code, gpa, A.ldrSp(10, 8)); // x10 = candidate (the reg it marks)
    try emit(&code, gpa, A.movz(24, 0, 0)); // descriptor = 0 → the pop word-scans conservatively
    try emitConservativeMark(&code, &relocs, gpa, true);

    try emit(&code, gpa, A.ldrSp(19, 0)); // restore caller's x19
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 16));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "gc_mark");
}

/// Build the `gc_mark_leaf(candidate)` builtin: set ONLY the mark bit of the candidate's
/// cell (x0) — no push, so the sweep spares it but the collector never scans its body. The
/// container traces leaf-mark their backing arrays with this: a backing whose interior
/// holds scattered element/key pointers is kept alive, yet those pointers are reached only
/// through the container's explicit dense-element walk (never a whole-backing rescan), which
/// is what makes that walk load-bearing. Same frame/locate-CB shape as `gc_mark`. Caller
/// owns the result.
pub fn lowerGcMarkLeaf(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 16));
    try emit(&code, gpa, A.strSp(19, 0)); // preserve caller's x19
    try emit(&code, gpa, A.strSp(0, 8)); // save candidate (emitLocateCb clobbers x0)

    try emitLocateCb(&code, &relocs, gpa); // x9 = CB
    try emit(&code, gpa, A.movReg(19, 9)); // x19 = CB
    try emit(&code, gpa, A.ldrSp(10, 8)); // x10 = candidate
    try emitConservativeMark(&code, &relocs, gpa, false); // mark bit only, no push

    try emit(&code, gpa, A.ldrSp(19, 0)); // restore caller's x19
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 16));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "gc_mark_leaf");
}

/// Build the `gc_collect()` builtin: a synchronous stop-the-world mark-sweep with a HYBRID
/// root scan. Walks the x29 chain from its own frame outward; for each parent frame it
/// identifies the enclosing fn by the child-saved return address and looks that fn's precise
/// root map up in the address-keyed stack-map table (`text_off -> root bitmap`), enumerating
/// only the managed cells the map names. A frame with no map entry (a hand-emitted builtin,
/// a failed lower, or a foreign C-runtime frame — all carrying the `0xFFFFFFFF` conservative
/// marker) falls back to a whole-frame word scan. Precise roots are always a subset of that
/// whole-frame set, and the marker validates each candidate before any dereference, so a
/// misidentified frame over-retains at worst — never a use-after-free. OBJECT TRACING is
/// descriptor-driven where a descriptor rode the push: on pop, a container (descriptor 1)
/// dispatches to `ga_trace`/`mp_trace` (leaf-mark the backings, trace the live elements
/// through their stashed element/K/V descriptors); everything else (descriptor 0) is
/// word-scanned conservatively. Marked objects are traced to a fixpoint over the mark stack,
/// then every class span is linear-swept (dead cells → per-class freelists, mark bits
/// cleared). Large objects leak (their mark is just reset). Caller owns the result.
///
/// Register contract: x19=CB, x24=pending descriptor for the next push, x25=text_base,
/// x26=&stackmap, x27=cur fp, x28=parent fp, x20/x21/x22=scan/enumeration cursors,
/// x23=corruption cap — all callee-saved, saved/restored so the collector honors the ABI
/// toward the allocating caller.
pub fn lowerGcCollect(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    try emitFramePrologue(&code, gpa);
    try emit(&code, gpa, A.subImm(A.SP, A.SP, 80)); // save area for x19-x28
    try emit(&code, gpa, A.strSp(19, 0));
    try emit(&code, gpa, A.strSp(20, 8));
    try emit(&code, gpa, A.strSp(21, 16));
    try emit(&code, gpa, A.strSp(22, 24));
    try emit(&code, gpa, A.strSp(23, 32));
    try emit(&code, gpa, A.strSp(24, 40)); // x24 = pending push descriptor (root tag / 0)
    try emit(&code, gpa, A.strSp(25, 48));
    try emit(&code, gpa, A.strSp(26, 56));
    try emit(&code, gpa, A.strSp(27, 64));
    try emit(&code, gpa, A.strSp(28, 72));

    try emitLocateCb(&code, &relocs, gpa); // x9 = CB
    try emit(&code, gpa, A.movReg(19, 9)); // x19 = CB

    // Self-locate text_base into x25, and load the GC stack-map table base into x26 via the
    // reserved `.cstr` hash (the exact adrp+add path `lowerPanic` uses for the symtab). Both
    // are callee-saved, surviving every malloc/realloc/munmap the collector calls.
    try emitTextBase(&code, &relocs, gpa, 25);
    {
        const site_a: u32 = @intCast(code.items.len);
        try relocs.append(gpa, .{ .site = site_a, .target = .{ .cstr = Link.gc_stackmap_base_hash }, .kind = .adrp_page });
        try emit(&code, gpa, A.adrp(26, 0)); // adrp x26, stackmap@page
        const site_b: u32 = @intCast(code.items.len);
        try relocs.append(gpa, .{ .site = site_b, .target = .{ .cstr = Link.gc_stackmap_base_hash }, .kind = .add_lo12 });
        try emit(&code, gpa, A.addImm(26, 26, 0)); // x26 = &stackmap
    }

    // Ensure the mark stack is allocated (ms_cap == 0 on the first collection).
    try emit(&code, gpa, A.ldrRegUoff(0, 19, off_ms_cap));
    const cbnz_ms_ready: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(0, 0));
    try emit(&code, gpa, A.movz(0, 1, 1)); // MS_INIT*16 = 0x10000
    try emitImportPreamble(&code, &relocs, gpa, "malloc", 16);
    try emit(&code, gpa, A.blr(16));
    try emit(&code, gpa, A.strRegUoff(0, 19, off_ms_ptr));
    try emit(&code, gpa, A.movz(1, ms_init, 0));
    try emit(&code, gpa, A.strRegUoff(1, 19, off_ms_cap));
    const ms_ready: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbnz_ms_ready, ms_ready);
    try emit(&code, gpa, A.strRegUoff(A.XZR, 19, off_ms_len)); // ms_len = 0

    // Reset the 13 per-class freelist heads — sweep rebuilds them from scratch.
    try emit(&code, gpa, A.addImm(0, 19, off_freelist));
    try emit(&code, gpa, A.movz(1, size_classes.len, 0));
    const reset_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.strRegUoff(A.XZR, 0, 0));
    try emit(&code, gpa, A.addImm(0, 0, 8));
    try emit(&code, gpa, A.subImm(1, 1, 1));
    const cbnz_reset: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(1, 0));
    patchCbzTo(code.items, cbnz_reset, reset_top);

    // Hybrid root scan: walk the x29 chain from gc_collect's own frame outward. For each
    // parent frame, identify its fn by the return address the child saved (`*(cur+8)`, less
    // text_base, less 4 to land on the `bl`), scan the address-keyed stack map for the
    // greatest row off <= that fn offset, and enumerate ITS precise root cells at
    // `parent_sp + sp_off`. A row-less/sentinel/foreign frame carries the `0xFFFFFFFF`
    // marker → fall back to the whole-frame conservative scan (matching the old behavior for
    // builtin/C-runtime frames). The marker validates every candidate before dereferencing,
    // so a misidentified frame over-retains at worst.
    // Loop-carried state (all callee-saved, surviving emitConservativeMark/emitPush):
    //   x19=CB, x25=text_base, x26=&stackmap, x27=cur fp, x28=parent fp,
    //   x20/x21/x22=scan+enumeration cursors, x23=corruption cap.
    try emit(&code, gpa, A.movReg(27, A.FP)); // cur = gc_collect.fp (its own frame holds no roots)
    try emit(&code, gpa, A.movz(23, 4096, 0)); // corruption cap
    const walk_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.ldrRegUoff(28, 27, 0)); // parent_fp = *cur
    const cbz_walk_done_a: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(28, 0)); // fp == 0 → done
    const cbz_walk_done_b: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(23, 0)); // cap exhausted → done
    try emit(&code, gpa, A.ldrRegUoff(20, 27, 8)); // ra into the PARENT's fn
    try emit(&code, gpa, A.subReg(20, 20, 25)); // - text_base
    try emit(&code, gpa, A.subImm(20, 20, 4)); // -> parent fn body offset

    // Scan the sorted table for the greatest entry off <= x20; x21 = its payload_off.
    // The first fn starts at off 0, so a match always exists.
    try emit(&code, gpa, A.ldrRegUoff(4, 26, 0)); // count
    try emit(&code, gpa, A.addImm(5, 26, 8)); // &entry[0]
    try emit(&code, gpa, A.movz(6, 0, 0)); // i
    try emit(&code, gpa, A.movz(21, 0, 0)); // best payload_off
    const scan_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cmpReg(6, 4));
    const scan_bhs: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0)); // i >= count → scan_done
    try emit(&code, gpa, A.ldrRegUoff(9, 5, 0)); // entry.off
    try emit(&code, gpa, A.cmpReg(9, 20));
    const scan_bhi: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hi, 0)); // off > target → scan_done
    try emit(&code, gpa, A.ldrRegUoff(21, 5, 8)); // best = entry.payload_off
    try emit(&code, gpa, A.addImm(5, 5, 16));
    try emit(&code, gpa, A.addImm(6, 6, 1));
    const scan_b: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → scan_top
    const scan_done: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, scan_bhs, scan_done);
    patchBCondTo(code.items, scan_bhi, scan_done);
    patchBTo(code.items, scan_b, scan_top);

    try emit(&code, gpa, A.addReg(9, 26, 21)); // payload ptr = table_base + payload_off
    try emit(&code, gpa, A.ldrwRegUoff(10, 9, 0)); // n (or 0xFFFFFFFF marker), zero-extended
    // Build 0x0000_0000_FFFF_FFFF (NOT the 64-bit movn all-ones) so it matches the
    // zero-extended marker word.
    try emit(&code, gpa, A.movz(11, 0xFFFF, 0));
    try emit(&code, gpa, A.movk(11, 0xFFFF, 1));
    try emit(&code, gpa, A.cmpReg(10, 11));
    const beq_cons: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.eq, 0)); // conservative marker → whole-frame scan

    // PRECISE: enumerate n roots at parent_sp + sp_off (region_lo = cur+16 = parent's sp).
    // Each root is an 8-byte `{u32 sp_off, u32 tag}` pair; the tag rides the push in x24 as
    // the popped candidate's descriptor (1 = gc_array container header, 0 = word-scan).
    try emit(&code, gpa, A.addImm(22, 27, 16)); // region_lo
    try emit(&code, gpa, A.addImm(20, 9, 4)); // &root[0]
    try emit(&code, gpa, A.movReg(21, 10)); // remaining = n
    const p_top: u32 = @intCast(code.items.len);
    const cbz_p_advance: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(21, 0)); // remaining == 0 → advance
    try emit(&code, gpa, A.ldrwRegUoff(12, 20, 0)); // sp_off
    try emit(&code, gpa, A.ldrwRegUoff(24, 20, 4)); // tag → x24 (descriptor for this root)
    try emit(&code, gpa, A.addReg(10, 22, 12)); // cell = region_lo + sp_off
    try emit(&code, gpa, A.ldrRegUoff(10, 10, 0)); // word
    const cbz_p_next: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(10, 0)); // null → skip
    try emitConservativeMark(&code, &relocs, gpa, true);
    const p_next: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_p_next, p_next);
    try emit(&code, gpa, A.addImm(20, 20, 8)); // next {sp_off,tag} pair
    try emit(&code, gpa, A.subImm(21, 21, 1));
    const b_p_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → p_top
    const b_precise_advance: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → advance
    patchBTo(code.items, b_p_top, p_top);

    // CONSERVATIVE: whole-frame scan [cur+16, parent_fp).
    const cons: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, beq_cons, cons);
    try emit(&code, gpa, A.addImm(20, 27, 16)); // p = cur+16
    const c_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cmpReg(20, 28));
    const bhs_c_advance: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0)); // p >= parent_fp → advance
    try emit(&code, gpa, A.ldrRegUoff(10, 20, 0));
    const cbz_c_next: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(10, 0));
    try emit(&code, gpa, A.movz(24, 0, 0)); // conservative root → word-scan on pop
    try emitConservativeMark(&code, &relocs, gpa, true);
    const c_next: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_c_next, c_next);
    try emit(&code, gpa, A.addImm(20, 20, 8));
    const b_c_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → c_top
    patchBTo(code.items, b_c_top, c_top);

    // ADVANCE: cur = parent_fp; --cap; loop.
    const advance: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_p_advance, advance);
    patchBTo(code.items, b_precise_advance, advance);
    patchBCondTo(code.items, bhs_c_advance, advance);
    try emit(&code, gpa, A.movReg(27, 28)); // cur = parent_fp
    try emit(&code, gpa, A.subImm(23, 23, 1)); // --cap
    const b_walk_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → walk_top
    patchBTo(code.items, b_walk_top, walk_top);
    const walk_done: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_walk_done_a, walk_done);
    patchCbzTo(code.items, cbz_walk_done_b, walk_done);

    // Trace to fixpoint: pop a ptr, find its size, mark every nonzero word it holds.
    const trace_pop: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.ldrRegUoff(11, 19, off_ms_len));
    const cbz_trace_done: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(11, 0));
    try emit(&code, gpa, A.subImm(11, 11, 1));
    try emit(&code, gpa, A.strRegUoff(11, 19, off_ms_len)); // pop
    try emit(&code, gpa, A.ldrRegUoff(0, 19, off_ms_ptr));
    try emit(&code, gpa, A.movz(1, 4, 0));
    try emit(&code, gpa, A.lslv(2, 11, 1));
    try emit(&code, gpa, A.addReg(3, 0, 2));
    try emit(&code, gpa, A.ldrRegUoff(22, 3, 0)); // popped ptr
    // The entry's descriptor half selects the trace strategy. desc == 1 → a gc_array
    // container header: read its self-describing `shape@32` (0 = Vec → ga_trace, else
    // Map/Set → mp_trace) and run the precise walk, then loop. Any other desc (0 today)
    // falls through to the conservative word-scan below.
    try emit(&code, gpa, A.ldrRegUoff(4, 3, 8)); // desc
    try emit(&code, gpa, A.subImm(4, 4, 1));
    const cbnz_not_container: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(4, 0)); // desc != 1 → conservative
    try emit(&code, gpa, A.ldrRegUoff(5, 22, 32)); // shape
    try emit(&code, gpa, A.movReg(0, 22)); // x0 = header
    const cbnz_map: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(5, 0)); // shape != 0 → mp_trace
    try emitBuiltinCall(&code, &relocs, gpa, "ga_trace");
    const b_after_ga: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → trace_pop
    const map_call: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbnz_map, map_call);
    try emitBuiltinCall(&code, &relocs, gpa, "mp_trace");
    const b_after_mp: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → trace_pop
    patchBTo(code.items, b_after_ga, trace_pop);
    patchBTo(code.items, b_after_mp, trace_pop);
    const cons_scan: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbnz_not_container, cons_scan);
    try emit(&code, gpa, A.movz(4, 0x3FFF, 0));
    try emit(&code, gpa, A.mvn(5, 4));
    try emit(&code, gpa, A.andReg(6, 22, 5)); // cand_base
    try emit(&code, gpa, A.ldrRegUoff(7, 19, off_span_head));
    const tp_span_top: u32 = @intCast(code.items.len);
    const cbz_tp_large: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(7, 0));
    try emit(&code, gpa, A.cmpReg(7, 6));
    const beq_tp_span: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.eq, 0));
    try emit(&code, gpa, A.ldrRegUoff(7, 7, sp_next));
    const b_tp_span: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_tp_span, tp_span_top);
    const tp_span_found: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, beq_tp_span, tp_span_found);
    try emit(&code, gpa, A.ldrRegUoff(8, 7, sp_class));
    try emit(&code, gpa, A.addReg(23, 22, 8)); // end = ptr + size_class
    const b_tp_scan: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0)); // → tp_scan
    const tp_try_large: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_tp_large, tp_try_large);
    try emit(&code, gpa, A.ldrRegUoff(7, 19, off_large_head));
    const tp_large_top: u32 = @intCast(code.items.len);
    const cbz_tp_pop: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(7, 0)); // not found → pop next
    try emit(&code, gpa, A.cmpReg(7, 6));
    const beq_tp_large: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.eq, 0));
    try emit(&code, gpa, A.ldrRegUoff(7, 7, lg_next));
    const b_tp_large: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_tp_large, tp_large_top);
    const tp_large_found: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, beq_tp_large, tp_large_found);
    try emit(&code, gpa, A.ldrRegUoff(8, 7, lg_reqsize));
    try emit(&code, gpa, A.addReg(23, 22, 8)); // end = ptr + req_size
    const tp_scan: u32 = @intCast(code.items.len);
    patchBTo(code.items, b_tp_scan, tp_scan);
    try emit(&code, gpa, A.cmpReg(22, 23));
    const bhs_tp_pop: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0)); // scanned whole object → pop next
    try emit(&code, gpa, A.ldrRegUoff(10, 22, 0));
    const cbz_tp_next: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(10, 0));
    try emit(&code, gpa, A.movz(24, 0, 0)); // sub-object word found conservatively → word-scan on pop
    try emitConservativeMark(&code, &relocs, gpa, true);
    const tp_scan_next: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_tp_next, tp_scan_next);
    try emit(&code, gpa, A.addImm(22, 22, 8));
    const b_tp_scan2: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_tp_scan2, tp_scan);
    patchCbzTo(code.items, cbz_tp_pop, trace_pop);
    patchBCondTo(code.items, bhs_tp_pop, trace_pop);
    // trace_done target:
    const trace_done: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_trace_done, trace_done);

    // Sweep class spans: dead cells → freelists (zeroed first), marked cells → bit cleared.
    try emit(&code, gpa, A.ldrRegUoff(0, 19, off_span_head));
    const sw_span_top: u32 = @intCast(code.items.len);
    const cbz_sweep_large: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(0, 0));
    try emit(&code, gpa, A.ldrRegUoff(1, 0, sp_class));
    try emit(&code, gpa, A.ldrRegUoff(2, 0, sp_cellcount));
    try emit(&code, gpa, A.ldrRegUoff(3, 0, sp_cellsbase));
    try emit(&code, gpa, A.ldrRegUoff(4, 0, sp_classidx));
    try emit(&code, gpa, A.movz(5, 3, 0));
    try emit(&code, gpa, A.lslv(6, 4, 5));
    try emit(&code, gpa, A.addReg(7, 19, 6)); // freelist slot base (head at [x7,#64])
    try emit(&code, gpa, A.addImm(8, 0, sp_bitmap));
    try emit(&code, gpa, A.movz(9, 0, 0)); // i
    const sw_cell_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cmpReg(9, 2));
    const bhs_cell_done: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.bCond(.hs, 0));
    try emit(&code, gpa, A.movz(10, 3, 0));
    try emit(&code, gpa, A.lsrv(11, 9, 10)); // i/8
    try emit(&code, gpa, A.addReg(12, 8, 11)); // byte addr
    try emit(&code, gpa, A.ldrbRegUoff(13, 12, 0));
    try emit(&code, gpa, A.andLowBits(14, 9, 2)); // i & 7
    try emit(&code, gpa, A.movz(15, 1, 0));
    try emit(&code, gpa, A.lslv(15, 15, 14)); // mask
    try emit(&code, gpa, A.andReg(16, 13, 15));
    const cbnz_marked: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(16, 0)); // marked → clear bit
    // dead cell: zero the whole cell, then push onto the freelist.
    try emit(&code, gpa, A.mul(10, 9, 1));
    try emit(&code, gpa, A.addReg(11, 3, 10)); // cell
    try emit(&code, gpa, A.movz(12, 3, 0));
    try emit(&code, gpa, A.lsrv(13, 1, 12)); // words = class/8
    try emit(&code, gpa, A.movReg(14, 11));
    const zero_top: u32 = @intCast(code.items.len);
    const cbz_zero_done: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(13, 0));
    try emit(&code, gpa, A.strRegUoff(A.XZR, 14, 0));
    try emit(&code, gpa, A.addImm(14, 14, 8));
    try emit(&code, gpa, A.subImm(13, 13, 1));
    const b_zero_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_zero_top, zero_top);
    const zero_done: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_zero_done, zero_done);
    try emit(&code, gpa, A.ldrRegUoff(12, 7, off_freelist));
    try emit(&code, gpa, A.strRegUoff(12, 11, 0)); // cell.next = head
    try emit(&code, gpa, A.strRegUoff(11, 7, off_freelist)); // head = cell
    const b_cell_next: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    const sw_marked: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbnz_marked, sw_marked);
    try emit(&code, gpa, A.mvn(10, 15));
    try emit(&code, gpa, A.andReg(13, 13, 10));
    try emit(&code, gpa, A.strb(13, 12, 0)); // clear bit
    const sw_cell_next: u32 = @intCast(code.items.len);
    patchBTo(code.items, b_cell_next, sw_cell_next);
    try emit(&code, gpa, A.addImm(9, 9, 1));
    const b_cell_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_cell_top, sw_cell_top);
    const sw_cell_done: u32 = @intCast(code.items.len);
    patchBCondTo(code.items, bhs_cell_done, sw_cell_done);
    try emit(&code, gpa, A.ldrRegUoff(0, 0, sp_next));
    const b_span_top2: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_span_top2, sw_span_top);

    // Sweep large objects: marked → clear its mark; unmarked → unlink and munmap the
    // whole span (large objects live on `large_head`, not a freelist, so reclamation is a
    // direct return to the kernel). `pp` (x20) tracks the address of the link slot holding
    // `cur` (x21) so a dead node is unlinked without a trailing pointer; `next`/`maplen`
    // are read before munmap since the node's prologue vanishes with it. x19-x23 survive
    // the libSystem call by the AArch64 callee-saved convention.
    const sweep_large: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_sweep_large, sweep_large);
    try emit(&code, gpa, A.addImm(20, 19, off_large_head)); // pp = &large_head
    try emit(&code, gpa, A.ldrRegUoff(21, 20, 0)); // cur = *pp
    const sw_lg_top: u32 = @intCast(code.items.len);
    const cbz_sweep_epi: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbz(21, 0));
    try emit(&code, gpa, A.ldrRegUoff(22, 21, lg_mark));
    try emit(&code, gpa, A.ldrRegUoff(23, 21, lg_next));
    const cbnz_lg_survive: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.cbnz(22, 0)); // marked → survive
    // dead: unlink (*pp = next) then munmap(cur, cur.maplen); pp stays put.
    try emit(&code, gpa, A.strRegUoff(23, 20, 0));
    try emit(&code, gpa, A.ldrRegUoff(1, 21, lg_maplen));
    try emit(&code, gpa, A.movReg(0, 21));
    try emitImportPreamble(&code, &relocs, gpa, "munmap", 16);
    try emit(&code, gpa, A.blr(16));
    try emit(&code, gpa, A.movReg(21, 23)); // cur = next
    const b_lg_top: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    const sw_lg_survive: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbnz_lg_survive, sw_lg_survive);
    try emit(&code, gpa, A.strRegUoff(A.XZR, 21, lg_mark)); // clear mark
    try emit(&code, gpa, A.addImm(20, 21, lg_next)); // pp = &cur.next
    try emit(&code, gpa, A.movReg(21, 23)); // cur = next
    const b_lg_top2: u32 = @intCast(code.items.len);
    try emit(&code, gpa, A.b(0));
    patchBTo(code.items, b_lg_top, sw_lg_top);
    patchBTo(code.items, b_lg_top2, sw_lg_top);

    const sweep_epi: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_sweep_epi, sweep_epi);
    try emit(&code, gpa, A.ldrRegUoff(0, 19, off_collections));
    try emit(&code, gpa, A.addImm(0, 0, 1));
    try emit(&code, gpa, A.strRegUoff(0, 19, off_collections)); // collections++
    try emit(&code, gpa, A.strRegUoff(A.XZR, 19, off_watermark)); // watermark = 0

    try emit(&code, gpa, A.ldrSp(19, 0));
    try emit(&code, gpa, A.ldrSp(20, 8));
    try emit(&code, gpa, A.ldrSp(21, 16));
    try emit(&code, gpa, A.ldrSp(22, 24));
    try emit(&code, gpa, A.ldrSp(23, 32));
    try emit(&code, gpa, A.ldrSp(24, 40));
    try emit(&code, gpa, A.ldrSp(25, 48));
    try emit(&code, gpa, A.ldrSp(26, 56));
    try emit(&code, gpa, A.ldrSp(27, 64));
    try emit(&code, gpa, A.ldrSp(28, 72));
    try emit(&code, gpa, A.addImm(A.SP, A.SP, 80));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "gc_collect");
}

/// Build the `gc_span_count()` builtin: the number of spans mapped so far. Forces lazy
/// CB init if called before any allocation, so it always returns a truthful count.
pub fn lowerGcSpanCount(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    try emitFramePrologue(&code, gpa);
    try emitLocateCb(&code, &relocs, gpa);
    try emit(&code, gpa, A.ldrRegUoff(0, 9, off_span_count));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "gc_span_count");
}

/// Build the `gc_stats()` builtin: the number of collections run so far. Forces lazy CB
/// init if called before any allocation, so it always returns a truthful count.
pub fn lowerGcStats(gpa: std.mem.Allocator) error{OutOfMemory}!Link.FnCode {
    const A = Aarch64;
    const emit = emitWord;
    var code: std.ArrayList(u8) = .empty;
    errdefer code.deinit(gpa);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    errdefer deinitBuiltinRelocs(&relocs, gpa);

    try emitFramePrologue(&code, gpa);
    try emitLocateCb(&code, &relocs, gpa);
    try emit(&code, gpa, A.ldrRegUoff(0, 9, off_collections));
    try emit(&code, gpa, A.ldpFpLrPost);
    try emit(&code, gpa, A.ret);

    return finishBuiltin(&code, &relocs, gpa, "gc_stats");
}

test "gc_mark_leaf marks without pushing (no mark-stack realloc), unlike gc_mark" {
    const gpa = std.testing.allocator;
    // The push path (emitPush) is the ONLY thing that reallocs the mark stack, so a body
    // that never pushes carries no `realloc` import — the observable difference between a
    // leaf mark (bit only) and a full mark (bit + push).
    const hasRealloc = struct {
        fn f(fc: Link.FnCode) bool {
            for (fc.relocs) |r| if (r.target.name()) |nm| {
                if (std.mem.eql(u8, nm, "realloc")) return true;
            };
            return false;
        }
    }.f;

    var mark = try lowerGcMark(gpa);
    defer mark.deinit(gpa);
    var leaf = try lowerGcMarkLeaf(gpa);
    defer leaf.deinit(gpa);

    try std.testing.expect(hasRealloc(mark)); // gc_mark pushes → reallocs when full
    try std.testing.expect(!hasRealloc(leaf)); // gc_mark_leaf only sets the bit
    // The leaf body is therefore strictly smaller (it omits the whole push sequence).
    try std.testing.expect(leaf.code.len < mark.code.len);
}

test "gc_mark: names the builtin, opens a frame, ends in ret, byte-identical across relowers" {
    const gpa = testing.allocator;
    var a = try lowerGcMark(gpa);
    defer a.deinit(gpa);
    var b = try lowerGcMark(gpa);
    defer b.deinit(gpa);

    try testing.expectEqualStrings("gc_mark", a.sym.name);
    try testing.expectEqual(Link.SymKind.builtin, a.sym.kind);
    try testing.expect(a.code.len % 4 == 0);
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, a.code[0..4], .little));
    try testing.expectEqual(Aarch64.ret, std.mem.readInt(u32, a.code[a.code.len - 4 ..][0..4], .little));
    // The mark step reuses the collector's `emitPush`, whose grow path pulls `realloc`; it
    // calls no other builtin, so there is no `.call26` edge.
    var saw_realloc = false;
    for (a.relocs) |r| {
        try testing.expect(r.kind != .call26);
        if (r.target == .import and std.mem.eql(u8, r.target.import.name, "realloc")) saw_realloc = true;
    }
    try testing.expect(saw_realloc);
    try testing.expectEqualSlices(u8, a.code, b.code); // pure fixed bytes (-jN identity)
}

test "gc_alloc: names the builtin, opens a frame, and carries a bl gc_collect call reloc" {
    const gpa = testing.allocator;
    var fc = try lowerGcAlloc(gpa);
    defer fc.deinit(gpa);

    try testing.expectEqualStrings("gc_alloc", fc.sym.name);
    try testing.expectEqual(Link.SymKind.builtin, fc.sym.kind);
    try testing.expect(fc.code.len % 4 == 0);
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, fc.code[0..4], .little));

    // The alloc-triggered collection: two `.call26` relocs, both to the `gc_collect`
    // builtin — the small path's freelist-empty-past-watermark arm and the large path's
    // watermark arm (the edges the emit.zig coupling line must keep alive).
    var call26s: usize = 0;
    var saw_collect = false;
    var saw_mmap = false;
    for (fc.relocs) |r| {
        if (r.kind == .call26) {
            call26s += 1;
            try testing.expect(r.target == .func);
            try testing.expectEqual(Link.SymKind.builtin, r.target.func.kind);
            if (std.mem.eql(u8, r.target.func.name, "gc_collect")) saw_collect = true;
        }
        if (r.target == .import and std.mem.eql(u8, r.target.import.name, "mmap")) saw_mmap = true;
    }
    try testing.expectEqual(@as(usize, 2), call26s);
    try testing.expect(saw_collect);
    try testing.expect(saw_mmap);
}

test "gc_collect: names the builtin, opens a frame, ends in ret, and pulls malloc + realloc" {
    const gpa = testing.allocator;
    var fc = try lowerGcCollect(gpa);
    defer fc.deinit(gpa);

    try testing.expectEqualStrings("gc_collect", fc.sym.name);
    try testing.expectEqual(Link.SymKind.builtin, fc.sym.kind);
    try testing.expect(fc.code.len % 4 == 0);
    try testing.expectEqual(Aarch64.stpFpLrPre, std.mem.readInt(u32, fc.code[0..4], .little));
    try testing.expectEqual(Aarch64.ret, std.mem.readInt(u32, fc.code[fc.code.len - 4 ..][0..4], .little));

    // The mark stack lives on the malloc heap (off-image), so realloc-grown: both imports
    // must be minted. The pop dispatch `bl`s the container traces (`ga_trace`/`mp_trace`),
    // the only `.call26` edges out of the collector.
    var saw_malloc = false;
    var saw_realloc = false;
    var saw_ga_trace = false;
    var saw_mp_trace = false;
    // The hybrid root scan self-locates text_base (a movw_g0/movw_g1 self-relative pair)
    // and loads the stack-map table base via the reserved `.cstr` hash (an adrp_page +
    // add_lo12 pair, exactly like the panic symtab).
    var saw_movw_g0 = false;
    var saw_movw_g1 = false;
    var saw_sm_adrp = false;
    var saw_sm_add = false;
    for (fc.relocs) |r| {
        if (r.target == .func and r.kind == .call26 and std.mem.eql(u8, r.target.func.name, "ga_trace")) saw_ga_trace = true;
        if (r.target == .func and r.kind == .call26 and std.mem.eql(u8, r.target.func.name, "mp_trace")) saw_mp_trace = true;
        if (r.target == .import and std.mem.eql(u8, r.target.import.name, "malloc")) saw_malloc = true;
        if (r.target == .import and std.mem.eql(u8, r.target.import.name, "realloc")) saw_realloc = true;
        if (r.kind == .movw_g0) saw_movw_g0 = true;
        if (r.kind == .movw_g1) saw_movw_g1 = true;
        if (r.target == .cstr and r.target.cstr == Link.gc_stackmap_base_hash and r.kind == .adrp_page) saw_sm_adrp = true;
        if (r.target == .cstr and r.target.cstr == Link.gc_stackmap_base_hash and r.kind == .add_lo12) saw_sm_add = true;
    }
    try testing.expect(saw_malloc);
    try testing.expect(saw_realloc);
    try testing.expect(saw_ga_trace and saw_mp_trace);
    try testing.expect(saw_movw_g0 and saw_movw_g1);
    try testing.expect(saw_sm_adrp and saw_sm_add);

    // Pure fixed bytes ⇒ byte-identical across relowers (-jN determinism).
    var fc2 = try lowerGcCollect(gpa);
    defer fc2.deinit(gpa);
    try testing.expectEqualSlices(u8, fc.code, fc2.code);
}

test "gc_stats/gc_span_count: read the CB counters and are byte-identical across relowers" {
    const gpa = testing.allocator;
    inline for (.{ .{ "gc_stats", lowerGcStats }, .{ "gc_span_count", lowerGcSpanCount } }) |pair| {
        var a = try pair[1](gpa);
        defer a.deinit(gpa);
        var b = try pair[1](gpa);
        defer b.deinit(gpa);
        try testing.expectEqualStrings(pair[0], a.sym.name);
        try testing.expectEqual(Link.SymKind.builtin, a.sym.kind);
        try testing.expectEqual(Aarch64.ret, std.mem.readInt(u32, a.code[a.code.len - 4 ..][0..4], .little));
        try testing.expectEqualSlices(u8, a.code, b.code); // pure fixed bytes (-jN identity)
    }
}
