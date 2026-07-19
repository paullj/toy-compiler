//! A tiny shared harness for the hand-emitted `write`-calling builtins
//! (`lowerPanic` and friends). Each still writes its own body; these own only the
//! boilerplate every one repeats verbatim — opening a frame, the `write` GOT-import
//! preamble (byte-identical bar the destination register), the error-path reloc-name
//! cleanup, and packaging the finished FnCode. Plus the x29-chain walk shape and the
//! backpatch primitives the bodies share.

const std = @import("std");
const Aarch64 = @import("../Aarch64.zig");
const Link = @import("../../link/Link.zig");
const testing = std.testing;

/// Append one little-endian AArch64 instruction word to a builtin's code buffer.
pub fn emitWord(c: *std.ArrayList(u8), a: std.mem.Allocator, word: u32) error{OutOfMemory}!void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, word, .little);
    try c.appendSlice(a, &buf);
}

/// Open a standard frame: `stp x29,x30,[sp,#-16]! ; mov x29,sp`. A valid frame chain
/// is required both to call `write` and for `panic` to walk it.
pub fn emitFramePrologue(code: *std.ArrayList(u8), gpa: std.mem.Allocator) error{OutOfMemory}!void {
    try emitWord(code, gpa, Aarch64.stpFpLrPre);
    try emitWord(code, gpa, Aarch64.movFpSp);
}

/// Emit the `write` GOT-import preamble — `adrp x16, write@GOT ; ldr xRd,[x16]` — and
/// append its two `.import` relocs (patched to the `__got` slot after layout). The fn
/// pointer lands in xRd; the caller issues `blr xRd`. Each name is freed on this fn's
/// own error path; on success `relocs` owns them (freed via `deinitBuiltinRelocs` or,
/// once packaged, `FnCode.deinit`).
pub fn emitWriteImport(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, rd: u32) error{OutOfMemory}!void {
    try emitImportPreamble(code, relocs, gpa, "write", rd);
}

/// Emit the GOT-import preamble for dyld symbol `name` — `adrp x16, name@GOT ; ldr
/// xRd,[x16]` — and append its two `.import` relocs (patched to the `__got` slot
/// after layout). The fn pointer lands in xRd; the caller issues `blr xRd`. Each
/// name is freed on this fn's error path; on success `relocs` owns them (freed via
/// `deinitBuiltinRelocs` / `FnCode.deinit`). The generalization of `emitWriteImport`
/// over the symbol name, so an `extern` call reaches its own `__got` slot.
///
/// pub: also the driver's extern-call path (`genCall`) reaches its `__got` slot here.
pub fn emitImportPreamble(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, name: []const u8, rd: u32) error{OutOfMemory}!void {
    {
        const nm = try gpa.dupe(u8, name);
        errdefer gpa.free(nm);
        try relocs.append(gpa, .{ .site = @intCast(code.items.len), .target = .{ .import = .{ .kind = .import, .name = nm } }, .kind = .adrp_page });
    }
    try emitWord(code, gpa, Aarch64.adrp(16, 0));
    {
        const nm = try gpa.dupe(u8, name);
        errdefer gpa.free(nm);
        try relocs.append(gpa, .{ .site = @intCast(code.items.len), .target = .{ .import = .{ .kind = .import, .name = nm } }, .kind = .ldr_lo12 });
    }
    try emitWord(code, gpa, Aarch64.ldrRegUoff(rd, 16, 0));
}

/// Free a partially-built builtin's reloc target names + the reloc array (deinit of the
/// array alone does NOT free the interned `.func`/`.import` names). The builtins'
/// error-path errdefer; on success ownership passes to the FnCode.
pub fn deinitBuiltinRelocs(relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator) void {
    for (relocs.items) |r| if (r.target.name()) |nm| gpa.free(nm);
    relocs.deinit(gpa);
}

/// Package a hand-emitted builtin: name it `builtin` and take ownership of its code +
/// relocs. On failure the caller's `code`/`relocs` errdefers reclaim the buffers.
pub fn finishBuiltin(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, name_str: []const u8) error{OutOfMemory}!Link.FnCode {
    const name = try gpa.dupe(u8, name_str);
    errdefer gpa.free(name);
    return .{
        .sym = .{ .kind = .builtin, .name = name },
        .code = try code.toOwnedSlice(gpa),
        .relocs = try relocs.toOwnedSlice(gpa),
        .literals = &.{},
    };
}

/// Self-locate the runtime base of `__text` into `dst_reg` (the PIE code address of
/// the first byte of the emitted text section). The two movw relocs bake the `adr`'s
/// own absolute __text offset (addend = adr_site - mov_site); they carry `.none` (the
/// baked value is site-relative). Intermediate scratch x9/x10 are consumed here and
/// left dead; `dst_reg` receives the result last. The one PIE way to a code address —
/// shared by `lowerPanic` (dst x19) and the `text_base()` builtin (dst x0).
pub fn emitTextBase(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, dst_reg: u32) error{OutOfMemory}!void {
    const A = Aarch64;
    const emit = emitWord;
    const adr_pos: u32 = @intCast(code.items.len);
    try emit(code, gpa, A.adr(9, 0)); // x9 = text_base + adr_off
    var site: u32 = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .none, .kind = .movw_g0, .addend = @as(i64, adr_pos) - @as(i64, site) });
    try emit(code, gpa, A.movz(10, 0, 0)); // x10 = adr_off (lo)
    site = @intCast(code.items.len);
    try relocs.append(gpa, .{ .site = site, .target = .none, .kind = .movw_g1, .addend = @as(i64, adr_pos) - @as(i64, site) });
    try emit(code, gpa, A.movk(10, 0, 1)); // x10 |= adr_off (hi)
    try emit(code, gpa, A.subReg(dst_reg, 9, 10)); // dst = text_base
}

/// Backpatch a `cbz`/`cbnz` placeholder at byte `site` to branch to byte `target`
/// (preserving opcode + rt), given the resolved word delta.
pub fn patchCbzTo(buf: []u8, site: u32, target: u32) void {
    const delta: i19 = @intCast(@divExact(@as(i64, target) - @as(i64, site), 4));
    const word = std.mem.readInt(u32, buf[site..][0..4], .little);
    std.mem.writeInt(u32, buf[site..][0..4], Aarch64.patchCbz(word, delta), .little);
}

/// Backpatch a `b.cond` placeholder at byte `site` to branch to byte `target`
/// (preserving opcode + cond).
pub fn patchBCondTo(buf: []u8, site: u32, target: u32) void {
    const delta: i19 = @intCast(@divExact(@as(i64, target) - @as(i64, site), 4));
    const word = std.mem.readInt(u32, buf[site..][0..4], .little);
    std.mem.writeInt(u32, buf[site..][0..4], Aarch64.patchBCond(word, delta), .little);
}

/// Backpatch an unconditional `b` placeholder at byte `site` to branch to `target`.
pub fn patchBTo(buf: []u8, site: u32, target: u32) void {
    const delta: i26 = @intCast(@divExact(@as(i64, target) - @as(i64, site), 4));
    const word = std.mem.readInt(u32, buf[site..][0..4], .little);
    std.mem.writeInt(u32, buf[site..][0..4], Aarch64.patchB(word, delta), .little);
}

/// The two x29-chain walk shapes. They differ behaviorally, not cosmetically:
/// `each_frame` guards the current fp, runs per-frame work on it, then advances —
/// falling through with the cursor at 0; `to_outermost` advances to the last
/// non-null fp and keeps it in the cursor, doing no per-frame work.
const FpWalkMode = enum { each_frame, to_outermost };

/// Emit an x29 frame-chain walk shared by `panic` (symbolize every live frame)
/// and the collector root scan (locate the outermost frame). Emits a null-fp +
/// cap-backstop loop that advances `cursor = *cursor` and backpatches its two
/// `cbz`s to the fall-through exit and its `b` to the loop top. The caller seeds
/// `cursor` (from x29) and `cap_reg` before the call and consumes the terminal
/// state after it. `.each_frame` emits `per_frame` (if non-null) once per live
/// frame with the frame's fp in `cursor` and ignores `scratch`; `.to_outermost`
/// uses `scratch` as the parent register and ignores `per_frame`.
pub fn emitFpWalk(
    code: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
    comptime mode: FpWalkMode,
    cursor: u32,
    cap_reg: u32,
    scratch: u32,
    per_frame: ?*const fn (*std.ArrayList(u8), std.mem.Allocator) error{OutOfMemory}!void,
) error{OutOfMemory}!void {
    const A = Aarch64;
    const loop_top: u32 = @intCast(code.items.len);
    var cbz_null: u32 = undefined;
    var cbz_cap: u32 = undefined;
    switch (mode) {
        .each_frame => {
            cbz_null = @intCast(code.items.len);
            try emitWord(code, gpa, A.cbz(cursor, 0)); // fp == 0 -> done
            cbz_cap = @intCast(code.items.len);
            try emitWord(code, gpa, A.cbz(cap_reg, 0)); // cap exhausted -> done
            if (per_frame) |cb| try cb(code, gpa);
            try emitWord(code, gpa, A.ldrRegUoff(cursor, cursor, 0)); // fp = *fp
            try emitWord(code, gpa, A.subImm(cap_reg, cap_reg, 1));
        },
        .to_outermost => {
            try emitWord(code, gpa, A.ldrRegUoff(scratch, cursor, 0)); // parent = *fp
            cbz_null = @intCast(code.items.len);
            try emitWord(code, gpa, A.cbz(scratch, 0)); // parent == 0 -> keep cursor
            try emitWord(code, gpa, A.subImm(cap_reg, cap_reg, 1));
            cbz_cap = @intCast(code.items.len);
            try emitWord(code, gpa, A.cbz(cap_reg, 0)); // cap exhausted -> done
            try emitWord(code, gpa, A.movReg(cursor, scratch));
        },
    }
    const b_site: u32 = @intCast(code.items.len);
    try emitWord(code, gpa, A.b(0));
    const done: u32 = @intCast(code.items.len);
    patchCbzTo(code.items, cbz_null, done);
    patchCbzTo(code.items, cbz_cap, done);
    patchBTo(code.items, b_site, loop_top);
}

/// Emit `bl <builtin>` with a `.call26` reloc, so one hand-emitted builtin can call
/// another (here: `gc_alloc`'s alloc-triggered `bl gc_collect`). The link tail resolves
/// the target like any intra-image `.func` call once both bodies are appended.
pub fn emitBuiltinCall(code: *std.ArrayList(u8), relocs: *std.ArrayList(Link.Reloc), gpa: std.mem.Allocator, name_str: []const u8) error{OutOfMemory}!void {
    const nm = try gpa.dupe(u8, name_str);
    errdefer gpa.free(nm);
    try relocs.append(gpa, .{ .site = @intCast(code.items.len), .target = .{ .func = .{ .kind = .builtin, .name = nm } }, .kind = .call26 });
    try emitWord(code, gpa, Aarch64.bl(0));
}

test "emitFpWalk each_frame: top-guarded loop advancing in place, cbz/b patched" {
    const gpa = std.testing.allocator;
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(gpa);
    try emitFpWalk(&code, gpa, .each_frame, 20, 21, undefined, null);

    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(gpa);
    try emitWord(&want, gpa, Aarch64.cbz(20, 5)); // @0  -> done @20
    try emitWord(&want, gpa, Aarch64.cbz(21, 4)); // @4  -> done @20
    try emitWord(&want, gpa, Aarch64.ldrRegUoff(20, 20, 0)); // fp = *fp
    try emitWord(&want, gpa, Aarch64.subImm(21, 21, 1));
    try emitWord(&want, gpa, Aarch64.b(-4)); // @16 -> loop top @0
    try std.testing.expectEqualSlices(u8, want.items, code.items);
}

test "emitFpWalk to_outermost: parent-load loop keeping the last non-null fp" {
    const gpa = std.testing.allocator;
    var code: std.ArrayList(u8) = .empty;
    defer code.deinit(gpa);
    try emitFpWalk(&code, gpa, .to_outermost, 0, 1, 2, null);

    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(gpa);
    try emitWord(&want, gpa, Aarch64.ldrRegUoff(2, 0, 0)); // parent = *fp
    try emitWord(&want, gpa, Aarch64.cbz(2, 5)); // @4  -> done @24
    try emitWord(&want, gpa, Aarch64.subImm(1, 1, 1));
    try emitWord(&want, gpa, Aarch64.cbz(1, 3)); // @12 -> done @24
    try emitWord(&want, gpa, Aarch64.movReg(0, 2));
    try emitWord(&want, gpa, Aarch64.b(-5)); // @20 -> loop top @0
    try std.testing.expectEqualSlices(u8, want.items, code.items);
}

test "panic backpatch wrappers resolve signed word deltas, preserving opcode/rt/cond" {
    var buf: [16]u8 = undefined;
    // cbz x20 @ site 0 → target 8: +2 words, opcode + rt preserved.
    std.mem.writeInt(u32, buf[0..4], Aarch64.cbz(20, 0), .little);
    patchCbzTo(&buf, 0, 8);
    try testing.expectEqual(Aarch64.cbz(20, 2), std.mem.readInt(u32, buf[0..4], .little));
    // b.hi @ site 4 → target 12: +2 words, condition preserved.
    std.mem.writeInt(u32, buf[4..8], Aarch64.bCond(.hi, 0), .little);
    patchBCondTo(&buf, 4, 12);
    try testing.expectEqual(Aarch64.bCond(.hi, 2), std.mem.readInt(u32, buf[4..8], .little));
    // b @ site 12 → target 4: −2 words (backward).
    std.mem.writeInt(u32, buf[12..16], Aarch64.b(0), .little);
    patchBTo(&buf, 12, 4);
    try testing.expectEqual(Aarch64.b(-2), std.mem.readInt(u32, buf[12..16], .little));
}
