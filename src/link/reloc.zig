//! Relocations: the vocabulary + the kind→bytes machinery, in one place.
//!
//! WHY this module exists: a relocation's life used to be smeared across the whole
//! backend — the types here, the intra-module patch in `Link.fnLinkJob`, the
//! cross-segment patch in `Link.dataRelocJob`, and the raw encoders in `Aarch64`.
//! Understanding "how does one reloc kind become a patched byte" meant reading all of
//! them. This module concentrates that: the `Reloc`/`SymbolId`/`RelocKind` types, the
//! single source of truth for which targets own a heap name (`SymbolId.name`), which
//! phase a kind resolves in (`RelocKind.phase`), and the two patch primitives
//! (`patchIntra`/`patchCross`). `Link` keeps only the ORCHESTRATION — laying out code,
//! resolving handles/vmaddrs, and calling these — and re-exports the types so every
//! `Link.Reloc`/`Link.SymbolId` callsite is unchanged.

const std = @import("std");
const Aarch64 = @import("../codegen/Aarch64.zig");
const symbols = @import("../symbols/Sym.zig");

pub const SymKind = symbols.SymKind;
pub const SymName = symbols.SymName;

/// A symbolic reference to a definition. NEVER an address — the linker (for
/// `.func`/intra-module) or the post-vmaddr pass (for the cross-segment cases) maps it
/// to a final address.
///
///   * `.func`   — a function, named by its stable `SymName`. Patched by `link`
///                 (PC-relative `bl`).
///   * `.cstr`   — a string literal. While lowered (and on disk) this holds the
///                 literal's CONTENT HASH; the relink tail rewrites it to a byte offset
///                 into the program-wide `__cstring` blob, after which the cross-segment
///                 pass reads it as that offset.
///   * `.import` — an external symbol, named by its stable `SymName` (always
///                 `{.import,"write"}`). Reached through its `__got` slot; patched by
///                 the cross-segment pass.
///   * `.none`   — no target. For a reloc whose patched value is derived purely from its
///                 own site + addend (the `.movw_g0`/`.movw_g1` self-locate), so there is
///                 nothing to name or free.
pub const SymbolId = union(enum) {
    func: SymName,
    cstr: u64,
    import: SymName,
    none,

    /// The heap-owned name this target carries, if any. THE single source of truth for
    /// the "which targets own a name" rule — every free/dupe boundary asks here instead
    /// of re-deciding, so a new variant is a one-line change, not a six-site edit.
    pub fn name(self: SymbolId) ?[]const u8 {
        return switch (self) {
            .func, .import => |s| s.name,
            .cstr, .none => null,
        };
    }
};

/// The kind of patch a relocation requests.
///
///   * `.call26`    — an AArch64 `bl`: imm26 = (target − site)/4. Intra-module.
///   * `.adrp_page` — an `adrp`: page delta between the site and a cstring/GOT vmaddr.
///   * `.add_lo12`  — an `add (imm12)`: the low-12 bits of the cstring vmaddr.
///   * `.ldr_lo12`  — an `ldr (unsigned offset)`: the low-12 bits of the GOT-slot vmaddr.
///   * `.movw_g0` / `.movw_g1` — bake a SELF-RELATIVE __text offset (`site_abs + addend`)
///                    into a `movz`/`movk` imm16 (low / high halfword). Intra-module.
pub const RelocKind = enum {
    call26,
    adrp_page,
    add_lo12,
    ldr_lo12,
    movw_g0,
    movw_g1,

    /// Where a kind resolves. `.intra` — link time, from `__text` offsets alone
    /// (offset, not address → PIE-safe, no runtime reloc; patched in place by `link`).
    /// `.cross` — after MachO assigns segment vmaddrs (deferred to `applyDataRelocs`).
    /// This classification is the ONE authority for the intra/cross partition the two
    /// patch executors used to enforce by scattered `unreachable` arms.
    pub const Phase = enum { intra, cross };
    pub fn phase(self: RelocKind) Phase {
        return switch (self) {
            .call26, .movw_g0, .movw_g1 => .intra,
            .adrp_page, .add_lo12, .ldr_lo12 => .cross,
        };
    }
};

/// One patch to apply during linking. `site` is the byte offset of the word to patch
/// within its owning function's `code`. `addend` shifts the resolved value: 0 for
/// `.call26`/`.adrp_page`/`.add_lo12`/`.ldr_lo12`; for `.movw_g0`/`.movw_g1` it names an
/// instruction relative to the reloc site (the `__panic` self-locate `adr`), so the
/// baked value is `site_abs + addend`.
pub const Reloc = struct {
    site: u32,
    target: SymbolId,
    kind: RelocKind,
    addend: i64 = 0,
};

pub const IntraError = error{CallTargetTooFar};

/// Patch one INTRA-MODULE reloc in place at `text[site_abs..][0..4]`. `target_off` is the
/// resolved `__text` offset of a `.call26` callee (ignored by the movw kinds, which are
/// self-relative). Raised `CallTargetTooFar` if a `bl` displacement overflows imm26 — it
/// cannot occur within one 0x4000 page, but surface it rather than truncate-and-miscompile.
pub fn patchIntra(text: []u8, kind: RelocKind, site_abs: u32, target_off: u32, addend: i64) IntraError!void {
    switch (kind) {
        .call26 => {
            const target_abs: i64 = @as(i64, target_off) + addend;
            const delta: i64 = target_abs - @as(i64, site_abs);
            std.debug.assert(@mod(delta, 4) == 0); // BL targets are word-aligned
            const imm: i64 = @divExact(delta, 4);
            if (imm < -(@as(i64, 1) << 25) or imm > (@as(i64, 1) << 25) - 1) return error.CallTargetTooFar;
            std.mem.writeInt(u32, text[site_abs..][0..4], Aarch64.bl(@intCast(imm)), .little);
        },
        .movw_g0, .movw_g1 => {
            // Bake a self-relative __text OFFSET (site_abs + addend) into the movz/movk
            // imm16 — an offset, not an address, so no runtime reloc (PIE-safe).
            const val: u32 = @intCast(@as(i64, site_abs) + addend);
            const imm: u16 = if (kind == .movw_g0) @truncate(val) else @truncate(val >> 16);
            const word = std.mem.readInt(u32, text[site_abs..][0..4], .little);
            std.mem.writeInt(u32, text[site_abs..][0..4], Aarch64.patchMovImm16(word, imm), .little);
        },
        .adrp_page, .add_lo12, .ldr_lo12 => unreachable, // cross-segment; use patchCross
    }
}

/// Re-encode one CROSS-SEGMENT reloc's `word` from the resolved `target_vmaddr` and the
/// site's runtime `site_vmaddr`. Pure (the caller reads/writes the site). The page delta
/// uses the 4096-byte ARM page (`>>12`); under PIE/ASLR dyld slides all segments together
/// so the delta is exact at runtime.
pub fn patchCross(word: u32, kind: RelocKind, site_vmaddr: u64, target_vmaddr: u64) u32 {
    return switch (kind) {
        .adrp_page => blk: {
            const pages: i64 = @as(i64, @intCast(target_vmaddr >> 12)) - @as(i64, @intCast(site_vmaddr >> 12));
            break :blk Aarch64.patchAdrp(word, @intCast(pages));
        },
        .add_lo12 => Aarch64.patchAddImm12(word, @intCast(target_vmaddr & 0xFFF)),
        .ldr_lo12 => blk: {
            std.debug.assert(target_vmaddr & 7 == 0);
            break :blk Aarch64.patchLdrUoff(word, @intCast(target_vmaddr & 0xFFF));
        },
        .call26, .movw_g0, .movw_g1 => unreachable, // intra-module; use patchIntra
    };
}

/// The content-hash sentinel a `.cstr` reloc carries to name the backtrace symbol
/// table's base. The relink tail reserves it in the cstring offset map (→ the table's
/// byte offset in the appended `__cstring` blob), so `__panic` reaches the table via the
/// same adrp+add path a string literal uses. Not a real hash (astronomically unlikely to
/// collide; the relink tail rejects an actual collision).
pub const symtab_base_hash: u64 = 0x5717_ab1e_ba5e_0000;

/// The content-hash sentinel naming the descriptor table's base. A distinct high prefix
/// from `symtab_base_hash` so the two reserved sentinels never alias; the per-type
/// `descHash(T)` sentinels share this prefix (see `Link.descHash`). Reserved in the
/// cstring offset map by the relink tail exactly like `symtab_base_hash`.
pub const desc_table_base_hash: u64 = 0xDE5C_0000_0000_0000;

const testing = std.testing;

test "SymbolId.name is the single source of truth for name ownership" {
    try testing.expectEqualStrings("f", (SymbolId{ .func = .{ .kind = .user_fn, .name = "f" } }).name().?);
    try testing.expectEqualStrings("write", (SymbolId{ .import = .{ .kind = .import, .name = "write" } }).name().?);
    try testing.expect((SymbolId{ .cstr = 7 }).name() == null);
    try testing.expect(@as(SymbolId, .none).name() == null);
}

test "RelocKind.phase partitions intra-module from cross-segment kinds" {
    try testing.expectEqual(RelocKind.Phase.intra, RelocKind.call26.phase());
    try testing.expectEqual(RelocKind.Phase.intra, RelocKind.movw_g0.phase());
    try testing.expectEqual(RelocKind.Phase.intra, RelocKind.movw_g1.phase());
    try testing.expectEqual(RelocKind.Phase.cross, RelocKind.adrp_page.phase());
    try testing.expectEqual(RelocKind.Phase.cross, RelocKind.add_lo12.phase());
    try testing.expectEqual(RelocKind.Phase.cross, RelocKind.ldr_lo12.phase());
}

test "patchIntra: call26 encodes the signed word delta; overflow surfaces" {
    var text = [_]u8{0} ** 12;
    // bl at site 8 targeting offset 0 → delta −8 → 0x97FFFFFE (ground truth).
    try patchIntra(&text, .call26, 8, 0, 0);
    try testing.expectEqual(@as(u32, 0x97FFFFFE), std.mem.readInt(u32, text[8..12], .little));
    // A displacement past ±128 MiB (via addend) overflows imm26.
    try testing.expectError(error.CallTargetTooFar, patchIntra(&text, .call26, 0, 0, (@as(i64, 1) << 27) + 4));
}

test "patchIntra: movw bakes the site-relative offset halfwords" {
    var text: [8]u8 = undefined;
    std.mem.writeInt(u32, text[0..4], Aarch64.movz(10, 0, 0), .little);
    std.mem.writeInt(u32, text[4..8], Aarch64.movk(10, 0, 1), .little);
    // site 0, addend +0x12340 → value 0x12340; g0 = low16 (0x2340), g1 = high16 (0x0001).
    try patchIntra(&text, .movw_g0, 0, 0, 0x12340);
    try patchIntra(&text, .movw_g1, 4, 0, 0x12340 - 4);
    try testing.expectEqual(Aarch64.movz(10, 0x2340, 0), std.mem.readInt(u32, text[0..4], .little));
    try testing.expectEqual(Aarch64.movk(10, 0x0001, 1), std.mem.readInt(u32, text[4..8], .little));
}

test "patchCross: adrp/add/ldr reproduce the clang reference words" {
    // Ground truth from the clang `write` reference (mirrors the Link applyDataRelocs test):
    //   adrp x8,<cstr> @ site 0x478, target 0x4b0 → 0x90000008
    //   add  x8,x8,#0x4b0                          → 0x9112C108
    //   ldr  x16,[x16]  @ target GOT 0x4000        → 0xF9400210
    const base: u64 = 0x100000000;
    try testing.expectEqual(@as(u32, 0x90000008), patchCross(Aarch64.adrp(8, 0), .adrp_page, base + 0x478, base + 0x4b0));
    try testing.expectEqual(@as(u32, 0x9112C108), patchCross(Aarch64.addImm(8, 8, 0), .add_lo12, base + 0x47c, base + 0x4b0));
    try testing.expectEqual(@as(u32, 0xF9400210), patchCross(Aarch64.ldrRegUoff(16, 16, 0), .ldr_lo12, base + 0x4a8, base + 0x4000));
}
