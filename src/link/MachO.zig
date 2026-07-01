//! Assemble a runnable arm64 macOS Mach-O executable from a `__text` code blob.
//!
//! WHY: M1 emits and "links" the executable itself — no system linker. This file
//! is the linker: it lays out the single page of code we generated into a PIE
//! `MH_EXECUTE` image with the minimal set of load commands the macOS kernel and
//! dyld actually require to run a process.
//!
//! ENTRY MODEL (empirically forced): the originally-planned dyld-free
//! LC_UNIXTHREAD layout is rejected by this kernel (`EBADEXEC`/`SIGKILL`) even
//! when correctly ad-hoc signed. The layout that RUNS is a PIE `LC_MAIN`
//! executable that loads dyld and libSystem: `main` is the LC_MAIN entry function
//! and dyld's start glue calls `exit(w0 & 0xFF)` on its return. So we ship:
//!   __PAGEZERO, __TEXT(+__text), __LINKEDIT,
//!   LC_DYLD_CHAINED_FIXUPS (empty 56-byte blob), LC_LOAD_DYLINKER /usr/lib/dyld,
//!   LC_MAIN, LC_LOAD_DYLIB libSystem, LC_CODE_SIGNATURE (always LAST).
//!
//! M2 adds a SECOND layout for programs that emit output. When the lowered
//! program carries interned `__cstring` bytes and/or imports (it called `print`),
//! `assemble` takes the MULTI-SEGMENT path:
//!   __PAGEZERO, __TEXT(+__text +__cstring), __DATA_CONST(+__got), __LINKEDIT,
//!   LC_DYLD_CHAINED_FIXUPS (a real one-import blob), LC_LOAD_DYLINKER,
//!   LC_MAIN, LC_LOAD_DYLIB libSystem, LC_CODE_SIGNATURE (LAST).
//! The `__got` slot is seeded with the chained-fixups bind sentinel
//! (0x8000000000000000), and `LC_DYLD_CHAINED_FIXUPS` names `_write` against
//! libSystem so dyld binds the slot at load. The cross-segment adrp/add/ldr
//! relocations are patched by `Link.applyDataRelocs` (in `link/emit.zig`'s
//! `assembleAndSign`) once `assemble` has assigned every segment's vmaddr —
//! which it reports in the extended `Layout`. `assemble` NO LONGER signs;
//! `assembleAndSign` orchestrates assemble → applyDataRelocs → CodeSign.sign so
//! the signature covers the final, patched bytes.
//!
//! Everything is fixed-size, so layout is a single pass: place the header + load
//! commands + code in __TEXT (one 0x4000 page), then __LINKEDIT carries the
//! chained-fixups blob and, at the very end, the ad-hoc code signature. The sig
//! region is reserved/zeroed here with a deterministic length; `CodeSign.sign`
//! (Stage D) hashes `image[0..sig_file_off)` and fills it in place.
//!
//! All header / load-command scalars are little-endian (native arm64). The code
//! signature blob uses big-endian scalars — that lives in `CodeSign.zig`.

const std = @import("std");
const builtin = @import("builtin");
const CodeSign = @import("CodeSign.zig");

const Self = @This();

// Mach-O constants (verified via `otool -hlv` on a cc-built reference).

const MH_MAGIC_64: u32 = 0xFEEDFACF;
const CPU_TYPE_ARM64: u32 = 0x0100000C;
const CPU_SUBTYPE_ARM64_ALL: u32 = 0x00000000;
const MH_EXECUTE: u32 = 0x2;
// NOUNDEFS | DYLDLINK | TWOLEVEL | PIE
const MH_FLAGS: u32 = 0x00200085;

const LC_REQ_DYLD: u32 = 0x80000000;
const LC_SEGMENT_64: u32 = 0x19;
const LC_LOAD_DYLINKER: u32 = 0xe;
const LC_LOAD_DYLIB: u32 = 0xc;
const LC_MAIN: u32 = 0x28 | LC_REQ_DYLD;
const LC_DYLD_CHAINED_FIXUPS: u32 = 0x34 | LC_REQ_DYLD;
const LC_CODE_SIGNATURE: u32 = 0x1d;

const VM_PROT_READ: u32 = 0x1;
const VM_PROT_WRITE: u32 = 0x2;
const VM_PROT_EXECUTE: u32 = 0x4;

const S_REGULAR: u32 = 0x0;
// S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS
const TEXT_SECT_FLAGS: u32 = 0x80000400;
// __cstring section type: NUL-terminated literal strings.
const S_CSTRING_LITERALS: u32 = 0x2;
// __got section type: non-lazy symbol pointers (one slot per import).
const S_NON_LAZY_SYMBOL_POINTERS: u32 = 0x6;
// __DATA_CONST is read-only after dyld binds it.
const SG_READ_ONLY: u32 = 0x10;

/// One dyld-imported external symbol. M2 always uses exactly one: `_write` from
/// libSystem (the only LC_LOAD_DYLIB, so its dylib ordinal is 1).
pub const Import = struct {
    name: []const u8,
};

/// The chained-fixups bind sentinel written into each `__got` slot on disk: a
/// `dyld_chained_ptr_64_bind` with bind=1 (bit 63) and ordinal 0 → imports[0].
/// dyld overwrites it in memory with the bound address at load; the on-disk byte
/// is what we hash, so it stays stable.
const GOT_BIND_SENTINEL: u64 = 0x8000000000000000;

/// Segment file alignment / vm page size for this target (16 KB).
pub const PAGE: u64 = 0x4000;
/// PIE image base for arm64-macos.
pub const BASE: u64 = 0x100000000;

const DYLD_PATH = "/usr/lib/dyld";
const LIBSYSTEM_PATH = "/usr/lib/libSystem.B.dylib";

/// The empty-imports `LC_DYLD_CHAINED_FIXUPS` payload, copied verbatim from a
/// cc-built reference (`xxd -s 0x4000 -l 56 /tmp/ref`). 56 bytes: no imports, a
/// single segment with no fixup starts.
const CHAINED_FIXUPS_BLOB = [56]u8{
    0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x30, 0x00, 0x00, 0x00, 0x30, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
};

// Each is built from real bytes below; these consts let layout be a single pass.

const SEG_CMD_SIZE: u32 = 72; // sizeof(segment_command_64)
const SECT_SIZE: u32 = 80; // sizeof(section_64)
const TEXT_SEG_CMD_SIZE: u32 = SEG_CMD_SIZE + SECT_SIZE; // one section
const FIXUPS_CMD_SIZE: u32 = 16; // linkedit_data_command
const MAIN_CMD_SIZE: u32 = 24; // entry_point_command
const SIG_CMD_SIZE: u32 = 16; // linkedit_data_command

/// `cmdsize` for a dylinker/dylib command: a fixed head plus the NUL-terminated
/// path, rounded up to 8.
fn pathCmdSize(head: u32, path: []const u8) u32 {
    const raw = head + @as(u32, @intCast(path.len)) + 1; // +1 for NUL
    return std.mem.alignForward(u32, raw, 8);
}

const DYLINKER_CMD_SIZE: u32 = blk: {
    @setEvalBranchQuota(2000);
    break :blk pathCmdSize(12, DYLD_PATH);
};
const DYLIB_CMD_SIZE: u32 = blk: {
    @setEvalBranchQuota(2000);
    break :blk pathCmdSize(24, LIBSYSTEM_PATH);
};

const NCMDS: u32 = 8;
const SIZEOFCMDS: u32 = SEG_CMD_SIZE + // __PAGEZERO
    TEXT_SEG_CMD_SIZE + // __TEXT + __text
    SEG_CMD_SIZE + // __LINKEDIT
    FIXUPS_CMD_SIZE +
    DYLINKER_CMD_SIZE +
    MAIN_CMD_SIZE +
    DYLIB_CMD_SIZE +
    SIG_CMD_SIZE;

const HEADER_SIZE: u32 = 32; // mach_header_64

/// Structural facts about an assembled (but possibly unsigned) image; returned
/// by `assemble` so tests, `CodeSign`, and `Link.applyDataRelocs` can inspect /
/// fill / patch the right regions.
pub const Layout = struct {
    image: []u8,
    /// File offset where `__text` (the code blob) starts.
    code_file_off: u32,
    /// Size of the code blob (== `__text` section size).
    text_size: u32,
    /// File offset of the reserved code-signature region (== codeLimit).
    sig_file_off: u32,
    /// Reserved code-signature length in bytes.
    sig_len: u32,
    /// Runtime address `__text` loads at (= BASE + code_file_off).
    text_vmaddr: u64 = BASE,
    /// Runtime address of the `__cstring` section base (multi-seg path only; 0
    /// when there are no interned strings).
    cstring_vmaddr: u64 = 0,
    /// Runtime address of the `__got` section base (multi-seg path only; 0 when
    /// there are no imports).
    got_vmaddr: u64 = 0,
};

/// Assemble the image with the code-signature region reserved (and currently
/// zeroed). The header / load commands / code are final; only the sig region is
/// left for `CodeSign.sign` to fill. Caller owns `image`.
///
/// `cstrings` are the interned read-only string bytes (Codegen's blob), and
/// `imports` the dyld symbols to bind. When BOTH are empty this is a pure M1/M3
/// program and we take the unchanged single-__TEXT layout; otherwise the
/// multi-segment path (__cstring section + __DATA_CONST/__got + a real
/// chained-fixups blob). `assemble` does NOT sign — the caller patches
/// cross-segment relocs first, then signs.
pub fn assemble(
    gpa: std.mem.Allocator,
    identifier: []const u8,
    code: []const u8,
    entry_text_off: u32,
    cstrings: []const u8,
    imports: []const Import,
) !Layout {
    if (cstrings.len == 0 and imports.len == 0) {
        return assembleEmpty(gpa, identifier, code, entry_text_off);
    }
    return assembleMulti(gpa, identifier, code, entry_text_off, cstrings, imports);
}

/// The unchanged M1/M3 single-segment layout (no __cstring, no imports). Kept
/// byte-identical so existing binaries/tests stay green.
fn assembleEmpty(gpa: std.mem.Allocator, identifier: []const u8, code: []const u8, entry_text_off: u32) !Layout {
    // Code immediately follows the header + load commands, 4-byte aligned.
    const code_file_off: u32 = std.mem.alignForward(u32, HEADER_SIZE + SIZEOFCMDS, 4);
    const text_size: u32 = @intCast(code.len);

    // The entry must land inside the code blob (it is `main`'s text offset).
    std.debug.assert(entry_text_off < text_size);

    // __TEXT spans as many 0x4000 pages as the header+cmds+code need. A program
    // that fits one page yields text_seg_size == PAGE — byte-identical to the
    // original single-page layout — so existing binaries/tests stay green; larger
    // programs simply grow __TEXT (and slide __LINKEDIT) by whole pages.
    const text_end: u64 = @as(u64, code_file_off) + code.len;
    const text_seg_size: u32 = @intCast(std.mem.alignForward(u64, text_end, PAGE));

    // __LINKEDIT begins right after __TEXT (page-aligned): the chained-fixups blob
    // first, then (16-aligned) the code signature.
    const linkedit_off: u32 = text_seg_size;
    const fixups_off: u32 = linkedit_off;
    const after_fixups: u32 = fixups_off + @as(u32, CHAINED_FIXUPS_BLOB.len);
    const sig_file_off: u32 = std.mem.alignForward(u32, after_fixups, 16);

    // The signature length is deterministic from the identifier and the number
    // of code pages, so we can reserve it before hashing.
    const sig_len: u32 = CodeSign.signatureLen(identifier, sig_file_off);

    const total: u32 = sig_file_off + sig_len;
    const le_filesize: u32 = total - linkedit_off;

    const image = try gpa.alloc(u8, total);
    errdefer gpa.free(image);
    @memset(image, 0);

    // Mach-O header (little-endian).
    var w = Writer{ .buf = image, .pos = 0 };
    w.put32(MH_MAGIC_64);
    w.put32(CPU_TYPE_ARM64);
    w.put32(CPU_SUBTYPE_ARM64_ALL);
    w.put32(MH_EXECUTE);
    w.put32(NCMDS);
    w.put32(SIZEOFCMDS);
    w.put32(MH_FLAGS);
    w.put32(0);

    // LC_SEGMENT_64 __PAGEZERO.
    w.segment("__PAGEZERO", .{
        .vmaddr = 0,
        .vmsize = BASE,
        .fileoff = 0,
        .filesize = 0,
        .maxprot = 0,
        .initprot = 0,
        .nsects = 0,
    });

    // LC_SEGMENT_64 __TEXT (+ __text section).
    w.put32(LC_SEGMENT_64);
    w.put32(TEXT_SEG_CMD_SIZE);
    w.name16("__TEXT");
    w.put64(BASE);
    w.put64(text_seg_size); // vmsize: whole-page span
    w.put64(0);
    w.put64(text_seg_size); // filesize: whole-page span
    w.put32(VM_PROT_READ | VM_PROT_EXECUTE); // r-x
    w.put32(VM_PROT_READ | VM_PROT_EXECUTE); // r-x
    w.put32(1);
    w.put32(0);
    // section_64 __text
    w.name16("__text");
    w.name16("__TEXT");
    w.put64(BASE + code_file_off);
    w.put64(text_size);
    w.put32(code_file_off);
    w.put32(2); // align 2^2 = 4
    w.put32(0);
    w.put32(0);
    w.put32(TEXT_SECT_FLAGS);
    w.put32(0);
    w.put32(0);
    w.put32(0);

    // LC_SEGMENT_64 __LINKEDIT.
    w.segment("__LINKEDIT", .{
        .vmaddr = BASE + linkedit_off,
        // vmsize must cover filesize; round up so it can never be smaller.
        .vmsize = std.mem.alignForward(u32, le_filesize, PAGE),
        .fileoff = linkedit_off,
        .filesize = le_filesize,
        .maxprot = VM_PROT_READ,
        .initprot = VM_PROT_READ,
        .nsects = 0,
    });

    // LC_DYLD_CHAINED_FIXUPS.
    w.put32(LC_DYLD_CHAINED_FIXUPS);
    w.put32(FIXUPS_CMD_SIZE);
    w.put32(fixups_off);
    w.put32(@intCast(CHAINED_FIXUPS_BLOB.len));

    // LC_LOAD_DYLINKER /usr/lib/dyld.
    w.put32(LC_LOAD_DYLINKER);
    w.put32(DYLINKER_CMD_SIZE);
    w.put32(12);
    w.path(DYLD_PATH, DYLINKER_CMD_SIZE, 12);

    // LC_MAIN.
    w.put32(LC_MAIN);
    w.put32(MAIN_CMD_SIZE);
    w.put64(code_file_off + entry_text_off); // entryoff = file offset of `main`
    w.put64(0);

    // LC_LOAD_DYLIB /usr/lib/libSystem.B.dylib.
    w.put32(LC_LOAD_DYLIB);
    w.put32(DYLIB_CMD_SIZE);
    w.put32(24);
    w.put32(2);
    w.put32(0x051F1304); // current_version
    w.put32(0x00010000); // compatibility_version
    w.path(LIBSYSTEM_PATH, DYLIB_CMD_SIZE, 24);

    // LC_CODE_SIGNATURE (LAST).
    w.put32(LC_CODE_SIGNATURE);
    w.put32(SIG_CMD_SIZE);
    w.put32(sig_file_off);
    w.put32(sig_len);

    // Sanity: we wrote exactly the header + all load commands.
    std.debug.assert(w.pos == HEADER_SIZE + SIZEOFCMDS);

    @memcpy(image[code_file_off..][0..code.len], code);

    // __LINKEDIT: chained-fixups blob (sig region stays zeroed).
    @memcpy(image[fixups_off..][0..CHAINED_FIXUPS_BLOB.len], &CHAINED_FIXUPS_BLOB);

    return .{
        .image = image,
        .code_file_off = code_file_off,
        .text_size = text_size,
        .sig_file_off = sig_file_off,
        .sig_len = sig_len,
    };
}

// Multi-segment (output) layout.
//
// Segment order is locked to match the chained-fixups `seg_info_offset` indices
// (verified against the clang `write` reference):
//   [0] __PAGEZERO  [1] __TEXT(__text+__cstring)  [2] __DATA_CONST(__got)
//   [3] __LINKEDIT
// Each segment occupies its own 0x4000 page. __cstring rides in __TEXT's page so
// it shares the same vmaddr page as __text; the __got lives in __DATA_CONST one
// page later; __LINKEDIT (fixups blob + signature) two pages later.

const NCMDS_MULTI: u32 = 9; // +1 over the empty path (the __DATA_CONST segment)
const DATAC_SEG_CMD_SIZE: u32 = SEG_CMD_SIZE + SECT_SIZE; // one section (__got)

const SIZEOFCMDS_MULTI: u32 = SEG_CMD_SIZE + // __PAGEZERO
    SEG_CMD_SIZE + 2 * SECT_SIZE + // __TEXT + __text + __cstring
    DATAC_SEG_CMD_SIZE + // __DATA_CONST + __got
    SEG_CMD_SIZE + // __LINKEDIT
    FIXUPS_CMD_SIZE +
    DYLINKER_CMD_SIZE +
    MAIN_CMD_SIZE +
    DYLIB_CMD_SIZE +
    SIG_CMD_SIZE;

/// Assemble the multi-segment output image: __TEXT carries the code plus an
/// `__cstring` section, a new `__DATA_CONST` segment carries a `__got` with one
/// slot per import, and `LC_DYLD_CHAINED_FIXUPS` names the imports so dyld binds
/// the slots at load. Vmaddrs are reported in `Layout` for `applyDataRelocs`.
fn assembleMulti(
    gpa: std.mem.Allocator,
    identifier: []const u8,
    code: []const u8,
    entry_text_off: u32,
    cstrings: []const u8,
    imports: []const Import,
) !Layout {
    const code_file_off: u32 = std.mem.alignForward(u32, HEADER_SIZE + SIZEOFCMDS_MULTI, 4);
    const text_size: u32 = @intCast(code.len);

    std.debug.assert(entry_text_off < text_size);

    // __cstring follows __text, 4-byte aligned, in the SAME __TEXT page (so the
    // adrp page delta from code to a string is small / zero, matching clang).
    const cstring_file_off: u32 = std.mem.alignForward(u32, code_file_off + text_size, 4);
    const cstring_end: u32 = cstring_file_off + @as(u32, @intCast(cstrings.len));
    // __TEXT (header + cmds + code + cstrings) spans whole 0x4000 pages. A program
    // that fits one page yields text_seg_size == PAGE — byte-identical to the
    // original single-page layout; larger programs grow __TEXT (and slide the
    // segments after it) by whole pages.
    const text_seg_size: u32 = @intCast(std.mem.alignForward(u64, cstring_end, PAGE));

    // __DATA_CONST is the page after __TEXT; __got holds one 8-byte slot per import.
    const datac_file_off: u32 = text_seg_size;
    const got_size: u32 = @intCast(imports.len * 8);
    std.debug.assert(got_size <= PAGE);

    // __LINKEDIT is the page after that: the fixups blob first, then the sig.
    const linkedit_off: u32 = text_seg_size + @as(u32, @intCast(PAGE));
    const fixups_off: u32 = linkedit_off;

    const fixups_blob = try buildChainedFixups(gpa, imports, datac_file_off, NSEGS_MULTI, DATAC_SEG_INDEX);
    defer gpa.free(fixups_blob);

    const after_fixups: u32 = fixups_off + @as(u32, @intCast(fixups_blob.len));
    const sig_file_off: u32 = std.mem.alignForward(u32, after_fixups, 16);
    const sig_len: u32 = CodeSign.signatureLen(identifier, sig_file_off);

    const total: u32 = sig_file_off + sig_len;
    const le_filesize: u32 = total - linkedit_off;

    // Segment vmaddrs (one page each): __TEXT at BASE, __DATA_CONST one page up,
    // __LINKEDIT two pages up.
    const text_vmaddr: u64 = BASE; // section addr is BASE + file off
    const datac_vmaddr: u64 = BASE + datac_file_off;
    const linkedit_vmaddr: u64 = BASE + linkedit_off;
    const cstring_vmaddr: u64 = BASE + cstring_file_off;
    const got_vmaddr: u64 = datac_vmaddr; // __got is at __DATA_CONST's base

    const image = try gpa.alloc(u8, total);
    errdefer gpa.free(image);
    @memset(image, 0);

    var w = Writer{ .buf = image, .pos = 0 };

    w.put32(MH_MAGIC_64);
    w.put32(CPU_TYPE_ARM64);
    w.put32(CPU_SUBTYPE_ARM64_ALL);
    w.put32(MH_EXECUTE);
    w.put32(NCMDS_MULTI);
    w.put32(SIZEOFCMDS_MULTI);
    w.put32(MH_FLAGS);
    w.put32(0);

    // [0] __PAGEZERO
    w.segment("__PAGEZERO", .{
        .vmaddr = 0,
        .vmsize = BASE,
        .fileoff = 0,
        .filesize = 0,
        .maxprot = 0,
        .initprot = 0,
        .nsects = 0,
    });

    // [1] __TEXT (+ __text + __cstring)
    w.put32(LC_SEGMENT_64);
    w.put32(SEG_CMD_SIZE + 2 * SECT_SIZE);
    w.name16("__TEXT");
    w.put64(text_vmaddr);
    w.put64(text_seg_size); // vmsize: whole-page span
    w.put64(0);
    w.put64(text_seg_size); // filesize: whole-page span
    w.put32(VM_PROT_READ | VM_PROT_EXECUTE); // r-x
    w.put32(VM_PROT_READ | VM_PROT_EXECUTE); // r-x
    w.put32(2); // __text + __cstring
    w.put32(0);
    // section_64 __text
    w.name16("__text");
    w.name16("__TEXT");
    w.put64(text_vmaddr + code_file_off);
    w.put64(text_size);
    w.put32(code_file_off);
    w.put32(2); // align 2^2 = 4
    w.put32(0);
    w.put32(0);
    w.put32(TEXT_SECT_FLAGS);
    w.put32(0);
    w.put32(0);
    w.put32(0);
    // section_64 __cstring
    w.name16("__cstring");
    w.name16("__TEXT");
    w.put64(cstring_vmaddr);
    w.put64(@intCast(cstrings.len));
    w.put32(cstring_file_off);
    w.put32(0); // align 2^0 = 1
    w.put32(0);
    w.put32(0);
    w.put32(S_CSTRING_LITERALS);
    w.put32(0);
    w.put32(0);
    w.put32(0);

    // [2] __DATA_CONST (+ __got)
    w.put32(LC_SEGMENT_64);
    w.put32(DATAC_SEG_CMD_SIZE);
    w.name16("__DATA_CONST");
    w.put64(datac_vmaddr);
    w.put64(PAGE); // vmsize: full page
    w.put64(datac_file_off);
    w.put64(PAGE); // filesize: full page
    w.put32(VM_PROT_READ | VM_PROT_WRITE); // rw-
    w.put32(VM_PROT_READ | VM_PROT_WRITE); // rw-
    w.put32(1); // __got
    w.put32(SG_READ_ONLY);
    // section_64 __got
    w.name16("__got");
    w.name16("__DATA_CONST");
    w.put64(got_vmaddr);
    w.put64(got_size);
    w.put32(datac_file_off);
    w.put32(3); // align 2^3 = 8
    w.put32(0);
    w.put32(0);
    w.put32(S_NON_LAZY_SYMBOL_POINTERS);
    w.put32(0); // indirect-symtab index; unused, we use chained fixups
    w.put32(0);
    w.put32(0);

    // [3] __LINKEDIT
    w.segment("__LINKEDIT", .{
        .vmaddr = linkedit_vmaddr,
        // vmsize must cover filesize; round up so it can never be smaller.
        .vmsize = std.mem.alignForward(u32, le_filesize, PAGE),
        .fileoff = linkedit_off,
        .filesize = le_filesize,
        .maxprot = VM_PROT_READ,
        .initprot = VM_PROT_READ,
        .nsects = 0,
    });

    // LC_DYLD_CHAINED_FIXUPS.
    w.put32(LC_DYLD_CHAINED_FIXUPS);
    w.put32(FIXUPS_CMD_SIZE);
    w.put32(fixups_off);
    w.put32(@intCast(fixups_blob.len));

    // LC_LOAD_DYLINKER /usr/lib/dyld.
    w.put32(LC_LOAD_DYLINKER);
    w.put32(DYLINKER_CMD_SIZE);
    w.put32(12);
    w.path(DYLD_PATH, DYLINKER_CMD_SIZE, 12);

    // LC_MAIN.
    w.put32(LC_MAIN);
    w.put32(MAIN_CMD_SIZE);
    w.put64(code_file_off + entry_text_off); // entryoff = file offset of `main`
    w.put64(0);

    // LC_LOAD_DYLIB /usr/lib/libSystem.B.dylib.
    w.put32(LC_LOAD_DYLIB);
    w.put32(DYLIB_CMD_SIZE);
    w.put32(24);
    w.put32(2);
    w.put32(0x051F1304); // current_version
    w.put32(0x00010000); // compatibility_version
    w.path(LIBSYSTEM_PATH, DYLIB_CMD_SIZE, 24);

    // LC_CODE_SIGNATURE (LAST).
    w.put32(LC_CODE_SIGNATURE);
    w.put32(SIG_CMD_SIZE);
    w.put32(sig_file_off);
    w.put32(sig_len);

    std.debug.assert(w.pos == HEADER_SIZE + SIZEOFCMDS_MULTI);

    @memcpy(image[code_file_off..][0..code.len], code);
    @memcpy(image[cstring_file_off..][0..cstrings.len], cstrings);

    // __DATA_CONST: seed each __got slot with the bind sentinel.
    var i: usize = 0;
    while (i < imports.len) : (i += 1) {
        std.mem.writeInt(u64, image[datac_file_off + i * 8 ..][0..8], GOT_BIND_SENTINEL, .little);
    }

    // __LINKEDIT: the chained-fixups blob (sig region stays zeroed).
    @memcpy(image[fixups_off..][0..fixups_blob.len], fixups_blob);

    return .{
        .image = image,
        .code_file_off = code_file_off,
        .text_size = text_size,
        .sig_file_off = sig_file_off,
        .sig_len = sig_len,
        .text_vmaddr = text_vmaddr,
        .cstring_vmaddr = cstring_vmaddr,
        .got_vmaddr = got_vmaddr,
    };
}

// Segment count and the __DATA_CONST index for the multi-seg chained-fixups
// `starts_in_image` table (PAGEZERO, TEXT, DATA_CONST, LINKEDIT).
const NSEGS_MULTI: u32 = 4;
const DATAC_SEG_INDEX: u32 = 2;

//
// Produces the LC_DYLD_CHAINED_FIXUPS payload, verified byte-for-byte against
// the clang `write` reference (one libSystem import, one __got slot). Layout:
//   header (dyld_chained_fixups_header, 28 bytes, padded to 0x20)
//   starts_in_image @0x20: seg_count + per-seg offsets (only DATA_CONST set)
//   starts_in_segment (24 bytes): page_size 0x4000, pointer_format 6, the
//     segment file offset, page_count 1, page_start[0] 0
//   imports @imports_off: one u32 per import (ordinal|weak<<8|name_offset<<9)
//   symbols @symbols_off: leading NUL then each name NUL-terminated
// All scalars little-endian. `datac_fileoff` is the __DATA_CONST file offset
// dyld walks the chain over (the `segment_offset` field).

const DYLD_CHAINED_PTR_64_OFFSET: u16 = 6;

/// Build the parameterized chained-fixups blob. Caller owns the returned bytes.
fn buildChainedFixups(
    gpa: std.mem.Allocator,
    imports: []const Import,
    datac_fileoff: u64,
    seg_count: u32,
    datac_seg_index: u32,
) ![]u8 {
    std.debug.assert(datac_seg_index < seg_count);
    // A program may have __cstring data but NO imports (a `str` materialized but
    // never `print`ed). Then there is no __got slot and no fixup chain — emit a
    // valid blob with imports_count 0 and no segment chain (so dyld binds nothing).
    const has_fixups = imports.len > 0;

    // starts_in_image begins at 0x20 (after the 28-byte header padded to 8).
    const starts_image_off: u32 = 0x20;
    // starts_in_image = u32 seg_count + seg_count * u32 offset, aligned to 8.
    const starts_image_size: u32 = std.mem.alignForward(u32, 4 + seg_count * 4, 8);
    const seg_struct_off: u32 = starts_image_off + starts_image_size;
    const SEG_STRUCT_SIZE: u32 = 24; // starts_in_segment (no extra page_starts)

    // No imports → no starts_in_segment struct.
    const imports_off: u32 = if (has_fixups) seg_struct_off + SEG_STRUCT_SIZE else seg_struct_off;
    const imports_size: u32 = @intCast(imports.len * 4);
    const symbols_off: u32 = imports_off + imports_size;

    // Symbol table: a leading NUL, then each import name NUL-terminated.
    var sym_size: u32 = 1; // leading NUL
    for (imports) |im| sym_size += @intCast(im.name.len + 1);

    const total: u32 = std.mem.alignForward(u32, symbols_off + sym_size, 8);

    const blob = try gpa.alloc(u8, total);
    errdefer gpa.free(blob);
    @memset(blob, 0);

    var w = Writer{ .buf = blob, .pos = 0 };

    // --- dyld_chained_fixups_header -----------------------------------------
    w.put32(0); // fixups_version
    w.put32(starts_image_off); // starts_offset
    w.put32(imports_off); // imports_offset
    w.put32(symbols_off); // symbols_offset
    w.put32(@intCast(imports.len)); // imports_count
    w.put32(1); // imports_format = DYLD_CHAINED_IMPORT
    w.put32(0); // symbols_format = uncompressed
    w.pos = starts_image_off; // pad to 0x20

    // --- starts_in_image ----------------------------------------------------
    w.put32(seg_count);
    var s: u32 = 0;
    while (s < seg_count) : (s += 1) {
        // Only __DATA_CONST has a fixup chain, and only when there are imports.
        w.put32(if (has_fixups and s == datac_seg_index) seg_struct_off - starts_image_off else 0);
    }

    // --- starts_in_segment (only when there is a chain to describe) ----------
    if (has_fixups) {
        w.pos = seg_struct_off; // align to the seg struct
        w.put32(SEG_STRUCT_SIZE); // size
        w.put16(@intCast(PAGE)); // page_size (0x4000)
        w.put16(DYLD_CHAINED_PTR_64_OFFSET); // pointer_format = 6
        w.put64(datac_fileoff); // segment_offset (file off dyld walks)
        w.put32(0); // max_valid_pointer
        w.put16(1); // page_count
        w.put16(0); // page_start[0] (first fixup at slot 0)
    }

    // --- imports ------------------------------------------------------------
    w.pos = imports_off; // (padding bytes between sections stay zero)
    // Name offsets index into the symbols table, which has a leading NUL, so the
    // first name starts at offset 1.
    var name_off: u32 = 1;
    for (imports) |im| {
        // dyld_chained_import: ordinal:8 | weak:1 | name_offset:23.
        const ordinal: u32 = 1; // libSystem is the only LC_LOAD_DYLIB → ordinal 1
        const entry: u32 = (ordinal & 0xFF) | (name_off << 9);
        w.put32(entry);
        name_off += @intCast(im.name.len + 1);
    }

    // --- symbols ------------------------------------------------------------
    std.debug.assert(w.pos == symbols_off);
    w.put8(0); // leading NUL (name_offset 0 = unused/anonymous)
    for (imports) |im| {
        @memcpy(blob[w.pos..][0..im.name.len], im.name);
        w.pos += im.name.len;
        w.put8(0);
    }

    return blob;
}

// Little-endian byte writer.

const Writer = struct {
    buf: []u8,
    pos: usize,

    fn put8(self: *Writer, v: u8) void {
        self.buf[self.pos] = v;
        self.pos += 1;
    }

    fn put16(self: *Writer, v: u16) void {
        std.mem.writeInt(u16, self.buf[self.pos..][0..2], v, .little);
        self.pos += 2;
    }

    fn put32(self: *Writer, v: u32) void {
        std.mem.writeInt(u32, self.buf[self.pos..][0..4], v, .little);
        self.pos += 4;
    }

    fn put64(self: *Writer, v: u64) void {
        std.mem.writeInt(u64, self.buf[self.pos..][0..8], v, .little);
        self.pos += 8;
    }

    /// Write a 16-byte segment/section name field, NUL-padded.
    fn name16(self: *Writer, n: []const u8) void {
        std.debug.assert(n.len <= 16);
        @memset(self.buf[self.pos..][0..16], 0);
        @memcpy(self.buf[self.pos..][0..n.len], n);
        self.pos += 16;
    }

    /// Write a NUL-terminated path inside a load command, then advance to the
    /// command's end (the bytes between path end and `cmdsize` stay zero).
    fn path(self: *Writer, p: []const u8, cmdsize: u32, name_off: u32) void {
        @memcpy(self.buf[self.pos..][0..p.len], p);
        // Advance to cmd_start + cmdsize. The command started `name_off` bytes
        // before the current position (we are right after the head fields).
        const cmd_start = self.pos - name_off;
        self.pos = cmd_start + cmdsize;
    }

    const SegArgs = struct {
        vmaddr: u64,
        vmsize: u64,
        fileoff: u64,
        filesize: u64,
        maxprot: u32,
        initprot: u32,
        nsects: u32,
    };

    /// Write a section-less LC_SEGMENT_64 (used for __PAGEZERO and __LINKEDIT).
    fn segment(self: *Writer, n: []const u8, a: SegArgs) void {
        self.put32(LC_SEGMENT_64);
        self.put32(SEG_CMD_SIZE);
        self.name16(n);
        self.put64(a.vmaddr);
        self.put64(a.vmsize);
        self.put64(a.fileoff);
        self.put64(a.filesize);
        self.put32(a.maxprot);
        self.put32(a.initprot);
        self.put32(a.nsects);
        self.put32(0);
    }
};

// Tests — parse our own header/load commands back and assert key fields against
// the documented byte map / `otool -hlv` ground truth.

const testing = std.testing;

/// A tiny but representative code blob (a few words). Content is irrelevant to
/// the structural assertions; only its length matters for layout.
const stub_code = [_]u8{
    0xFD, 0x7B, 0xBF, 0xA9, // stp x29,x30,[sp,#-16]!
    0xFD, 0x03, 0x00, 0x91, // mov x29,sp
    0x00, 0x05, 0x80, 0xD2, // movz x0,#40
    0xFD, 0x7B, 0xC1, 0xA8, // ldp x29,x30,[sp],#16
    0xC0, 0x03, 0x5F, 0xD6, // ret
};

fn rd32(image: []const u8, off: usize) u32 {
    return std.mem.readInt(u32, image[off..][0..4], .little);
}
fn rd64(image: []const u8, off: usize) u64 {
    return std.mem.readInt(u64, image[off..][0..8], .little);
}

test "header fields" {
    const layout = try assemble(testing.allocator, "stub", &stub_code, 0, &.{}, &.{});
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    try testing.expectEqual(MH_MAGIC_64, rd32(img, 0));
    try testing.expectEqual(CPU_TYPE_ARM64, rd32(img, 4));
    try testing.expectEqual(CPU_SUBTYPE_ARM64_ALL, rd32(img, 8));
    try testing.expectEqual(MH_EXECUTE, rd32(img, 12));
    try testing.expectEqual(NCMDS, rd32(img, 16));
    try testing.expectEqual(SIZEOFCMDS, rd32(img, 20));
    try testing.expectEqual(MH_FLAGS, rd32(img, 24));
    try testing.expectEqual(@as(u32, 0), rd32(img, 28));
    // ground truth: sizeofcmds is 440 (0x1B8) for the 8-command layout.
    try testing.expectEqual(@as(u32, 0x1B8), SIZEOFCMDS);
    try testing.expectEqual(@as(u32, 8), NCMDS);
}

test "segments and load command order" {
    const layout = try assemble(testing.allocator, "stub", &stub_code, 0, &.{}, &.{});
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    var off: usize = HEADER_SIZE;
    var cmds: [NCMDS]u32 = undefined;
    var i: usize = 0;
    while (i < NCMDS) : (i += 1) {
        const cmd = rd32(img, off);
        const cmdsize = rd32(img, off + 4);
        cmds[i] = cmd;
        off += cmdsize;
    }
    // After the last command we are exactly at the code blob start.
    try testing.expectEqual(@as(usize, HEADER_SIZE + SIZEOFCMDS), off);

    try testing.expectEqual(LC_SEGMENT_64, cmds[0]); // __PAGEZERO
    try testing.expectEqual(LC_SEGMENT_64, cmds[1]); // __TEXT
    try testing.expectEqual(LC_SEGMENT_64, cmds[2]); // __LINKEDIT
    try testing.expectEqual(LC_DYLD_CHAINED_FIXUPS, cmds[3]);
    try testing.expectEqual(LC_LOAD_DYLINKER, cmds[4]);
    try testing.expectEqual(LC_MAIN, cmds[5]);
    try testing.expectEqual(LC_LOAD_DYLIB, cmds[6]);
    try testing.expectEqual(LC_CODE_SIGNATURE, cmds[7]); // MUST be last
}

test "__PAGEZERO / __TEXT / __LINKEDIT geometry" {
    const layout = try assemble(testing.allocator, "stub", &stub_code, 0, &.{}, &.{});
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    // __PAGEZERO at HEADER_SIZE: vmaddr 0, vmsize 0x1_0000_0000, no file.
    const pz = HEADER_SIZE;
    try testing.expectEqual(@as(u64, 0), rd64(img, pz + 24)); // vmaddr
    try testing.expectEqual(BASE, rd64(img, pz + 32)); // vmsize
    try testing.expectEqual(@as(u64, 0), rd64(img, pz + 48)); // filesize

    // __TEXT: vmaddr BASE, fileoff 0, filesize PAGE, nsects 1.
    const tx = pz + SEG_CMD_SIZE;
    try testing.expectEqual(BASE, rd64(img, tx + 24)); // vmaddr
    try testing.expectEqual(PAGE, rd64(img, tx + 32)); // vmsize
    try testing.expectEqual(@as(u64, 0), rd64(img, tx + 40)); // fileoff
    try testing.expectEqual(PAGE, rd64(img, tx + 48)); // filesize
    try testing.expectEqual(@as(u32, 1), rd32(img, tx + 64)); // nsects
    // __text section addr == BASE + code_file_off, size == text_size.
    const sect = tx + SEG_CMD_SIZE;
    try testing.expectEqual(BASE + layout.code_file_off, rd64(img, sect + 32)); // addr
    try testing.expectEqual(@as(u64, layout.text_size), rd64(img, sect + 40)); // size
    try testing.expectEqual(layout.code_file_off, rd32(img, sect + 48)); // offset

    // __LINKEDIT: fileoff == PAGE.
    const le = tx + TEXT_SEG_CMD_SIZE;
    try testing.expectEqual(PAGE, rd64(img, le + 40)); // fileoff
}

test "chained fixups, LC_MAIN entryoff, and sig placement" {
    const layout = try assemble(testing.allocator, "stub", &stub_code, 0, &.{}, &.{});
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    var off: usize = HEADER_SIZE;
    var fixups_datasize: ?u32 = null;
    var main_entryoff: ?u64 = null;
    var sig_dataoff: ?u32 = null;
    var sig_datasize: ?u32 = null;
    var i: usize = 0;
    while (i < NCMDS) : (i += 1) {
        const cmd = rd32(img, off);
        const cmdsize = rd32(img, off + 4);
        switch (cmd) {
            LC_DYLD_CHAINED_FIXUPS => fixups_datasize = rd32(img, off + 12),
            LC_MAIN => main_entryoff = rd64(img, off + 8),
            LC_CODE_SIGNATURE => {
                sig_dataoff = rd32(img, off + 8);
                sig_datasize = rd32(img, off + 12);
            },
            else => {},
        }
        off += cmdsize;
    }

    try testing.expectEqual(@as(u32, 56), fixups_datasize.?);
    try testing.expectEqual(@as(u64, layout.code_file_off), main_entryoff.?);
    try testing.expectEqual(layout.sig_file_off, sig_dataoff.?);
    try testing.expectEqual(layout.sig_len, sig_datasize.?);

    // Code fits in the first page.
    try testing.expect(layout.code_file_off + stub_code.len <= PAGE);
    // sig region is at 0x4040 for tiny code (0x4000 + 56 -> align16 = 0x4040).
    try testing.expectEqual(@as(u32, 0x4040), layout.sig_file_off);
    // The 56-byte fixups blob is present verbatim at the page boundary.
    try testing.expect(std.mem.eql(u8, img[PAGE..][0..56], &CHAINED_FIXUPS_BLOB));
}

test "nonzero entry_text_off shifts LC_MAIN entryoff" {
    // When `main` is not first in __text, its resolved offset is non-zero and the
    // entryoff must be code_file_off + that offset (so the kernel jumps to main).
    const layout = try assemble(testing.allocator, "stub", &stub_code, 8, &.{}, &.{});
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    var off: usize = HEADER_SIZE;
    var main_entryoff: ?u64 = null;
    var i: usize = 0;
    while (i < NCMDS) : (i += 1) {
        const cmd = rd32(img, off);
        const cmdsize = rd32(img, off + 4);
        if (cmd == LC_MAIN) main_entryoff = rd64(img, off + 8);
        off += cmdsize;
    }
    try testing.expectEqual(@as(u64, layout.code_file_off) + 8, main_entryoff.?);
}

test "code spanning multiple pages grows __TEXT instead of being rejected" {
    // A blob larger than a page (once headers are added) used to be rejected; it
    // now lays __TEXT across as many whole pages as it needs, sliding __LINKEDIT
    // after it. Verify the segment sizes and that the image is exactly that big.
    const gpa = testing.allocator;
    const big = try gpa.alloc(u8, @intCast(PAGE));
    defer gpa.free(big);
    @memset(big, 0);
    // 4-byte-aligned blob so the layout assertions hold (real code is too).
    std.mem.writeInt(u32, big[0..4], 0xD65F03C0, .little); // ret at entry

    const layout = try assemble(gpa, "stub", big, 0, &.{}, &.{});
    defer gpa.free(layout.image);

    // header+cmds+PAGE bytes of code spill into a second page -> __TEXT is 2 pages.
    const text_seg_size = std.mem.alignForward(u64, @as(u64, layout.code_file_off) + big.len, PAGE);
    try testing.expectEqual(@as(u64, 2 * PAGE), text_seg_size);
    // __LINKEDIT (hence the whole image) starts after the grown __TEXT.
    try testing.expect(layout.sig_file_off >= text_seg_size);
}

// Multi-segment (output) path — exercised by any program that calls `print`.
// Ground truth is the clang `write` reference dumped on this host
// (`otool -hlv`, `xxd`, `dyld_info -fixups`).

/// "hello world\n\0", the demo's interned __cstring blob.
const demo_cstrings = "hello world\n\x00";
const demo_imports = [_]Import{.{ .name = "_write" }};

/// The 96-byte chained-fixups blob captured verbatim from the clang `write`
/// reference (`xxd -s 0x8000 -l 96 /tmp/ref`). Our builder must reproduce it
/// for the one-import, 4-segment case.
const CLANG_FIXUPS_96 = [96]u8{
    0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
    0x50, 0x00, 0x00, 0x00, 0x54, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x18, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x18, 0x00, 0x00, 0x00, 0x00, 0x40, 0x06, 0x00,
    0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x02, 0x00, 0x00, 0x00, 0x5f, 0x77, 0x72,
    0x69, 0x74, 0x65, 0x00, 0x00, 0x00, 0x00, 0x00,
};

test "buildChainedFixups reproduces the clang 96-byte blob" {
    const blob = try buildChainedFixups(testing.allocator, &demo_imports, PAGE, NSEGS_MULTI, DATAC_SEG_INDEX);
    defer testing.allocator.free(blob);
    try testing.expectEqual(@as(usize, 96), blob.len);
    try testing.expectEqualSlices(u8, &CLANG_FIXUPS_96, blob);
}

test "multi-seg: 9 commands in the expected order" {
    const layout = try assemble(testing.allocator, "hello", &stub_code, 0, demo_cstrings, &demo_imports);
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    try testing.expectEqual(NCMDS_MULTI, rd32(img, 16)); // ncmds
    try testing.expectEqual(SIZEOFCMDS_MULTI, rd32(img, 20)); // sizeofcmds

    var off: usize = HEADER_SIZE;
    var cmds: [NCMDS_MULTI]u32 = undefined;
    var i: usize = 0;
    while (i < NCMDS_MULTI) : (i += 1) {
        cmds[i] = rd32(img, off);
        off += rd32(img, off + 4);
    }
    try testing.expectEqual(@as(usize, HEADER_SIZE + SIZEOFCMDS_MULTI), off);

    try testing.expectEqual(LC_SEGMENT_64, cmds[0]); // __PAGEZERO
    try testing.expectEqual(LC_SEGMENT_64, cmds[1]); // __TEXT
    try testing.expectEqual(LC_SEGMENT_64, cmds[2]); // __DATA_CONST
    try testing.expectEqual(LC_SEGMENT_64, cmds[3]); // __LINKEDIT
    try testing.expectEqual(LC_DYLD_CHAINED_FIXUPS, cmds[4]);
    try testing.expectEqual(LC_LOAD_DYLINKER, cmds[5]);
    try testing.expectEqual(LC_MAIN, cmds[6]);
    try testing.expectEqual(LC_LOAD_DYLIB, cmds[7]);
    try testing.expectEqual(LC_CODE_SIGNATURE, cmds[8]); // MUST be last
}

test "multi-seg: __TEXT has __text + __cstring; __cstring carries the bytes" {
    const layout = try assemble(testing.allocator, "hello", &stub_code, 0, demo_cstrings, &demo_imports);
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    // __TEXT is the second command (after __PAGEZERO); nsects == 2.
    const tx = HEADER_SIZE + SEG_CMD_SIZE;
    try testing.expectEqual(BASE, rd64(img, tx + 24)); // vmaddr
    try testing.expectEqual(@as(u32, 2), rd32(img, tx + 64)); // nsects
    // __text section
    const text_sect = tx + SEG_CMD_SIZE;
    try testing.expectEqual(BASE + layout.code_file_off, rd64(img, text_sect + 32)); // addr
    try testing.expectEqual(@as(u64, layout.text_size), rd64(img, text_sect + 40)); // size
    try testing.expectEqual(TEXT_SECT_FLAGS, rd32(img, text_sect + 64)); // flags
    // __cstring section (immediately after __text section_64).
    const cstr_sect = text_sect + SECT_SIZE;
    try testing.expectEqualStrings("__cstring", std.mem.sliceTo(img[cstr_sect..][0..16], 0));
    try testing.expectEqual(layout.cstring_vmaddr, rd64(img, cstr_sect + 32)); // addr
    try testing.expectEqual(@as(u64, demo_cstrings.len), rd64(img, cstr_sect + 40)); // size
    try testing.expectEqual(S_CSTRING_LITERALS, rd32(img, cstr_sect + 64)); // flags
    // The bytes live at the section's file offset.
    const cstr_off = rd32(img, cstr_sect + 48);
    try testing.expectEqualSlices(u8, demo_cstrings, img[cstr_off..][0..demo_cstrings.len]);
}

test "multi-seg: __DATA_CONST geometry, __got, and the seeded slot" {
    const layout = try assemble(testing.allocator, "hello", &stub_code, 0, demo_cstrings, &demo_imports);
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    // __DATA_CONST is the third command. Walk to it.
    const dc = HEADER_SIZE + SEG_CMD_SIZE + (SEG_CMD_SIZE + 2 * SECT_SIZE);
    try testing.expectEqual(LC_SEGMENT_64, rd32(img, dc));
    try testing.expectEqualStrings("__DATA_CONST", std.mem.sliceTo(img[dc + 8 ..][0..16], 0));
    try testing.expectEqual(BASE + PAGE, rd64(img, dc + 24)); // vmaddr
    try testing.expectEqual(PAGE, rd64(img, dc + 48)); // filesize
    try testing.expectEqual(VM_PROT_READ | VM_PROT_WRITE, rd32(img, dc + 56)); // maxprot rw-
    try testing.expectEqual(VM_PROT_READ | VM_PROT_WRITE, rd32(img, dc + 60)); // initprot rw-
    try testing.expectEqual(@as(u32, 1), rd32(img, dc + 64)); // nsects
    try testing.expectEqual(SG_READ_ONLY, rd32(img, dc + 68)); // seg flags
    // __got section
    const got = dc + SEG_CMD_SIZE;
    try testing.expectEqualStrings("__got", std.mem.sliceTo(img[got..][0..16], 0));
    try testing.expectEqual(layout.got_vmaddr, rd64(img, got + 32)); // addr
    try testing.expectEqual(@as(u64, 8), rd64(img, got + 40)); // size (one import)
    try testing.expectEqual(S_NON_LAZY_SYMBOL_POINTERS, rd32(img, got + 64)); // flags
    // The on-disk got slot is the bind sentinel (file offset == 0x4000).
    const got_off = rd32(img, got + 48);
    try testing.expectEqual(@as(u32, @intCast(PAGE)), got_off);
    try testing.expectEqual(GOT_BIND_SENTINEL, rd64(img, got_off));
}

test "multi-seg: fixups blob + sig placement, sig last" {
    const layout = try assemble(testing.allocator, "hello", &stub_code, 0, demo_cstrings, &demo_imports);
    defer testing.allocator.free(layout.image);
    const img = layout.image;

    // The chained-fixups blob sits at __LINKEDIT's start (2 pages in) and equals
    // the clang ground truth.
    try testing.expectEqualSlices(u8, &CLANG_FIXUPS_96, img[2 * PAGE ..][0..96]);

    // LC_CODE_SIGNATURE dataoff/datasize and codeLimit (== sig_file_off).
    var off: usize = HEADER_SIZE;
    var sig_dataoff: ?u32 = null;
    var fixups_datasize: ?u32 = null;
    var i: usize = 0;
    while (i < NCMDS_MULTI) : (i += 1) {
        const cmd = rd32(img, off);
        switch (cmd) {
            LC_CODE_SIGNATURE => sig_dataoff = rd32(img, off + 8),
            LC_DYLD_CHAINED_FIXUPS => fixups_datasize = rd32(img, off + 12),
            else => {},
        }
        off += rd32(img, off + 4);
    }
    try testing.expectEqual(@as(u32, 96), fixups_datasize.?);
    try testing.expectEqual(layout.sig_file_off, sig_dataoff.?);
    // The sig region is the last thing in the file.
    try testing.expectEqual(@as(usize, layout.sig_file_off + layout.sig_len), img.len);
}
