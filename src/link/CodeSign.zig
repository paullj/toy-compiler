//! Ad-hoc Mach-O code signature (SuperBlob + CodeDirectory).
//!
//! WHY: macOS refuses to run an unsigned arm64 binary. M1 emits its own ad-hoc
//! signature (no X.509 certificate): a SuperBlob wrapping a single CodeDirectory
//! whose code-slot hashes are SHA-256 of each page of the file up to the
//! signature's own offset (`codeLimit`). All SuperBlob/CodeDirectory scalar
//! fields are BIG-endian (Apple's legacy convention); the SHA-256 digests are
//! raw bytes.
//!
//! The CodeDirectory header layout (v0x20400, 88-byte fixed head through the
//! exec-segment fields) was verified byte-for-byte against a `cc`-built ad-hoc
//! reference via `xxd`/`codesign -dvvv`. Field offsets and values mirror that
//! ground truth exactly; the kernel re-hashes the code pages at exec time and
//! accepts the binary because the digests match (no certificate is needed for
//! the ad-hoc flag).

const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;

/// CodeDirectory fixed header size through the v0x20400 extension fields, which
/// is exactly the identifier offset (verified field-by-field against /tmp/ref).
const CD_HEADER_SIZE: u32 = 88;
const HASH_SIZE: u32 = 32; // SHA-256
const PAGE_SIZE: u32 = 4096; // code-hash page (NOT the 0x4000 segment page)

const SUPERBLOB_HEADER: u32 = 12; // magic + length + count
const BLOB_INDEX: u32 = 8; // type + offset

// --- Signature magic / version / flag constants (verified vs. ground truth) --

const CSMAGIC_EMBEDDED_SIGNATURE: u32 = 0xFADE0CC0;
const CSMAGIC_CODEDIRECTORY: u32 = 0xFADE0C02;
const CSSLOT_CODEDIRECTORY: u32 = 0;
const CS_SUPPORTSEXECSEG: u32 = 0x20400; // CodeDirectory version
const CS_ADHOC: u32 = 0x00000002;
const CS_LINKER_SIGNED: u32 = 0x00020000;
// Matches `codesign`'s own ad-hoc output and verifies cleanly (adhoc,linker-signed).
const CD_FLAGS: u32 = CS_ADHOC | CS_LINKER_SIGNED; // 0x00020002
const CS_HASHTYPE_SHA256: u8 = 2;
const CS_PAGE_SIZE_SHIFT: u8 = 12; // 2^12 = 4096
const CS_EXECSEG_MAIN_BINARY: u64 = 0x1;

/// Number of code-hash slots covering `[0, code_limit)`, including a partial
/// final page.
fn nCodeSlots(code_limit: u32) u32 {
    return (code_limit + PAGE_SIZE - 1) / PAGE_SIZE;
}

/// Deterministic total length of the signature blob for the given identifier and
/// code limit. Layout reserves exactly this many bytes.
pub fn signatureLen(identifier: []const u8, code_limit: u32) u32 {
    const ident_len: u32 = @intCast(identifier.len);
    const hash_offset: u32 = CD_HEADER_SIZE + ident_len + 1; // +1 NUL
    const cd_len: u32 = hash_offset + nCodeSlots(code_limit) * HASH_SIZE;
    return SUPERBLOB_HEADER + BLOB_INDEX + cd_len;
}

/// Big-endian byte writer over a pre-sized buffer. Every SuperBlob/CodeDirectory
/// scalar is big-endian (Apple's legacy convention); only the SHA-256 digests
/// are copied raw.
const BeWriter = struct {
    buf: []u8,
    pos: usize,

    fn put32(self: *BeWriter, v: u32) void {
        std.mem.writeInt(u32, self.buf[self.pos..][0..4], v, .big);
        self.pos += 4;
    }
    fn put64(self: *BeWriter, v: u64) void {
        std.mem.writeInt(u64, self.buf[self.pos..][0..8], v, .big);
        self.pos += 8;
    }
    fn put8(self: *BeWriter, v: u8) void {
        self.buf[self.pos] = v;
        self.pos += 1;
    }
};

/// Sign `image` in place: hash `image[0..sig_file_off)` and write the SuperBlob +
/// CodeDirectory into the reserved region at `sig_file_off`.
///
/// `image` already has the signature region reserved/zeroed by `MachO.assemble`,
/// and `codeLimit == sig_file_off` so the hashed bytes are final before we touch
/// the region. `text_size` is the `__text` section size (used for the
/// exec-segment limit). The `gpa` parameter is unused — hashing streams directly
/// over `image` — but is kept to match the planned signature.
pub fn sign(
    gpa: std.mem.Allocator,
    image: []u8,
    identifier: []const u8,
    sig_file_off: u32,
    text_size: u32,
) !void {
    _ = gpa;

    const code_limit: u32 = sig_file_off;
    const ident_len: u32 = @intCast(identifier.len);
    const n_slots: u32 = nCodeSlots(code_limit);
    const hash_offset: u32 = CD_HEADER_SIZE + ident_len + 1; // +1 NUL
    const cd_len: u32 = hash_offset + n_slots * HASH_SIZE;
    const sig_len: u32 = SUPERBLOB_HEADER + BLOB_INDEX + cd_len;

    // Defensive: the region MachO reserved must match what we are about to write.
    std.debug.assert(sig_len == signatureLen(identifier, code_limit));
    std.debug.assert(sig_file_off + sig_len <= image.len);

    // CodeDirectory begins right after the SuperBlob header + the one index.
    const cd_off: u32 = SUPERBLOB_HEADER + BLOB_INDEX; // 20

    var w = BeWriter{ .buf = image[sig_file_off..][0..sig_len], .pos = 0 };

    // --- SuperBlob (CSMAGIC_EMBEDDED_SIGNATURE) -----------------------------
    w.put32(CSMAGIC_EMBEDDED_SIGNATURE);
    w.put32(sig_len); // length (whole SuperBlob)
    w.put32(1); // count: a single sub-blob
    // BlobIndex[0]: the CodeDirectory.
    w.put32(CSSLOT_CODEDIRECTORY); // type
    w.put32(cd_off); // offset from SuperBlob start (= 20)

    std.debug.assert(w.pos == cd_off);

    // --- CodeDirectory (CSMAGIC_CODEDIRECTORY, v0x20400) --------------------
    w.put32(CSMAGIC_CODEDIRECTORY);
    w.put32(cd_len); // length
    w.put32(CS_SUPPORTSEXECSEG); // version 0x20400
    w.put32(CD_FLAGS); // 0x20002 (adhoc | linker-signed)
    w.put32(hash_offset); // where the code-slot hashes begin
    w.put32(CD_HEADER_SIZE); // identOffset (= 88)
    w.put32(0); // nSpecialSlots
    w.put32(n_slots); // nCodeSlots
    w.put32(code_limit); // codeLimit
    w.put8(@intCast(HASH_SIZE)); // hashSize (32)
    w.put8(CS_HASHTYPE_SHA256); // hashType
    w.put8(0); // platform (0 = not a platform binary)
    w.put8(CS_PAGE_SIZE_SHIFT); // pageSize log2 (12 -> 4096)
    w.put32(0); // spare2
    // v0x20100+ fields:
    w.put32(0); // scatterOffset
    w.put32(0); // teamOffset
    // v0x20200+:
    w.put32(0); // spare3
    w.put64(0); // codeLimit64 (0 -> use 32-bit codeLimit)
    // v0x20400 exec-segment fields:
    w.put64(0); // execSegBase (file offset of __TEXT)
    w.put64(text_size); // execSegLimit (__text section size)
    w.put64(CS_EXECSEG_MAIN_BINARY); // execSegFlags

    // The fixed header ends exactly at identOffset.
    std.debug.assert(w.pos == cd_off + CD_HEADER_SIZE);

    // --- Identifier C-string (NUL-terminated) ------------------------------
    @memcpy(w.buf[w.pos..][0..ident_len], identifier);
    w.pos += ident_len;
    w.put8(0); // NUL
    std.debug.assert(w.pos == cd_off + hash_offset);

    // --- Code-slot hashes: SHA-256 of each page up to codeLimit ------------
    // The final page may be partial — clamp the end to codeLimit. The hashed
    // bytes (`image[0..codeLimit)`) are final; only this region beyond is being
    // written, so the digests are stable.
    var slot: u32 = 0;
    while (slot < n_slots) : (slot += 1) {
        const start: u32 = slot * PAGE_SIZE;
        const end: u32 = @min(start + PAGE_SIZE, code_limit);
        var digest: [HASH_SIZE]u8 = undefined;
        Sha256.hash(image[start..end], &digest, .{});
        @memcpy(w.buf[w.pos..][0..HASH_SIZE], &digest);
        w.pos += HASH_SIZE;
    }

    std.debug.assert(w.pos == sig_len);
}

const testing = std.testing;

test "signatureLen matches the documented layout" {
    // ident "ref" (len 3), code_limit 0x4040 -> 5 code slots.
    // hash_offset = 88 + 3 + 1 = 92; cd_len = 92 + 5*32 = 252; sig = 12+8+252.
    try testing.expectEqual(@as(u32, 5), nCodeSlots(0x4040));
    try testing.expectEqual(@as(u32, 12 + 8 + 252), signatureLen("ref", 0x4040));
}

test "sign writes a parseable SuperBlob + CodeDirectory" {
    const gpa = testing.allocator;

    // Build a synthetic image: a small "code" region plus a zeroed reserved
    // signature region at a 0x4040 boundary (4 full pages + a 64-byte 5th page),
    // mirroring the real MachO layout's tiny-code case.
    const sig_off: u32 = 0x4040;
    const ident = "ref";
    const text_size: u32 = 20;
    const sig_len = signatureLen(ident, sig_off);
    const image = try gpa.alloc(u8, sig_off + sig_len);
    defer gpa.free(image);
    // Fill the code region with a recognizable pattern so the hashes are
    // non-trivial; zero the reserved region (as MachO.assemble does).
    for (image[0..sig_off], 0..) |*b, i| b.* = @intCast(i & 0xFF);
    @memset(image[sig_off..], 0);

    try sign(gpa, image, ident, sig_off, text_size);

    const rd32 = struct {
        fn f(img: []const u8, off: usize) u32 {
            return std.mem.readInt(u32, img[off..][0..4], .big);
        }
    }.f;
    const rd64 = struct {
        fn f(img: []const u8, off: usize) u64 {
            return std.mem.readInt(u64, img[off..][0..8], .big);
        }
    }.f;

    // --- SuperBlob ---
    try testing.expectEqual(CSMAGIC_EMBEDDED_SIGNATURE, rd32(image, sig_off));
    try testing.expectEqual(sig_len, rd32(image, sig_off + 4)); // length
    try testing.expectEqual(@as(u32, 1), rd32(image, sig_off + 8)); // count
    try testing.expectEqual(CSSLOT_CODEDIRECTORY, rd32(image, sig_off + 12)); // type
    try testing.expectEqual(@as(u32, 20), rd32(image, sig_off + 16)); // index offset

    // --- CodeDirectory ---
    const cd = sig_off + 20;
    try testing.expectEqual(CSMAGIC_CODEDIRECTORY, rd32(image, cd));
    try testing.expectEqual(sig_len - 20, rd32(image, cd + 4)); // cd length
    try testing.expectEqual(CS_SUPPORTSEXECSEG, rd32(image, cd + 8)); // version 0x20400
    try testing.expectEqual(CD_FLAGS, rd32(image, cd + 12)); // 0x20002
    const hash_offset = 88 + @as(u32, ident.len) + 1;
    try testing.expectEqual(hash_offset, rd32(image, cd + 16)); // hashOffset
    try testing.expectEqual(@as(u32, 88), rd32(image, cd + 20)); // identOffset
    try testing.expectEqual(@as(u32, 0), rd32(image, cd + 24)); // nSpecialSlots
    try testing.expectEqual(@as(u32, 5), rd32(image, cd + 28)); // nCodeSlots
    try testing.expectEqual(sig_off, rd32(image, cd + 32)); // codeLimit
    try testing.expectEqual(@as(u8, 32), image[cd + 36]); // hashSize
    try testing.expectEqual(@as(u8, 2), image[cd + 37]); // hashType (SHA-256)
    try testing.expectEqual(@as(u8, 0), image[cd + 38]); // platform
    try testing.expectEqual(@as(u8, 12), image[cd + 39]); // pageSize log2
    try testing.expectEqual(@as(u64, 0), rd64(image, cd + 64)); // execSegBase
    try testing.expectEqual(@as(u64, text_size), rd64(image, cd + 72)); // execSegLimit
    try testing.expectEqual(CS_EXECSEG_MAIN_BINARY, rd64(image, cd + 80)); // execSegFlags

    // --- Identifier ---
    try testing.expectEqualStrings("ref", image[cd + 88 ..][0..3]);
    try testing.expectEqual(@as(u8, 0), image[cd + 88 + 3]); // NUL

    // --- Code-slot hashes: recompute slot 0 (full page) and slot 4 (partial). ---
    const hashes = cd + hash_offset;
    var d0: [32]u8 = undefined;
    Sha256.hash(image[0..PAGE_SIZE], &d0, .{});
    try testing.expectEqualSlices(u8, &d0, image[hashes..][0..32]);

    // Slot 4 covers [0x4000, 0x4040) — a 64-byte partial page.
    var d4: [32]u8 = undefined;
    Sha256.hash(image[4 * PAGE_SIZE .. sig_off], &d4, .{});
    try testing.expectEqualSlices(u8, &d4, image[hashes + 4 * 32 ..][0..32]);
}
