//! The single backend boundary: lowered per-function `FnCode`s -> a signed,
//! runnable Mach-O image.
//!
//! WHY this is one entry point: the backend tail is a fixed, order-sensitive
//! pipeline — (1) detect `_write` use, append the `print` body, (2) intern the
//! per-fn content-hashed cstring literals program-wide IN A DETERMINISTIC ORDER
//! (fn source order, then in-fn literal order), (3) rewrite each `.cstr` reloc's
//! hash target to its global `__cstring` offset, (4) `Link.link` (layout +
//! intra-module `.call26` patch + collect cross-segment data_relocs), (5)
//! `MachO.assemble` (segments, __cstring, __got import, chained fixups), (6)
//! `Link.applyDataRelocs` (rebase cross-segment relocs after vmaddrs are known),
//! (7) `CodeSign.sign` LAST, over the final bytes. The ORDER IS BYTE-LOAD-BEARING
//! (esp. sign last). Hiding it behind `emitExecutable` keeps the program-wide
//! interning + `.cstr`-rewrite an image-emission concern (it lives here, not in
//! the driver), and gives the driver a clean handoff: lowered fns in, signed
//! image out.
//!
//! Internally the pipeline is split into `linkProgram` (steps 1-4, producing a
//! linked `Linked` tail) and `assembleAndSign` (steps 5-7). `emitExecutable`
//! fuses them; the driver also calls them individually on the asm/listing path
//! where it needs the linked tail without a final image, and the existing
//! byte-identity tests drive `assembleAndSign` directly.

const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const Io = std.Io;

const Link = @import("Link.zig");
const MachO = @import("MachO.zig");
const CodeSign = @import("CodeSign.zig");
const CodegenIr = @import("../codegen/CodegenIr.zig");
const sym = @import("../symbols/Sym.zig");

/// Emission options. `identifier` is the code-signing identity (the output
/// basename) and is byte-load-bearing (it enters both the signature and the
/// Mach-O layout). The imports list is DERIVED internally from `uses_write`
/// (M2: just `_write`), not a caller input.
pub const Options = struct {
    identifier: []const u8,
    /// Informational today; the only supported target is aarch64-macos and it is
    /// not threaded into the emitted bytes.
    target: []const u8 = "aarch64-macos",
};

/// The linked-program tail: the joined __text blob, the entry offset, and the
/// cross-segment data the assemble step needs. The driver packs this into its
/// own `LinkedProgram` (which additionally carries the codegen listing/diags).
pub const Linked = struct {
    text: []u8,
    entry_off: u32,
    /// Interned `__cstring` bytes (M2). Owned; empty for string-free programs.
    cstrings: []u8 = &.{},
    /// Cross-segment relocs (adrp/add/ldr to __cstring/__got) rebased to absolute
    /// __text offsets, patched by `Link.applyDataRelocs` after MachO assigns
    /// vmaddrs. Owned; empty for M1/M3 programs.
    data_relocs: []Link.Reloc = &.{},
    /// Whether the program calls `print` (→ one `_write` import).
    uses_write: bool = false,
};

/// Append the print body if referenced, intern strings deterministically,
/// rewrite `.cstr` targets, then `Link.link`. CONSUMES `fns` (frees each FnCode
/// and any appended print body). `entry` is the entry function's stable symbol
/// identity. Returns the linked tail; the caller owns its `text`/`cstrings`/
/// `data_relocs` (free `.import` data-reloc names individually).
pub fn linkProgram(gpa: std.mem.Allocator, fns: []Link.FnCode, entry: sym.SymName) !Linked {
    // 1) uses_write: any reloc targeting the print builtin.
    var uses_write = false;
    for (fns) |f| {
        for (f.relocs) |rl| switch (rl.target) {
            .func => |s| if (s.kind == .builtin and std.mem.eql(u8, s.name, "print")) {
                uses_write = true;
            },
            else => {},
        };
    }

    // Build the full fn set: the user fns + (if needed) the print body. We OWN
    // `fns`' elements now (the caller relinquished them); on any failure free the
    // ones not yet moved into `all` plus everything in `all`.
    var all: std.ArrayList(Link.FnCode) = .empty;
    var moved: usize = 0;
    errdefer {
        for (fns[moved..]) |*f| f.deinit(gpa);
    }
    defer {
        for (all.items) |*f| f.deinit(gpa);
        all.deinit(gpa);
    }
    for (fns) |f| {
        try all.append(gpa, f);
        moved += 1;
    }
    if (uses_write) {
        const pf = try CodegenIr.lowerPrint(gpa);
        try all.append(gpa, pf);
    }

    // 2) Intern strings program-wide, ordered by fn source order then in-fn
    //    literal order (stable, thread-schedule-independent). Build the cstring
    //    blob (bytes + NUL) and a hash→offset map. [C8]
    var cstrings: std.ArrayList(u8) = .empty;
    errdefer cstrings.deinit(gpa);
    var off_by_hash: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    defer off_by_hash.deinit(gpa);
    for (all.items) |f| {
        for (f.literals) |lit| {
            if (off_by_hash.get(lit.hash)) |existing| {
                // Hash hit: confirm the bytes actually match before dedup'ing. A
                // 64-bit hash collision between two DISTINCT literals would
                // otherwise silently point the second's reloc at the first's
                // bytes (wrong-output miscompile); fail loudly instead.
                std.debug.assert(std.mem.eql(u8, lit.bytes, cstrings.items[existing..][0..lit.bytes.len]));
                continue;
            }
            const off: u32 = @intCast(cstrings.items.len);
            try cstrings.appendSlice(gpa, lit.bytes);
            try cstrings.append(gpa, 0); // NUL (str.len excludes it)
            try off_by_hash.put(gpa, lit.hash, off);
        }
    }

    // 3) Rewrite every `.cstr` reloc target from content hash → global offset.
    for (all.items) |f| {
        for (f.relocs) |*rl| switch (rl.target) {
            .cstr => |h| rl.target = .{ .cstr = off_by_hash.get(h).? },
            else => {},
        };
    }

    // 4) Intern symbols (source order) and link.
    var si: Link.SymInterner = .{};
    defer si.deinit(gpa);
    const linked = try Link.link(gpa, all.items, &si, entry);
    errdefer gpa.free(linked.text);
    errdefer gpa.free(linked.data_relocs);

    // The data-relocs' `.import` target names point into `all`'s FnCodes, which
    // the deferred `all` deinit is about to free. Dupe them so the caller owns
    // them past this scope.
    {
        var done: usize = 0;
        errdefer for (linked.data_relocs[0..done]) |rl| switch (rl.target) {
            .import => |s| gpa.free(s.name),
            else => {},
        };
        for (linked.data_relocs) |*rl| {
            switch (rl.target) {
                .import => |s| rl.target = .{ .import = .{ .kind = s.kind, .name = try gpa.dupe(u8, s.name) } },
                else => {},
            }
            done += 1;
        }
    }

    const cstr_bytes = try cstrings.toOwnedSlice(gpa);

    return Linked{
        .text = linked.text,
        .entry_off = linked.entry_off,
        .cstrings = cstr_bytes,
        .data_relocs = linked.data_relocs,
        .uses_write = uses_write,
    };
}

/// Build the fully signed, runnable Mach-O image for `code`. `identifier` is the
/// code-signing identity (the output basename); `entry_off` is `main`'s byte
/// offset within `code`. `cstrings` are the interned `__cstring` bytes, and
/// `data_relocs` the cross-segment adrp/add/ldr patches `Link.link` rebased to
/// absolute __text offsets; `uses_write` is whether the program imports `_write`.
///
/// ORDER MATTERS: MachO assigns segment vmaddrs, THEN `Link.applyDataRelocs`
/// patches the cross-segment relocs in __text, and ONLY THEN does `CodeSign.sign`
/// hash the (now final) bytes. Patching after signing would invalidate the hash.
/// Caller owns the returned bytes and writes them mode 0o755.
pub fn assembleAndSign(
    gpa: std.mem.Allocator,
    identifier: []const u8,
    code: []const u8,
    entry_off: u32,
    cstrings: []const u8,
    data_relocs: []const Link.Reloc,
    uses_write: bool,
) ![]u8 {
    // M2 has exactly one import (`_write`) when `print` is used.
    const write_import = [_]MachO.Import{.{ .name = "_write" }};
    const imports: []const MachO.Import = if (uses_write) &write_import else &.{};

    const layout = try MachO.assemble(gpa, identifier, code, entry_off, cstrings, imports);
    errdefer gpa.free(layout.image);

    // The single import `write` lives in GOT slot 0 (M2). `applyDataRelocs` looks
    // up each import target's name in this map.
    var import_slots: std.StringHashMapUnmanaged(u32) = .empty;
    defer import_slots.deinit(gpa);
    if (uses_write) try import_slots.put(gpa, "write", 0);

    // Patch the cross-segment relocs now that vmaddrs are known. The reloc sites
    // are absolute within __text, so slice the image at __text and pass them
    // through directly.
    if (data_relocs.len > 0) {
        Link.applyDataRelocs(
            layout.image[layout.code_file_off..][0..layout.text_size],
            data_relocs,
            layout.text_vmaddr + layout.code_file_off,
            layout.cstring_vmaddr,
            layout.got_vmaddr,
            &import_slots,
        );
    }

    // Sign last: the hash must cover the final, patched bytes.
    try CodeSign.sign(gpa, layout.image, identifier, layout.sig_file_off, layout.text_size);

    return layout.image;
}

/// THE SINGLE PUBLIC BOUNDARY: lowered `FnCode`s -> a signed, runnable Mach-O
/// image. CONSUMES `fns`. Fuses `linkProgram` + `assembleAndSign`; the step order
/// is byte-load-bearing (sign LAST). `entry` is the entry function's stable
/// symbol identity (e.g. `{.user_fn, "main"}`). Caller owns the returned bytes.
pub fn emitExecutable(gpa: std.mem.Allocator, fns: []Link.FnCode, entry: sym.SymName, opts: Options) ![]u8 {
    const lk = try linkProgram(gpa, fns, entry);
    defer {
        gpa.free(lk.text);
        gpa.free(lk.cstrings);
        for (lk.data_relocs) |rl| switch (rl.target) {
            .import => |s| gpa.free(s.name),
            else => {},
        };
        gpa.free(lk.data_relocs);
    }
    return assembleAndSign(gpa, opts.identifier, lk.text, lk.entry_off, lk.cstrings, lk.data_relocs, lk.uses_write);
}

test "link.emitExecutable: hand-built main returns 42 (no front-end)" {
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;

    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // A `main` that returns 42, built WITHOUT the front-end: just the two AArch64
    // words `movz w0, #42` (0x52800540) and `ret` (0xD65F03C0), little-endian.
    const code = try gpa.alloc(u8, 8);
    std.mem.writeInt(u32, code[0..4], 0x52800540, .little); // movz w0, #42
    std.mem.writeInt(u32, code[4..8], 0xD65F03C0, .little); // ret

    var fns = [_]Link.FnCode{.{
        .sym = .{ .kind = .user_fn, .name = try gpa.dupe(u8, "main") },
        .code = code,
        .relocs = try gpa.alloc(Link.Reloc, 0),
        .literals = try gpa.alloc(Link.Literal, 0),
    }};

    const image = try emitExecutable(gpa, &fns, .{ .kind = .user_fn, .name = "main" }, .{ .identifier = "m" });
    defer gpa.free(image);

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_name_buf: [64]u8 = undefined;
    const dir_name = std.fmt.bufPrint(&dir_name_buf, ".zig-cache/tmp/{s}", .{&tmp.sub_path}) catch unreachable;

    const out_path = std.fmt.allocPrint(gpa, "{s}/m", .{dir_name}) catch unreachable;
    defer gpa.free(out_path);
    {
        const perms: Io.File.Permissions = .fromMode(0o755);
        var f = try Io.Dir.cwd().createFile(io, out_path, .{ .permissions = perms });
        defer f.close(io);
        try f.writeStreamingAll(io, image);
        try f.setPermissions(io, perms);
    }

    const abs = try Io.Dir.cwd().realPathFileAlloc(io, out_path, gpa);
    defer gpa.free(abs);
    {
        var cs = try std.process.spawn(io, .{ .argv = &.{ "codesign", "-v", abs } });
        const cs_term = try cs.wait(io);
        try testing.expectEqual(std.process.Child.Term{ .exited = 0 }, cs_term);
    }

    var child = try std.process.spawn(io, .{ .argv = &.{abs} });
    const term = try child.wait(io);
    try testing.expectEqual(std.process.Child.Term{ .exited = 42 }, term);
}
