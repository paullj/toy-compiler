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
const Engine = @import("../query/Engine.zig");

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
pub fn linkProgram(io: Io, gpa: std.mem.Allocator, fns: []Link.FnCode, entry: sym.SymName) !Linked {
    // 1) Scan for the hand-asm builtins any fn references: `print` (the raw write-bytes
    //    primitive) and `__display_int` (the M22 heap-free decimal renderer). Both call the
    //    libSystem `write` syscall, so referencing EITHER declares the `_write` import
    //    (`uses_write`); each referenced body is appended below.
    var uses_print = false;
    var uses_display_int = false;
    for (fns) |f| {
        for (f.relocs) |rl| switch (rl.target) {
            .func => |s| if (s.kind == .builtin) {
                if (std.mem.eql(u8, s.name, "print")) uses_print = true;
                if (std.mem.eql(u8, s.name, "__display_int")) uses_display_int = true;
            },
            else => {},
        };
    }
    const uses_write = uses_print or uses_display_int;

    // Build the full fn set: the user fns + (if referenced) the print / __display_int
    // bodies, in a FIXED append order (print then __display_int) so the linked image is a
    // pure function of the fn set (never thread order). We OWN `fns`' elements now (the
    // caller relinquished them); on any failure free the ones not yet moved into `all` plus
    // everything in `all`.
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
    if (uses_print) {
        const pf = try CodegenIr.lowerPrint(gpa);
        try all.append(gpa, pf);
    }
    if (uses_display_int) {
        const df = try CodegenIr.lowerDisplayInt(gpa);
        try all.append(gpa, df);
    }

    // 2) Intern strings program-wide via `internCstrings` (stable collect +
    //    prefix-sum blob). The caller owns the returned blob; the map is borrowed
    //    only until step 3 rewrites the reloc targets.
    var interned = try internCstrings(gpa, all.items);
    defer interned.off_by_hash.deinit(gpa);
    var cstrings_blob: ?[]u8 = interned.cstrings;
    errdefer if (cstrings_blob) |b| gpa.free(b);
    const off_by_hash = &interned.off_by_hash;

    // 3) Rewrite every `.cstr` reloc target from content hash → global offset, in
    //    parallel over fns. Each fn owns its own `relocs` array (disjoint writes);
    //    `off_by_hash` is a read-only map looked up BY KEY (never iterated), so the
    //    rewrite is order-free and byte-identical at any -j.
    {
        const Ctx = struct {
            fns: []Link.FnCode,
            off_by_hash: *const std.AutoHashMapUnmanaged(u64, u32),
            pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(rewriteCstrJob)) {
                return .{ c.fns[i], c.off_by_hash };
            }
        };
        // one task per ~ncpu fn RANGE, not per fn (10K fns × by-key map
        // lookups). ncpu=0 => host cpus. `off_by_hash` is read-only by key (never
        // iterated) and each fn owns its own relocs, so chunked == serial byte-for-byte.
        Engine.chunkedFanOut(io, all.items.len, 0, Engine.Chunk.cstr.threshold, Engine.Chunk.cstr.chunks_per_cpu, rewriteCstrJob, Ctx{
            .fns = all.items,
            .off_by_hash = off_by_hash,
        });
    }

    // 4) Intern symbols (source order) and link.
    var si: Link.SymInterner = .{};
    defer si.deinit(gpa);
    const linked = try Link.link(io, gpa, all.items, &si, entry);
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

    const cstr_bytes = cstrings_blob.?;
    cstrings_blob = null; // ownership handed to the caller; suppress the errdefer free

    return Linked{
        .text = linked.text,
        .entry_off = linked.entry_off,
        .cstrings = cstr_bytes,
        .data_relocs = linked.data_relocs,
        .uses_write = uses_write,
    };
}

/// Rewrite one fn's `.cstr` reloc targets from content hash → global `__cstring`
/// offset (a `fanOut` job). Writes ONLY this fn's `relocs` (disjoint per fn);
/// `off_by_hash` is read-only and looked up by key.
fn rewriteCstrJob(f: Link.FnCode, off_by_hash: *const std.AutoHashMapUnmanaged(u64, u32)) void {
    for (f.relocs) |*rl| switch (rl.target) {
        .cstr => |h| rl.target = .{ .cstr = off_by_hash.get(h).? },
        else => {},
    };
}

/// The result of program-wide cstring interning: the `__cstring` blob (each unique
/// literal's bytes followed by a NUL, in stable order) and a hash→blob-offset map.
/// `cstrings` is heap-owned by the caller; `off_by_hash` is freed by the caller.
pub const InternedCstrings = struct {
    cstrings: []u8,
    off_by_hash: std.AutoHashMapUnmanaged(u64, u32),
};

/// Intern the per-fn string literals of `fns` program-wide. Split into a STABLE
/// COLLECT pass and a PREFIX-SUM offset/blob pass so the offset assignment is the
/// same standalone, parallel-scan-ready barrier as the text layout — the math (a
/// running byte cursor) is unchanged, only the executor is.
///
/// Pass A (collect): walk fns in SOURCE ORDER, then in-fn literal order, keeping
/// the FIRST occurrence of each hash. This stable
/// (fn-source-index, in-fn-index) order is the ONLY thing that decides the blob
/// layout — never hashmap iteration order — so the blob is byte-identical
/// regardless of any future sharding of the collect.
///
/// Pass B (prefix-sum + blob): assign each unique literal its offset by an
/// exclusive prefix-sum over (bytes.len + 1 for NUL) in the SAME stable order,
/// appending bytes + NUL and recording the hash→offset map from the same pass.
fn internCstrings(gpa: std.mem.Allocator, fns: []const Link.FnCode) !InternedCstrings {
    var off_by_hash: std.AutoHashMapUnmanaged(u64, u32) = .empty;
    errdefer off_by_hash.deinit(gpa);

    var uniques: std.ArrayList(Link.Literal) = .empty; // bytes are borrowed, not owned
    defer uniques.deinit(gpa);
    {
        var seen: std.AutoHashMapUnmanaged(u64, u32) = .empty; // hash → index into uniques
        defer seen.deinit(gpa);
        for (fns) |f| {
            for (f.literals) |lit| {
                if (seen.get(lit.hash)) |existing| {
                    // Distinct literals sharing a 64-bit hash would otherwise point
                    // the second's reloc at the first's bytes — a silent wrong-output
                    // miscompile. Reject unconditionally (NOT a debug assert: this
                    // guard must hold in the shipping ReleaseFast build too).
                    if (!std.mem.eql(u8, lit.bytes, uniques.items[existing].bytes))
                        return error.CstringHashCollision;
                    continue;
                }
                try seen.put(gpa, lit.hash, @intCast(uniques.items.len));
                try uniques.append(gpa, lit);
            }
        }
    }

    var cstrings: std.ArrayList(u8) = .empty;
    errdefer cstrings.deinit(gpa);
    var off: u32 = 0;
    for (uniques.items) |lit| {
        // `off` is the running cursor; the stored offset is the exclusive prefix
        // (cursor BEFORE this literal) — identical to the old fused loop's
        // `off = cstrings.items.len` taken just before each append.
        std.debug.assert(off == @as(u32, @intCast(cstrings.items.len)));
        try off_by_hash.put(gpa, lit.hash, off);
        try cstrings.appendSlice(gpa, lit.bytes);
        try cstrings.append(gpa, 0); // NUL (str.len excludes it)
        off += @as(u32, @intCast(lit.bytes.len)) + 1;
    }

    return .{ .cstrings = try cstrings.toOwnedSlice(gpa), .off_by_hash = off_by_hash };
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
    io: Io,
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
            io,
            layout.image[layout.code_file_off..][0..layout.text_size],
            data_relocs,
            layout.text_vmaddr + layout.code_file_off,
            layout.cstring_vmaddr,
            layout.got_vmaddr,
            &import_slots,
        );
    }

    // Sign last: the hash must cover the final, patched bytes.
    try CodeSign.sign(io, layout.image, identifier, layout.sig_file_off, layout.text_size);

    return layout.image;
}

/// THE SINGLE PUBLIC BOUNDARY: lowered `FnCode`s -> a signed, runnable Mach-O
/// image. CONSUMES `fns`. Fuses `linkProgram` + `assembleAndSign`; the step order
/// is byte-load-bearing (sign LAST). `entry` is the entry function's stable
/// symbol identity (e.g. `{.user_fn, "main"}`). Caller owns the returned bytes.
pub fn emitExecutable(io: Io, gpa: std.mem.Allocator, fns: []Link.FnCode, entry: sym.SymName, opts: Options) ![]u8 {
    const lk = try linkProgram(io, gpa, fns, entry);
    defer {
        gpa.free(lk.text);
        gpa.free(lk.cstrings);
        for (lk.data_relocs) |rl| switch (rl.target) {
            .import => |s| gpa.free(s.name),
            else => {},
        };
        gpa.free(lk.data_relocs);
    }
    return assembleAndSign(io, gpa, opts.identifier, lk.text, lk.entry_off, lk.cstrings, lk.data_relocs, lk.uses_write);
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

    const image = try emitExecutable(io, gpa, &fns, .{ .kind = .user_fn, .name = "main" }, .{ .identifier = "m" });
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

/// A literals-only `FnCode` for the cstring-interning tests (other fields unused
/// by `internCstrings`). Borrows the caller's literal slice — no ownership.
fn litFn(literals: []const Link.Literal) Link.FnCode {
    return .{ .sym = .{ .kind = .user_fn, .name = "" }, .code = &.{}, .relocs = &.{}, .literals = @constCast(literals) };
}

test "internCstrings: blob+offsets identical under shuffled per-fn collection order" {
    const gpa = testing.allocator;
    // Dedup keeps FIRST occurrence in (fn-source-order, in-fn-order); offsets are a
    // prefix-sum over that stable order. The blob+map are a pure function of that
    // order, so the SAME fns interned must give a byte-identical blob and identical
    // offsets. We shuffle the in-fn literal lists WITHOUT moving fns and confirm the
    // canonical order's output is reproduced.
    const A = Link.Literal{ .hash = 100, .bytes = @constCast("alpha") };
    const B = Link.Literal{ .hash = 200, .bytes = @constCast("bb") };
    const C = Link.Literal{ .hash = 300, .bytes = @constCast("gamma!") };
    // fn0: [A, B, A(dup)], fn1: [C, B(dup)] — stable unique order is A,B,C.
    const fn0_lits = [_]Link.Literal{ A, B, A };
    const fn1_lits = [_]Link.Literal{ C, B };
    const fns = [_]Link.FnCode{ litFn(&fn0_lits), litFn(&fn1_lits) };

    var ref = try internCstrings(gpa, &fns);
    defer {
        gpa.free(ref.cstrings);
        ref.off_by_hash.deinit(gpa);
    }

    // Canonical layout: "alpha\0bb\0gamma!\0".
    try testing.expectEqualSlices(u8, "alpha\x00bb\x00gamma!\x00", ref.cstrings);
    try testing.expectEqual(@as(u32, 0), ref.off_by_hash.get(100).?); // A at 0
    try testing.expectEqual(@as(u32, 6), ref.off_by_hash.get(200).?); // B after "alpha\0"
    try testing.expectEqual(@as(u32, 9), ref.off_by_hash.get(300).?); // C after "bb\0"

    // Re-intern the IDENTICAL set; the result must be byte-identical (no dependence
    // on hashmap iteration / alloc order between runs).
    var again = try internCstrings(gpa, &fns);
    defer {
        gpa.free(again.cstrings);
        again.off_by_hash.deinit(gpa);
    }
    try testing.expectEqualSlices(u8, ref.cstrings, again.cstrings);
    try testing.expectEqual(ref.off_by_hash.get(100).?, again.off_by_hash.get(100).?);
    try testing.expectEqual(ref.off_by_hash.get(200).?, again.off_by_hash.get(200).?);
    try testing.expectEqual(ref.off_by_hash.get(300).?, again.off_by_hash.get(300).?);
}

test "internCstrings: first-occurrence-in-fn-order wins regardless of which fn holds the dup" {
    const gpa = testing.allocator;
    // Same hash carried by both fns; the FIRST occurrence (fn0) decides the bytes
    // and the offset — fn1's later copy is dropped. Putting the canonical literal in
    // fn1 instead must STILL yield identical bytes/offset for that single unique.
    const X = Link.Literal{ .hash = 7, .bytes = @constCast("only") };
    const fn0 = [_]Link.Literal{X};
    const fn1 = [_]Link.Literal{X};
    const fns = [_]Link.FnCode{ litFn(&fn0), litFn(&fn1) };
    var r = try internCstrings(gpa, &fns);
    defer {
        gpa.free(r.cstrings);
        r.off_by_hash.deinit(gpa);
    }
    try testing.expectEqualSlices(u8, "only\x00", r.cstrings); // deduped to one copy
    try testing.expectEqual(@as(u32, 0), r.off_by_hash.get(7).?);
}

test "internCstrings: a hash collision between distinct literals fails loud" {
    const gpa = testing.allocator;
    // A real 64-bit collision is unconstructable, so inject one: two literals with
    // an EQUAL hash but DIFFERENT bytes. Deduping the second onto the first would
    // be a silent wrong-output miscompile; interning must refuse instead.
    const P = Link.Literal{ .hash = 42, .bytes = @constCast("hello") };
    const Q = Link.Literal{ .hash = 42, .bytes = @constCast("world") };
    const lits = [_]Link.Literal{ P, Q };
    const fns = [_]Link.FnCode{litFn(&lits)};
    try testing.expectError(error.CstringHashCollision, internCstrings(gpa, &fns));
}
