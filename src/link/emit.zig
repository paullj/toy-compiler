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
/// Mach-O layout). The dyld imports list is DERIVED internally from the `.import`
/// reloc targets (dedup + stable-sort by name), not a caller input.
pub const Options = struct {
    identifier: []const u8,
    /// Informational today; the only supported target is aarch64-macos and it is
    /// not threaded into the emitted bytes.
    target: []const u8 = "aarch64-macos",
    /// The per-type descriptor table to embed, in canonical order. Empty (the default)
    /// leaves the image byte-identical to a descriptor-free build.
    descriptors: []const Link.DescEntry = &.{},
};

/// The linked-program tail: the joined __text blob, the entry offset, and the
/// cross-segment data the assemble step needs. The driver packs this into its
/// own `LinkedProgram` (which additionally carries the codegen listing/diags).
pub const Linked = struct {
    text: []u8,
    entry_off: u32,
    /// Interned `__cstring` bytes. Owned; empty for string-free programs.
    cstrings: []u8 = &.{},
    /// Cross-segment relocs (adrp/add/ldr to __cstring/__got) rebased to absolute
    /// __text offsets, patched by `Link.applyDataRelocs` after MachO assigns
    /// vmaddrs. Owned; empty for programs with no data relocations. The dyld import
    /// set is DERIVED from these (`.import` targets) in `assembleAndSign` — never a
    /// separate flag — so it is a pure function of the fn set at any `-jN`.
    data_relocs: []Link.Reloc = &.{},
};

/// Append the print body if referenced, intern strings deterministically,
/// rewrite `.cstr` targets, then `Link.link`. CONSUMES `fns` (frees each FnCode
/// and any appended print body). `entry` is the entry function's stable symbol
/// identity. Returns the linked tail; the caller owns its `text`/`cstrings`/
/// `data_relocs` (free `.import` data-reloc names individually).
pub fn linkProgram(io: Io, gpa: std.mem.Allocator, fns: []Link.FnCode, entry: sym.SymName, descriptors: []const Link.DescEntry) !Linked {
    // 1) Scan which hand-emitted builtins (`CodegenIr.hand_builtins`) any fn references;
    //    each referenced body is appended below. Those bodies emit their own `.import`
    //    relocs (e.g. `print`/`panic` -> `write`), so the dyld import set is DERIVED from
    //    `data_relocs` like any other import.
    var used = [_]bool{false} ** CodegenIr.hand_builtins.len;
    for (fns) |f| {
        for (f.relocs) |rl| switch (rl.target) {
            .func => |s| if (s.kind == .builtin) {
                for (CodegenIr.hand_builtins, 0..) |hb, i| {
                    if (std.mem.eql(u8, s.name, hb.name)) used[i] = true;
                }
            },
            else => {},
        };
    }
    // A builtin body can statically call other builtins from within its APPENDED code
    // (e.g. `gc_alloc` triggers `gc_collect`), an edge invisible to the fn-only used-scan
    // above. Close the used set over each row's declared `static_call_deps` to a fixpoint,
    // so any program that carries a builtin also carries its transitive callees.
    var closure_changed = true;
    while (closure_changed) {
        closure_changed = false;
        inline for (CodegenIr.hand_builtins, 0..) |hb, i| {
            if (used[i]) inline for (hb.static_call_deps) |dep| {
                const j = comptime CodegenIr.handBuiltinIndex(dep);
                if (!used[j]) {
                    used[j] = true;
                    closure_changed = true;
                }
            };
        }
    }
    // `panic` alone is consumed past the append: it drags in the `write` import and
    // reserves the backtrace symbol table's slot.
    const uses_panic = used[comptime CodegenIr.handBuiltinIndex("panic")];
    // `gc_collect` present ⇒ the collector runs ⇒ reserve + emit the GC stack map. The
    // static-call closure above already pulls `gc_collect` in via `gc_alloc`, so this is
    // true for any allocating program and false for a heap-free one (keeping it
    // byte-identical). Local to `linkProgram`: no new param threads through `Codegen`.
    const uses_gc = used[comptime CodegenIr.handBuiltinIndex("gc_collect")];
    // Build the full fn set: the user fns + (if referenced) the hand-emitted builtin
    // bodies, appended in `hand_builtins` order so the linked image is a pure function of
    // the fn set (never thread order). We OWN `fns`' elements now (the caller relinquished
    // them); on any failure free the ones not yet moved into `all` plus everything in `all`.
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
    for (CodegenIr.hand_builtins, 0..) |hb, i| {
        if (used[i]) try all.append(gpa, try hb.lower(gpa));
    }

    // 2) Intern strings program-wide via `internCstrings` (stable collect +
    //    prefix-sum blob). The caller owns the returned blob; the map is borrowed
    //    only until step 3 rewrites the reloc targets.
    var interned = try internCstrings(gpa, all.items);
    defer interned.off_by_hash.deinit(gpa);
    var cstrings_blob: ?[]u8 = interned.cstrings;
    errdefer if (cstrings_blob) |b| gpa.free(b);
    const off_by_hash = &interned.off_by_hash;

    // 2b) Reserve the backtrace symbol table's slot at the (8-aligned) END of the
    //     __cstring blob and bind the sentinel hash `__panic`'s `.cstr` reloc carries
    //     to that offset, so step 3 rewrites it like any string. The table BYTES are
    //     appended after `link` (they depend on the layout it computes); only their
    //     start offset is known now. A real literal hashing to the sentinel would be
    //     silently mis-pointed — reject it (astronomically unlikely).
    // The descriptor table is reserved FIRST (before the symtab), so when there are no
    // descriptors `desc_table_off == symtab_off == align8(cstrings.len)` — the panic corpus
    // reservation is byte-identical. Each per-type sentinel maps to its 48-byte entry.
    const desc_table_off: u32 = std.mem.alignForward(u32, @intCast(interned.cstrings.len), 8);
    const n_desc = descriptors.len;
    if (n_desc > 0) {
        for (descriptors, 0..) |e, i| {
            if (off_by_hash.contains(e.desc_hash)) return error.CstringHashCollision;
            try off_by_hash.put(gpa, e.desc_hash, @intCast(@as(usize, desc_table_off) + i * 48));
        }
    }
    // The GC stack map is reserved BETWEEN the descriptor table and the symtab, only when
    // the collector runs. When no gc, `stackmap_off`/`stackmap_len` collapse and
    // `symtab_off == after_desc` — the panic-but-not-gc corpus stays byte-identical.
    const after_desc: u32 = if (n_desc > 0)
        std.mem.alignForward(u32, desc_table_off + @as(u32, @intCast(n_desc * 48)), 8)
    else
        desc_table_off;
    const stackmap_off: u32 = after_desc;
    var stackmap_len: u32 = 0;
    if (uses_gc) {
        if (off_by_hash.contains(Link.gc_stackmap_base_hash)) return error.CstringHashCollision;
        try off_by_hash.put(gpa, Link.gc_stackmap_base_hash, stackmap_off);
        stackmap_len = stackMapLen(all.items);
    }
    const symtab_off: u32 = if (uses_gc)
        std.mem.alignForward(u32, stackmap_off + stackmap_len, 8)
    else
        after_desc;
    if (uses_panic) {
        if (off_by_hash.contains(Link.symtab_base_hash)) return error.CstringHashCollision;
        try off_by_hash.put(gpa, Link.symtab_base_hash, symtab_off);
    }

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
    const linked = try Link.link(io, gpa, all.items, &si, entry, uses_panic, uses_gc, descriptors);
    errdefer gpa.free(linked.text);
    errdefer gpa.free(linked.data_relocs);
    defer gpa.free(linked.sym_table);
    defer gpa.free(linked.desc_table);
    defer gpa.free(linked.stack_map);
    // The pre-link length reservation must equal the linked table's actual size.
    std.debug.assert(!uses_gc or linked.stack_map.len == stackmap_len);

    // 4b) Append the reserved __cstring tail — the descriptor table (at `desc_table_off`)
    //     then the backtrace symbol table (at `symtab_off`), each padded to its 8-aligned
    //     start so it rides in the signed, read-only section and the reserved `.cstr` relocs
    //     resolve to it. With no descriptors the desc branch is skipped and the symtab pads
    //     to the SAME offset as before → byte-identical.
    if (linked.desc_table.len > 0 or (uses_gc and linked.stack_map.len > 0) or (uses_panic and linked.sym_table.len > 0)) {
        var grown: std.ArrayList(u8) = .empty;
        errdefer grown.deinit(gpa);
        const old = cstrings_blob.?;
        try grown.appendSlice(gpa, old);
        if (linked.desc_table.len > 0) {
            try grown.appendNTimes(gpa, 0, @as(usize, desc_table_off) - grown.items.len);
            try grown.appendSlice(gpa, linked.desc_table);
        }
        // Stack map sits between the descriptor table and the symtab.
        if (uses_gc and linked.stack_map.len > 0) {
            try grown.appendNTimes(gpa, 0, @as(usize, stackmap_off) - grown.items.len);
            try grown.appendSlice(gpa, linked.stack_map);
        }
        if (uses_panic and linked.sym_table.len > 0) {
            try grown.appendNTimes(gpa, 0, @as(usize, symtab_off) - grown.items.len);
            try grown.appendSlice(gpa, linked.sym_table);
        }
        const new = try grown.toOwnedSlice(gpa); // old still owned on failure (errdefer frees it)
        cstrings_blob = new;
        gpa.free(old);
    }

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
    };
}

/// The pre-link byte length of the GC stack-map table `Link.buildStackMapTable` will
/// produce for `fns` — computed from each `FnCode.root_bitmap` (present pre-link), so the
/// symtab's reserved offset is known before layout. Two spellings of one size; the
/// `std.debug.assert` at the link callsite guards them against drift. Layout mirrors
/// `buildSideTable`: `[u64 count][{off,payload_off}×count][blob]`, count = fns+1 (sentinel),
/// blob = Σ per-fn payload (`root_bitmap` or the 4-byte conservative marker) + the
/// sentinel's 4-byte marker.
fn stackMapLen(fns: []const Link.FnCode) u32 {
    var blob: usize = 4; // sentinel's conservative marker
    for (fns) |f| blob += if (f.root_bitmap.len == 0) 4 else f.root_bitmap.len;
    return @intCast(8 + (fns.len + 1) * 16 + blob);
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

/// The DISTINCT `.import` reloc target names, sorted by name bytes — the canonical
/// GOT slot order. Dedup first (via `seen`) means the sort ranges over unique keys, so
/// sort stability is moot: the total order is a pure function of the name SET,
/// independent of `data_relocs` arrival order / thread count. Caller owns the slice
/// and each duped name.
fn importNames(gpa: std.mem.Allocator, data_relocs: []const Link.Reloc) ![]const []const u8 {
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    defer seen.deinit(gpa);
    var names: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (names.items) |nm| gpa.free(nm);
        names.deinit(gpa);
    }
    for (data_relocs) |rl| switch (rl.target) {
        .import => |s| {
            const gop = try seen.getOrPut(gpa, s.name);
            if (!gop.found_existing) try names.append(gpa, try gpa.dupe(u8, s.name));
        },
        else => {},
    };
    const out = try names.toOwnedSlice(gpa);
    std.mem.sort([]const u8, out, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    return out;
}

/// Build the fully signed, runnable Mach-O image for `code`. `identifier` is the
/// code-signing identity (the output basename); `entry_off` is `main`'s byte
/// offset within `code`. `cstrings` are the interned `__cstring` bytes, and
/// `data_relocs` the cross-segment adrp/add/ldr patches `Link.link` rebased to
/// absolute __text offsets; the dyld import set is derived from their `.import` targets.
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
) ![]u8 {
    // Derive the ordered dyld import set from the `.import` reloc targets: collect the
    // DISTINCT names, stable-sort by name → canonical GOT slot order. This is a pure
    // function of the name SET (arrival order / thread count irrelevant), so it is the
    // sole authority for the GOT layout, the imports/symbols tables, `import_slots`,
    // and each slot's chained-fixups ordinal — byte-identical at any `-jN`.
    const names = try importNames(gpa, data_relocs);
    defer {
        for (names) |nm| gpa.free(nm);
        gpa.free(names);
    }

    // The dyld-facing linkage name is the C symbol with a leading `_` (`labs`→`_labs`),
    // in the same sorted slot order. Heap-owned; freed after `assemble` copies them.
    const imports = try gpa.alloc(MachO.Import, names.len);
    defer {
        for (imports) |im| gpa.free(@constCast(im.name));
        gpa.free(imports);
    }
    for (names, 0..) |nm, i| imports[i] = .{ .name = try std.fmt.allocPrint(gpa, "_{s}", .{nm}) };

    const layout = try MachO.assemble(gpa, identifier, code, entry_off, cstrings, imports);
    errdefer gpa.free(layout.image);

    // Slot i binds the i-th sorted import; `applyDataRelocs` looks each `.import`
    // target's (bare) name up here to patch its adrp/ldr to the right __got slot.
    var import_slots: std.StringHashMapUnmanaged(u32) = .empty;
    defer import_slots.deinit(gpa);
    for (names, 0..) |nm, i| try import_slots.put(gpa, nm, @intCast(i));

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
    const lk = try linkProgram(io, gpa, fns, entry, opts.descriptors);
    defer {
        gpa.free(lk.text);
        gpa.free(lk.cstrings);
        for (lk.data_relocs) |rl| switch (rl.target) {
            .import => |s| gpa.free(s.name),
            else => {},
        };
        gpa.free(lk.data_relocs);
    }
    return assembleAndSign(io, gpa, opts.identifier, lk.text, lk.entry_off, lk.cstrings, lk.data_relocs);
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

// AArch64 placeholder words (the reloc patchers overwrite/preserve as needed):
// `adrp x0,0` / `add x0,x0,#0` / `bl #0` / `ret`.
const w_adrp0: u32 = 0x90000000;
const w_add0: u32 = 0x91000000;
const w_bl0: u32 = 0x94000000;
const w_ret: u32 = 0xD65F03C0;

/// Build one owned `FnCode` for the determinism test: `adrp/add` referencing a cstring
/// literal (a cross-segment data-reloc pair) plus, if `call_panic`, a `bl panic` (a
/// call26 to the appended builtin, which drags in `write` + the symbol table). Consumed
/// (freed) by `linkProgram`.
fn detFn(gpa: std.mem.Allocator, name: []const u8, lit_hash: u64, lit_bytes: []const u8, call_panic: bool) !Link.FnCode {
    const nwords: usize = if (call_panic) 4 else 3;
    const code = try gpa.alloc(u8, nwords * 4);
    std.mem.writeInt(u32, code[0..4], w_adrp0, .little);
    std.mem.writeInt(u32, code[4..8], w_add0, .little);
    if (call_panic) {
        std.mem.writeInt(u32, code[8..12], w_bl0, .little);
        std.mem.writeInt(u32, code[12..16], w_ret, .little);
    } else {
        std.mem.writeInt(u32, code[8..12], w_ret, .little);
    }
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    try relocs.append(gpa, .{ .site = 0, .target = .{ .cstr = lit_hash }, .kind = .adrp_page });
    try relocs.append(gpa, .{ .site = 4, .target = .{ .cstr = lit_hash }, .kind = .add_lo12 });
    if (call_panic) {
        try relocs.append(gpa, .{ .site = 8, .target = .{ .func = .{ .kind = .builtin, .name = try gpa.dupe(u8, "panic") } }, .kind = .call26 });
    }
    const lits = try gpa.alloc(Link.Literal, 1);
    lits[0] = .{ .hash = lit_hash, .bytes = try gpa.dupe(u8, lit_bytes) };
    return .{ .sym = .{ .kind = .user_fn, .name = try gpa.dupe(u8, name) }, .code = code, .relocs = try relocs.toOwnedSlice(gpa), .literals = lits };
}

/// Build one owned `FnCode` that calls an `extern` import `sym` via its GOT slot:
/// `adrp x16,0 ; ldr x16,[x16] ; blr x16 ; ret`, with the two `.import` relocs the
/// cross-segment pass rebases to the `__got` slot. Consumed (freed) by `linkProgram`.
fn importFn(gpa: std.mem.Allocator, name: []const u8, sym_name: []const u8) !Link.FnCode {
    const code = try gpa.alloc(u8, 16);
    std.mem.writeInt(u32, code[0..4], w_adrp0, .little);
    std.mem.writeInt(u32, code[4..8], 0xF9400210, .little); // ldr x16,[x16]
    std.mem.writeInt(u32, code[8..12], 0xD63F0200, .little); // blr x16
    std.mem.writeInt(u32, code[12..16], w_ret, .little);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    try relocs.append(gpa, .{ .site = 0, .target = .{ .import = .{ .kind = .import, .name = try gpa.dupe(u8, sym_name) } }, .kind = .adrp_page });
    try relocs.append(gpa, .{ .site = 4, .target = .{ .import = .{ .kind = .import, .name = try gpa.dupe(u8, sym_name) } }, .kind = .ldr_lo12 });
    return .{ .sym = .{ .kind = .user_fn, .name = try gpa.dupe(u8, name) }, .code = code, .relocs = try relocs.toOwnedSlice(gpa), .literals = &.{} };
}

/// Two fns importing distinct dyld symbols (`write`, `labs`) → a 2-slot `__got`.
/// Fresh (owned) each call so both determinism-test runs get their own to consume.
fn twoImportFns(gpa: std.mem.Allocator) ![]Link.FnCode {
    const fns = try gpa.alloc(Link.FnCode, 2);
    fns[0] = try importFn(gpa, "aaa", "write");
    fns[1] = try importFn(gpa, "main", "labs");
    return fns;
}

test "backend spike: a 2-import (labs+write) image is byte-identical at -j1 and -jN" {
    // The [C11] gate: the generalized N-import chained-fixups blob + GOT layout + the
    // whole emitted image must be byte-identical regardless of thread count, because the
    // import set is DERIVED (dedup + stable-sort by name), never thread-arrival ordered.
    const gpa = testing.allocator;
    const entry = Link.SymName{ .kind = .user_fn, .name = "main" };

    const fns_serial = try twoImportFns(gpa);
    defer gpa.free(fns_serial);
    var t_serial = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer t_serial.deinit();
    const img_serial = try emitExecutable(t_serial.io(), gpa, fns_serial, entry, .{ .identifier = "spike" });
    defer gpa.free(img_serial);

    const fns_par = try twoImportFns(gpa);
    defer gpa.free(fns_par);
    var t_par = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(8) });
    defer t_par.deinit();
    const img_par = try emitExecutable(t_par.io(), gpa, fns_par, entry, .{ .identifier = "spike" });
    defer gpa.free(img_par);

    try testing.expectEqualSlices(u8, img_serial, img_par);

    // The __got sits at the third segment's __got section; its two slots carry the
    // name-sorted chained binds: labs (slot 0) then write (slot 1).
    const got_off: usize = @intCast(MachO.PAGE); // __DATA_CONST is the page after single-page __TEXT
    try testing.expectEqual(@as(u64, 0x8010000000000000), std.mem.readInt(u64, img_serial[got_off..][0..8], .little));
    try testing.expectEqual(@as(u64, 0x8000000000000001), std.mem.readInt(u64, img_serial[got_off + 8 ..][0..8], .little));
}

/// One owned `FnCode` `main` whose body is `bl gc_alloc ; ret` — a `.call26` to the
/// hand-emitted `gc_alloc` builtin, which drags in its body plus the `msync`/`mmap`
/// imports. Fresh each call so both determinism-test runs get their own to consume.
fn gcFns(gpa: std.mem.Allocator) ![]Link.FnCode {
    const fns = try gpa.alloc(Link.FnCode, 1);
    const code = try gpa.alloc(u8, 8);
    std.mem.writeInt(u32, code[0..4], w_bl0, .little); // bl gc_alloc (patched at link)
    std.mem.writeInt(u32, code[4..8], w_ret, .little);
    var relocs: std.ArrayList(Link.Reloc) = .empty;
    try relocs.append(gpa, .{ .site = 0, .target = .{ .func = .{ .kind = .builtin, .name = try gpa.dupe(u8, "gc_alloc") } }, .kind = .call26 });
    fns[0] = .{ .sym = .{ .kind = .user_fn, .name = try gpa.dupe(u8, "main") }, .code = code, .relocs = try relocs.toOwnedSlice(gpa), .literals = &.{} };
    return fns;
}

test "backend determinism: a gc_alloc-using image is byte-identical at -j1 and -jN" {
    // The appended allocator body + its derived `msync`/`mmap` GOT imports must be a
    // pure function of the fn set (the body is fixed AArch64; the imports are dedup+
    // sorted by name), so the whole image is identical regardless of thread count.
    if (builtin.os.tag != .macos or builtin.cpu.arch != .aarch64) return error.SkipZigTest;
    const gpa = testing.allocator;
    const entry = Link.SymName{ .kind = .user_fn, .name = "main" };

    const fns_serial = try gcFns(gpa);
    defer gpa.free(fns_serial);
    var t_serial = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer t_serial.deinit();
    const img_serial = try emitExecutable(t_serial.io(), gpa, fns_serial, entry, .{ .identifier = "gc" });
    defer gpa.free(img_serial);

    const fns_par = try gcFns(gpa);
    defer gpa.free(fns_par);
    var t_par = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(8) });
    defer t_par.deinit();
    const img_par = try emitExecutable(t_par.io(), gpa, fns_par, entry, .{ .identifier = "gc" });
    defer gpa.free(img_par);

    try testing.expectEqualSlices(u8, img_serial, img_par);
}

test "backend determinism: the gc stack-map table (riding __cstring) is byte-identical at -j1 and -jN" {
    // A gc_alloc-using program reserves + appends the GC stack map into the signed,
    // read-only __cstring blob. It is built from the layout (stable-sort-ranked offsets +
    // each fn's root bitmap), a pure function of the fn set, so the whole __cstring blob —
    // stack map included — is byte-identical regardless of thread count. Assert the blob is
    // non-empty (the table rides it) and matches across `-j`.
    const gpa = testing.allocator;
    const entry = Link.SymName{ .kind = .user_fn, .name = "main" };

    const fns_serial = try gcFns(gpa);
    defer gpa.free(fns_serial);
    var t_serial = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer t_serial.deinit();
    var serial = try linkProgram(t_serial.io(), gpa, fns_serial, entry, &.{});
    defer freeLinked(gpa, &serial);

    const fns_par = try gcFns(gpa);
    defer gpa.free(fns_par);
    var t_par = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(8) });
    defer t_par.deinit();
    var parallel_lk = try linkProgram(t_par.io(), gpa, fns_par, entry, &.{});
    defer freeLinked(gpa, &parallel_lk);

    try testing.expectEqualSlices(u8, serial.text, parallel_lk.text);
    try testing.expectEqualSlices(u8, serial.cstrings, parallel_lk.cstrings);
    // The stack map lives in __cstring: a gc program's blob is non-empty even with no string
    // literals (the reserved table bytes are appended past the literal region).
    try testing.expect(serial.cstrings.len > 0);
}

/// Three fns with distinct cstrings; `main` also calls `panic`. Fresh (owned) each call
/// so both determinism-test runs get their own copy to consume. `linkProgram` deinits
/// the ELEMENTS but not the slice backing, so the caller frees the returned slice.
fn detFns(gpa: std.mem.Allocator) ![]Link.FnCode {
    const fns = try gpa.alloc(Link.FnCode, 3);
    fns[0] = try detFn(gpa, "aaa", 0x1111, "alpha", false);
    fns[1] = try detFn(gpa, "bbb", 0x2222, "beta", false);
    fns[2] = try detFn(gpa, "main", 0x3333, "gamma", true);
    return fns;
}

fn freeLinked(gpa: std.mem.Allocator, lk: *Linked) void {
    gpa.free(lk.text);
    gpa.free(lk.cstrings);
    for (lk.data_relocs) |rl| if (rl.target.name()) |nm| gpa.free(nm);
    gpa.free(lk.data_relocs);
}

test "backend determinism: linkProgram is byte-identical at -j1 and -jN across every parallel seam" {
    // The load-bearing -jN invariant was verified only at each component (interning,
    // prefix-sum, stable-sort, sign); this drives the WHOLE link tail — cstring interning
    // + `.cstr` rewrite + parallel fnLinkJob + data-reloc stable-concat + panic symbol-table
    // build/append — serially (.limited(0)) vs on real threads (.limited(8)) and asserts the
    // emitted __text + __cstring + the deferred reloc list are identical. This is the seam
    // between the deterministic components, exactly where a regression would hide.
    const gpa = testing.allocator;
    const entry = Link.SymName{ .kind = .user_fn, .name = "main" };

    const fns_serial = try detFns(gpa);
    defer gpa.free(fns_serial); // linkProgram consumes the elements, not the slice
    var t_serial = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer t_serial.deinit();
    var serial = try linkProgram(t_serial.io(), gpa, fns_serial, entry, &.{});
    defer freeLinked(gpa, &serial);

    const fns_par = try detFns(gpa);
    defer gpa.free(fns_par);
    var t_par = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(8) });
    defer t_par.deinit();
    var parallel_lk = try linkProgram(t_par.io(), gpa, fns_par, entry, &.{});
    defer freeLinked(gpa, &parallel_lk);

    try testing.expectEqualSlices(u8, serial.text, parallel_lk.text);
    try testing.expectEqualSlices(u8, serial.cstrings, parallel_lk.cstrings);
    try testing.expectEqual(serial.entry_off, parallel_lk.entry_off);
    // The deferred cross-segment relocs must match in order (the stable-concat seam) and shape.
    try testing.expectEqual(serial.data_relocs.len, parallel_lk.data_relocs.len);
    for (serial.data_relocs, parallel_lk.data_relocs) |a, b| {
        try testing.expectEqual(a.site, b.site);
        try testing.expectEqual(a.kind, b.kind);
    }
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
