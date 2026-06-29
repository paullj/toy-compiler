//! Per-function code artifacts and the intra-module linker.
//!
//! WHY: codegen lowers each function INDEPENDENTLY into a `FnCode` artifact
//! ({ sym, code, relocs }) that names its call targets SYMBOLICALLY (by function
//! index, never an address). A later serial `link` pass lays the artifacts out
//! contiguously in __text, assigns each function a text offset, builds the
//! symbol→offset map, and patches every relocation in place. Keeping the reloc /
//! symbol types here (rather than in `MachO`) keeps the Mach-O writer a pure
//! container and makes this file the seam M5 (incremental/parallel codegen +
//! relink) will reuse: re-lower one changed function, then re-`link` the set.
//!
//! For M3 the only relocation is `.call26` — an AArch64 `bl`, PC-relative by a
//! signed word offset. Intra-module `bl` needs NO runtime relocation under PIE/
//! ASLR: the patched offset is the (target − site) word delta, fixed at link
//! time. `link()` is itself in Stage B; Stage A only needs the types below so
//! `Codegen` can emit `FnCode`/`Reloc` values.

const std = @import("std");
const Io = std.Io;
const Aarch64 = @import("../codegen/Aarch64.zig");
const Engine = @import("../query/Engine.zig");

// The stable symbol identity types now live in the `symbols/` peer data module
// (so resolve/types/codegen/link all depend on the DATA, not sideways on this
// backend file). Re-exported here so every `Link.SymName`/`Link.SymKind` callsite
// keeps compiling unchanged.
const symbols = @import("../symbols/Sym.zig");
pub const SymKind = symbols.SymKind;
pub const SymName = symbols.SymName;

/// A symbolic reference to a definition. NEVER an address — the linker (for
/// `.func`/intra-module) or the post-vmaddr pass (for the cross-segment cases)
/// maps it to a final address.
///
///   * `.func`   — a function, named by its stable `SymName` (matches the
///                 callee's identity). Patched by `link` (PC-relative `bl`).
///   * `.cstr`   — a string literal. While lowered (and on disk) this holds the
///                 literal's CONTENT HASH; the serial relink tail rewrites it to
///                 a byte offset into the program-wide `__cstring` blob, after
///                 which `applyDataRelocs` reads it as that offset.
///   * `.import` — an external symbol, named by its stable `SymName` (M2 always
///                 `{.import,"write"}`). Reached through its `__got` slot
///                 (adrp+ldr+blr); patched by `applyDataRelocs`.
pub const SymbolId = union(enum) {
    func: SymName,
    cstr: u64,
    import: SymName,
};

/// The kind of patch a relocation requests.
///
///   * `.call26`    — an AArch64 `bl`: imm26 = (target − site)/4. Intra-module,
///                    patched by `link`.
///   * `.adrp_page` — an `adrp`: page delta = (target>>12) − (site>>12). Used for
///                    BOTH `.cstr` and `.import` targets (the SymbolId picks which
///                    final vmaddr feeds in). Patched by `applyDataRelocs`.
///   * `.add_lo12`  — an `add (imm12)`: the low-12 bits of the cstring vmaddr.
///   * `.ldr_lo12`  — an `ldr (unsigned offset)`: the low-12 bits of the GOT-slot
///                    vmaddr (only for `.import` targets).
pub const RelocKind = enum {
    call26,
    adrp_page,
    add_lo12,
    ldr_lo12,
};

/// One patch to apply during linking. `site` is the byte offset of the word to
/// patch within its owning function's `code`. `addend` is part of the shape for
/// M5; it is 0 for every M3 reloc.
pub const Reloc = struct {
    site: u32,
    target: SymbolId,
    kind: RelocKind,
    addend: i64 = 0,
};

/// One decoded string literal a function references, keyed by its content hash
/// (the same hash a `.cstr` reloc target carries). `bytes` are the decoded
/// runtime bytes WITHOUT a trailing NUL (the relink tail appends the NUL when it
/// builds the program-wide `__cstring` blob). Owned by its `FnCode`.
pub const Literal = struct {
    hash: u64,
    bytes: []u8,
};

/// One independently-lowered function. `sym` is its stable identity; the linker
/// interns it to a dense handle for the offset map. `code`/`relocs`/`literals`
/// are heap-allocated and owned by the producer until linked.
///
/// OWNERSHIP ("always own"): every name (the fn's own `sym.name` and each
/// `.func`/`.import` reloc target name) is heap-owned by this `FnCode`, whether
/// it was freshly lowered (Codegen dupes at emit) or unpacked from disk. One
/// `deinit`, no `owned` flag, no double-free ambiguity. [C9]
pub const FnCode = struct {
    sym: SymName,
    code: []u8,
    relocs: []Reloc,
    literals: []Literal,

    pub fn deinit(fc: *FnCode, gpa: std.mem.Allocator) void {
        gpa.free(fc.sym.name);
        gpa.free(fc.code);
        for (fc.relocs) |r| switch (r.target) {
            .func, .import => |s| gpa.free(s.name),
            .cstr => {},
        };
        gpa.free(fc.relocs);
        for (fc.literals) |l| gpa.free(l.bytes);
        gpa.free(fc.literals);
        fc.* = undefined;
    }
};

/// In-session symbol→handle map. Never persisted: the on-disk identity is the
/// `SymName`; this just gives `link` an `offsets: []u32` it can index by a dense
/// u32 instead of a string-hash per call site. The composite `[kind byte][name]`
/// key keeps `user_fn "f"` ≠ `import "f"`.
///
/// HANDLES ARE A STABLE-SORT RANK, NOT ARRIVAL ORDER ([M18 DETERMINISM #1]):
/// `internAll` assigns each fn's handle as its rank in a stable sort of the fn
/// set by composite key, ties broken by original source index. This makes the
/// handle (and therefore every offset/concat/patch keyed by it) a pure function
/// of the fn SET, independent of how/when the keys were derived — the invariant
/// that lets the M18 parallel tail intern symbols off the worker threads without
/// any thread-arrival/pointer/hashmap-iteration order leaking into the output.
pub const SymInterner = struct {
    map: std.StringHashMapUnmanaged(u32) = .empty,
    count: u32 = 0,

    pub fn get(si: *const SymInterner, gpa: std.mem.Allocator, ref: SymName) !?u32 {
        const key = try encodeSym(gpa, ref);
        defer gpa.free(key);
        return si.map.get(key);
    }

    /// Assign EVERY fn in `fns` a dense handle = its rank in a STABLE SORT of the
    /// set by composite `[kind byte][name]` key, ties broken by original source
    /// index. Two phases: (A) derive each fn's key (pure, read-only over `fns`),
    /// (B) stable-sort indices by key then build the key→handle map from the
    /// sorted order. The map is the only owner of each key (freed by `deinit`).
    ///
    /// Determinism: the result is a function of the fn SET alone — never `count`,
    /// thread-arrival, pointer, or hashmap-iteration order. Re-interning the same
    /// fns in any input order yields the IDENTICAL handle→name mapping, so every
    /// downstream consumer keyed by `get(name)` (offsets, concat, call26 deltas,
    /// entry resolution) produces byte-identical output regardless of order.
    pub fn internAll(si: *SymInterner, gpa: std.mem.Allocator, fns: []const FnCode) !void {
        const keys = try gpa.alloc([]u8, fns.len);
        // Phase A failure: free the keys derived so far, then the spine.
        var derived: usize = 0;
        errdefer {
            for (keys[0..derived]) |k| gpa.free(k);
            gpa.free(keys);
        }
        for (fns, 0..) |f, i| {
            keys[i] = try encodeSym(gpa, f.sym);
            derived += 1;
        }
        defer gpa.free(keys); // the map dupes ownership of the survivors below

        const order = try gpa.alloc(u32, fns.len);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);

        const SortCtx = struct {
            keys: []const []const u8,
            // Stable order: composite key bytes, then original source index for ties.
            fn lessThan(c: @This(), a: u32, b: u32) bool {
                return switch (std.mem.order(u8, c.keys[a], c.keys[b])) {
                    .lt => true,
                    .gt => false,
                    .eq => a < b,
                };
            }
        };
        std.mem.sortUnstable(u32, order, SortCtx{ .keys = keys }, SortCtx.lessThan);

        // Build the map in sorted order: handle = rank. The composite key carries
        // its kind byte, so dedup here would only fire on a genuine duplicate
        // definition (an upstream error); each distinct fn gets a unique handle.
        try si.map.ensureTotalCapacity(gpa, @intCast(fns.len));
        var placed: usize = 0;
        errdefer {
            // Free only the keys NOT yet handed to the map (the map owns the rest,
            // released by `deinit`). `order[placed..]` are the still-unplaced keys.
            for (order[placed..]) |src| gpa.free(keys[src]);
        }
        for (order, 0..) |src, rank| {
            const gop = si.map.getOrPutAssumeCapacity(keys[src]);
            if (gop.found_existing) {
                gpa.free(keys[src]); // duplicate definition: drop the redundant key
            } else {
                gop.value_ptr.* = @intCast(rank);
                si.count += 1;
            }
            placed += 1;
        }
    }

    pub fn deinit(si: *SymInterner, gpa: std.mem.Allocator) void {
        var it = si.map.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        si.map.deinit(gpa);
        si.* = undefined;
    }
};

/// Encode a `SymName` into a freshly-allocated composite `[kind byte][name]`
/// key. The tag byte keeps `user_fn "f"` ≠ `import "f"`. Caller owns the result.
/// Sized from the name, so an arbitrarily long identifier cannot overflow.
fn encodeSym(gpa: std.mem.Allocator, ref: SymName) ![]u8 {
    const key = try gpa.alloc(u8, 1 + ref.name.len);
    key[0] = @intFromEnum(ref.kind);
    @memcpy(key[1..], ref.name);
    return key;
}

/// The linker's output: the concatenated __text blob, the entry function's
/// resolved text offset (threaded into MachO's LC_MAIN entryoff), and the
/// cross-segment relocations rebased to absolute __text offsets. `link` CANNOT
/// patch the cross-segment relocs — they need the final vmaddrs of __text (the
/// site) and __cstring/__got (the target), which only MachO assigns. So `link`
/// stays a pure intra-module relocator and hands these out for `applyDataRelocs`
/// to patch after layout. `data_relocs` is heap-owned by the caller.
pub const Linked = struct {
    text: []u8,
    entry_off: u32,
    data_relocs: []Reloc,
};

/// Raised when a `.call26` displacement does not fit AArch64's signed imm26
/// (±128 MiB in words). It cannot occur within one 0x4000 __TEXT page, but we
/// surface it cleanly rather than truncate-and-miscompile.
pub const LinkError = error{ CallTargetTooFar, UnresolvedSymbol, NoEntry } || std.mem.Allocator.Error;

/// Assign every fn its __text byte offset by an EXCLUSIVE PREFIX-SUM of code
/// lengths over SOURCE ORDER, writing each into `offsets` keyed by the fn's stable
/// handle; returns the total text size. `offsets.len == fns.len` and `si` must
/// already have interned `fns`.
///
/// WHY a standalone scan ([M18 PREFIX-SUM == SERIAL RUNNING SUM]): this is the one
/// true ordering barrier of the parallel tail (every fn's address depends on the
/// sum of all earlier ones), so it is factored out as a pure left-to-right scan
/// that the parallel tail can replace with a real parallel prefix-scan WITHOUT
/// touching the layout values. The result is byte-for-byte the value the old fused
/// `offsets[h]=cursor; cursor+=len` running sum produced — the running tally is
/// `acc`, the offset stored is the EXCLUSIVE prefix (acc BEFORE adding this fn).
fn prefixSumTextOffsets(
    gpa: std.mem.Allocator,
    fns: []const FnCode,
    si: *const SymInterner,
    offsets: []u32,
) !u32 {
    var acc: u32 = 0;
    for (fns) |f| {
        std.debug.assert(f.code.len % 4 == 0);
        const h = (try si.get(gpa, f.sym)).?; // interned by internAll
        offsets[h] = acc; // exclusive prefix: bytes laid down before this fn
        acc += @intCast(f.code.len);
    }
    return acc;
}

/// Lay the independently-lowered functions out contiguously in __text, build the
/// symbol→offset map, then patch every relocation in place; return the joined
/// blob and the entry function's resolved offset.
///
/// LAYOUT is source order (each `code.len` is already a multiple of 4, so the
/// blob stays 4-byte aligned with no padding). The map is indexed by `f.sym` —
/// the function's source index — so `main` need not be first.
///
/// PATCH (`.call26`): the AArch64 `bl` is PC-relative by a signed *word* delta,
/// imm26 = (target_off − site_off) / 4. Forward calls give a positive delta,
/// backward (and self/mutual) recursion a non-positive one; both encode directly
/// via `Aarch64.bl`. Because the delta is intra-module it is fixed at link time
/// and needs NO runtime relocation under PIE/ASLR.
pub fn link(io: Io, gpa: std.mem.Allocator, fns: []const FnCode, si: *SymInterner, entry: SymName) LinkError!Linked {
    // 1) LAYOUT: assign every fn a dense handle by STABLE SORT of the fn set
    //    (rank, never arrival/source order — the M18 determinism invariant), then
    //    walk `fns` in SOURCE ORDER to give each its text offset. Layout STAYS
    //    source order (cursor walks `fns`), so __text bytes are byte-identical to
    //    the pre-M18 baseline; only the handle VALUES that key the offset map
    //    change, and every consumer below keys by `get(f.sym)`, so the bytes the
    //    name resolves to are unchanged.
    try si.internAll(gpa, fns);

    const offsets = try gpa.alloc(u32, fns.len);
    defer gpa.free(offsets);

    // Text offsets by EXCLUSIVE PREFIX-SUM of code lengths in SOURCE ORDER
    // ([M18 PREFIX-SUM == SERIAL RUNNING SUM]). `offsets[h]` is the sum of the
    // code lengths of every fn appearing before fn-h in source order — exactly the
    // value the old `offsets[h]=cursor; cursor+=len` running sum produced, so
    // __text stays byte-identical to the pre-M18 baseline. The scan is split into
    // an explicit length gather + prefix accumulation (instead of one fused cursor
    // walk) so the address-assignment barrier is a standalone, parallel-scan-ready
    // pass; the math (a left-to-right running sum over source order) is unchanged,
    // only WHO computes it.
    const cursor = try prefixSumTextOffsets(gpa, fns, si, offsets);

    const text = try gpa.alloc(u8, cursor);
    errdefer gpa.free(text);

    // Resolve every fn's handle + every reloc target handle BEFORE the fan-out: the
    // SymInterner's StringHashMap `get` allocates a scratch key, so it is NOT safe to
    // call concurrently. Resolving here (single-threaded, source order) makes the
    // per-fn jobs pure disjoint-region writers over read-only state. `site_h` is the
    // owning fn's handle; `targets[i]` holds each call26 target's handle in the same
    // flat order the per-fn loop visits relocs (so the job indexes it by a running
    // per-fn cursor). A stale call26 target name surfaces as UnresolvedSymbol here,
    // exactly where the old serial loop raised it. [C4]
    const site_h = try gpa.alloc(u32, fns.len);
    defer gpa.free(site_h);
    var n_call26: usize = 0;
    for (fns, 0..) |f, i| {
        site_h[i] = (try si.get(gpa, f.sym)).?;
        for (f.relocs) |rl| if (rl.kind == .call26) {
            n_call26 += 1;
        };
    }
    const call_targets = try gpa.alloc(u32, n_call26);
    defer gpa.free(call_targets);
    {
        var k: usize = 0;
        for (fns) |f| {
            for (f.relocs) |rl| if (rl.kind == .call26) {
                call_targets[k] = (try si.get(gpa, rl.target.func)) orelse return error.UnresolvedSymbol;
                k += 1;
            };
        }
    }
    // First-call26-index of each fn, so a job can slice its own targets without
    // re-scanning earlier fns (exclusive prefix-sum over per-fn call26 counts).
    const call_base = try gpa.alloc(usize, fns.len);
    defer gpa.free(call_base);
    {
        var acc: usize = 0;
        for (fns, 0..) |f, i| {
            call_base[i] = acc;
            for (f.relocs) |rl| if (rl.kind == .call26) {
                acc += 1;
            };
        }
    }

    // 2+3) FUSED per-fn COPY + call26 PATCH, run in parallel: each job owns ONLY its
    //      own disjoint `text[offsets[h]..offsets[h]+len)` region (copy + every
    //      in-region call26 site), and collects its NON-call26 relocs into its OWN
    //      slot's local list. No two jobs touch the same byte; `offsets`/`text`/
    //      `call_targets`/`site_h` are read-only across workers (the only writes are
    //      to a job's own text region + its own slot). Determinism is by INDEX, not
    //      arrival: the data_relocs are stable-concatenated in source order below, so
    //      the slice is byte-identical at any -j ([M18 NO MAP-ITERATION-ORDER]).
    const jobs = try gpa.alloc(FnLinkSlot, fns.len);
    defer {
        for (jobs) |*j| j.relocs.deinit(gpa);
        gpa.free(jobs);
    }
    for (jobs) |*j| j.* = .{};

    const Ctx = struct {
        gpa: std.mem.Allocator,
        fns: []const FnCode,
        text: []u8,
        offsets: []const u32,
        site_h: []const u32,
        call_targets: []const u32,
        call_base: []const usize,
        jobs: []FnLinkSlot,
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(fnLinkJob)) {
            return .{ c.gpa, c.fns, c.text, c.offsets, c.site_h, c.call_targets, c.call_base, i, &c.jobs[i] };
        }
    };
    Engine.fanOut(io, fns.len, fnLinkJob, Ctx{
        .gpa = gpa,
        .fns = fns,
        .text = text,
        .offsets = offsets,
        .site_h = site_h,
        .call_targets = call_targets,
        .call_base = call_base,
        .jobs = jobs,
    });

    // First per-fn error wins (source order), so the reported error is the same one
    // the old serial loop would have hit first.
    for (jobs) |j| if (j.err) |e| return e;

    // STABLE-CONCAT the per-fn data_reloc lists in SOURCE ORDER (the `jobs`/`fns`
    // index), reproducing the exact order the old single shared-append loop produced
    // — never thread-arrival order. The `.import` name-dup loop downstream observes
    // this order, so it must be a pure function of the fn set. [M18 determinism]
    var data_relocs: std.ArrayList(Reloc) = .empty;
    errdefer data_relocs.deinit(gpa);
    for (jobs) |j| try data_relocs.appendSlice(gpa, j.relocs.items);

    // 4) ENTRY: the entry function's resolved text offset, resolved by name.
    const entry_h = (try si.get(gpa, entry)) orelse return error.NoEntry;
    return .{
        .text = text,
        .entry_off = offsets[entry_h],
        .data_relocs = try data_relocs.toOwnedSlice(gpa),
    };
}

/// One per-fn link job's output: the cross-segment relocs this fn contributed (in
/// its own in-fn order) plus any error it hit. The job writes ONLY this slot and
/// its own `text` region; the caller stable-concatenates `relocs` in source order.
const FnLinkSlot = struct {
    relocs: std.ArrayList(Reloc) = .empty,
    err: ?LinkError = null,
};

/// The per-fn COPY + call26 PATCH unit (a `fanOut` job): copy fn `i`'s code into its
/// disjoint `text` region, patch every in-region `.call26` site in place (PC-relative
/// `bl`), and collect its non-call26 relocs (rebased to absolute __text offsets) into
/// `slot`. Pure over disjoint state: writes only `text[offsets[h]..]` (this fn's) and
/// `slot`. All symbol handles were resolved single-threaded before the fan-out, so no
/// `SymInterner.get` (which allocates) runs here.
fn fnLinkJob(
    gpa: std.mem.Allocator,
    fns: []const FnCode,
    text: []u8,
    offsets: []const u32,
    site_h: []const u32,
    call_targets: []const u32,
    call_base: []const usize,
    i: usize,
    slot: *FnLinkSlot,
) void {
    const f = fns[i];
    const fh = site_h[i];
    const base = offsets[fh];
    @memcpy(text[base..][0 .. f.code.len], f.code);

    var call_i: usize = call_base[i];
    for (f.relocs) |rl| {
        const site_abs: u32 = base + rl.site;
        switch (rl.kind) {
            .call26 => {
                const th = call_targets[call_i];
                call_i += 1;
                const target_abs: i64 = @as(i64, offsets[th]) + rl.addend;
                const delta: i64 = target_abs - @as(i64, site_abs);
                std.debug.assert(@mod(delta, 4) == 0); // BL targets are word-aligned
                const imm: i64 = @divExact(delta, 4);
                if (imm < -(@as(i64, 1) << 25) or imm > (@as(i64, 1) << 25) - 1) {
                    if (slot.err == null) slot.err = error.CallTargetTooFar;
                    continue;
                }
                const word = Aarch64.bl(@intCast(imm));
                std.mem.writeInt(u32, text[site_abs..][0..4], word, .little);
            },
            .adrp_page, .add_lo12, .ldr_lo12 => {
                slot.relocs.append(gpa, .{
                    .site = site_abs,
                    .target = rl.target,
                    .kind = rl.kind,
                    .addend = rl.addend,
                }) catch |e| {
                    if (slot.err == null) slot.err = e;
                };
            },
        }
    }
}

/// Patch the cross-segment relocations `link` could not resolve, now that MachO
/// has assigned every segment's vmaddr. `text` is the __text bytes in the final
/// image; each reloc's `site` is already absolute within __text. `text_vmaddr`
/// is the runtime address __text loads at; `cstring_vmaddr`/`got_vmaddr` are the
/// __cstring section and __got section base vmaddrs.
///
/// The page delta uses the 4096-byte ARM page (`>>12`), NOT the 0x4000 Mach-O
/// page. Because every segment is page-aligned in the same image, the deltas are
/// exact at runtime under PIE/ASLR (dyld slides all segments together).
pub fn applyDataRelocs(
    io: Io,
    text: []u8,
    data_relocs: []const Reloc,
    text_vmaddr: u64,
    cstring_vmaddr: u64,
    got_vmaddr: u64,
    import_slots: *const std.StringHashMapUnmanaged(u32),
) void {
    // PARALLEL per-reloc PATCH: each reloc patches a UNIQUE disjoint 4-byte site
    // (`text[rl.site..site+4]`), so the fan-out is a race-free disjoint-region map.
    // `import_slots`/`text_vmaddr`/etc. are read-only across workers; the output is
    // independent of dispatch order because each site is written by exactly one
    // worker. [M18 SINGLE-WRITER DISJOINT REGIONS]
    const Ctx = struct {
        text: []u8,
        data_relocs: []const Reloc,
        text_vmaddr: u64,
        cstring_vmaddr: u64,
        got_vmaddr: u64,
        import_slots: *const std.StringHashMapUnmanaged(u32),
        pub fn args(c: @This(), i: usize) std.meta.ArgsTuple(@TypeOf(dataRelocJob)) {
            return .{ c.text, c.data_relocs[i], c.text_vmaddr, c.cstring_vmaddr, c.got_vmaddr, c.import_slots };
        }
    };
    Engine.fanOut(io, data_relocs.len, dataRelocJob, Ctx{
        .text = text,
        .data_relocs = data_relocs,
        .text_vmaddr = text_vmaddr,
        .cstring_vmaddr = cstring_vmaddr,
        .got_vmaddr = got_vmaddr,
        .import_slots = import_slots,
    });
}

/// Patch ONE cross-segment reloc's 4-byte site (a `fanOut` job). Reads `text[rl.site]`,
/// computes the adrp/add/ldr immediate from the target vmaddr, writes it back. Each
/// job owns a unique disjoint site, so no two jobs touch the same byte.
fn dataRelocJob(
    text: []u8,
    rl: Reloc,
    text_vmaddr: u64,
    cstring_vmaddr: u64,
    got_vmaddr: u64,
    import_slots: *const std.StringHashMapUnmanaged(u32),
) void {
    const site_vmaddr: u64 = text_vmaddr + rl.site;
    const target_vmaddr: u64 = switch (rl.target) {
        // The relink tail already rewrote the content hash to a blob offset.
        .cstr => |off| cstring_vmaddr + off,
        .import => |s| got_vmaddr + @as(u64, import_slots.get(s.name).?) * 8,
        .func => unreachable, // funcs are patched intra-module by `link`
    };
    const word = std.mem.readInt(u32, text[rl.site..][0..4], .little);
    const patched: u32 = switch (rl.kind) {
        .adrp_page => blk: {
            const pages: i64 = @as(i64, @intCast(target_vmaddr >> 12)) -
                @as(i64, @intCast(site_vmaddr >> 12));
            break :blk Aarch64.patchAdrp(word, @intCast(pages));
        },
        .add_lo12 => Aarch64.patchAddImm12(word, @intCast(target_vmaddr & 0xFFF)),
        .ldr_lo12 => blk: {
            std.debug.assert(target_vmaddr & 7 == 0);
            break :blk Aarch64.patchLdrUoff(word, @intCast(target_vmaddr & 0xFFF));
        },
        .call26 => unreachable, // already patched by `link`
    };
    std.mem.writeInt(u32, text[rl.site..][0..4], patched, .little);
}

// ---- FnCode (de)serialization ----------------------------------------------
//
// A lowered `FnCode` is the M5 cache payload. `Reloc.target` is a tagged union
// with an `i64` addend, so it is NOT raw-memcpy-able; serialize each region into
// `extern` records inside the same `[u64 checksum][payload]` envelope `Cache`
// uses for the parse Tree. Blob layout:
//   [FnHeader][sym name][code][RelocRec×n][names pool][LitRec×m][lits pool]
// Names (reloc `.func`/`.import` targets) are pooled; each RelocRec carries an
// (offset,len) into that pool. `unpack` reconstructs an "always own" FnCode so a
// hit and a fresh lower share one `deinit`. [C9]

/// "TOFC" — a magic so a foreign/corrupt blob is treated as a cache miss.
pub const fncode_magic: u32 = 0x544f4643;

const FnHeader = extern struct {
    magic: u32,
    version: u32 = 1,
    sym_kind: u8,
    _pad: [3]u8 = .{ 0, 0, 0 },
    sym_name_len: u32,
    code_len: u32,
    reloc_count: u32,
    lit_count: u32,
    names_len: u32,
    lits_len: u32,
};

const RelocRec = extern struct {
    site: u32,
    kind: u8,
    tgt_tag: u8, // 0=func, 1=cstr, 2=import (matches SymbolId field order)
    tgt_kind: u8, // SymKind for func/import; 0 for cstr
    _pad: u8 = 0,
    name_off: u32, // into the names pool (func/import); 0 for cstr
    name_len: u32, // 0 for cstr
    cstr_hash: u64, // content hash for cstr; 0 otherwise
    addend: i64,
};

const LitRec = extern struct {
    hash: u64,
    off: u32, // into the lits pool
    len: u32,
};

comptime {
    std.debug.assert(@sizeOf(FnHeader) % 4 == 0);
    std.debug.assert(@sizeOf(RelocRec) % 8 == 0); // keeps the i64/u64 fields aligned
    std.debug.assert(@sizeOf(LitRec) % 8 == 0);
}

/// Serialize `fc` into one flat blob (caller owns it). Deterministic: the same
/// `FnCode` always produces byte-identical output, so VERIFY mode can compare a
/// re-lowered blob to the cached one. [C11]
pub fn pack(gpa: std.mem.Allocator, fc: FnCode) ![]u8 {
    var names_len: usize = 0;
    for (fc.relocs) |r| switch (r.target) {
        .func, .import => |s| names_len += s.name.len,
        .cstr => {},
    };
    var lits_len: usize = 0;
    for (fc.literals) |l| lits_len += l.bytes.len;

    const total = @sizeOf(FnHeader) + fc.sym.name.len + fc.code.len +
        fc.relocs.len * @sizeOf(RelocRec) + names_len +
        fc.literals.len * @sizeOf(LitRec) + lits_len;
    const buf = try gpa.alloc(u8, total);
    errdefer gpa.free(buf);

    const hdr = FnHeader{
        .magic = fncode_magic,
        .sym_kind = @intFromEnum(fc.sym.kind),
        .sym_name_len = @intCast(fc.sym.name.len),
        .code_len = @intCast(fc.code.len),
        .reloc_count = @intCast(fc.relocs.len),
        .lit_count = @intCast(fc.literals.len),
        .names_len = @intCast(names_len),
        .lits_len = @intCast(lits_len),
    };
    @memcpy(buf[0..@sizeOf(FnHeader)], std.mem.asBytes(&hdr));
    var off: usize = @sizeOf(FnHeader);

    @memcpy(buf[off .. off + fc.sym.name.len], fc.sym.name);
    off += fc.sym.name.len;
    @memcpy(buf[off .. off + fc.code.len], fc.code);
    off += fc.code.len;

    // RelocRec array; names pool is laid out in the same iteration order, so a
    // RelocRec's name_off is a running cursor into it.
    var name_cursor: u32 = 0;
    for (fc.relocs) |r| {
        var rec = RelocRec{
            .site = r.site,
            .kind = @intFromEnum(r.kind),
            .tgt_tag = undefined,
            .tgt_kind = 0,
            .name_off = 0,
            .name_len = 0,
            .cstr_hash = 0,
            .addend = r.addend,
        };
        switch (r.target) {
            .func => |s| {
                rec.tgt_tag = 0;
                rec.tgt_kind = @intFromEnum(s.kind);
                rec.name_off = name_cursor;
                rec.name_len = @intCast(s.name.len);
                name_cursor += @intCast(s.name.len);
            },
            .cstr => |h| {
                rec.tgt_tag = 1;
                rec.cstr_hash = h;
            },
            .import => |s| {
                rec.tgt_tag = 2;
                rec.tgt_kind = @intFromEnum(s.kind);
                rec.name_off = name_cursor;
                rec.name_len = @intCast(s.name.len);
                name_cursor += @intCast(s.name.len);
            },
        }
        @memcpy(buf[off .. off + @sizeOf(RelocRec)], std.mem.asBytes(&rec));
        off += @sizeOf(RelocRec);
    }
    // Names pool.
    for (fc.relocs) |r| switch (r.target) {
        .func, .import => |s| {
            @memcpy(buf[off .. off + s.name.len], s.name);
            off += s.name.len;
        },
        .cstr => {},
    };

    // LitRec array + lits pool.
    var lit_cursor: u32 = 0;
    for (fc.literals) |l| {
        const rec = LitRec{ .hash = l.hash, .off = lit_cursor, .len = @intCast(l.bytes.len) };
        @memcpy(buf[off .. off + @sizeOf(LitRec)], std.mem.asBytes(&rec));
        off += @sizeOf(LitRec);
        lit_cursor += @intCast(l.bytes.len);
    }
    for (fc.literals) |l| {
        @memcpy(buf[off .. off + l.bytes.len], l.bytes);
        off += l.bytes.len;
    }

    std.debug.assert(off == total);
    return buf;
}

/// Reconstruct a `FnCode` from a blob produced by `pack`. Returns null on any
/// mismatch (corrupt/foreign blob) so the caller treats it as a cache miss. The
/// result is "always own": every name + byte slice is heap-duped from the blob,
/// so it frees uniformly via `FnCode.deinit`. Caller owns the result.
pub fn unpack(gpa: std.mem.Allocator, bytes: []const u8) !?FnCode {
    if (bytes.len < @sizeOf(FnHeader)) return null;
    var hdr: FnHeader = undefined;
    @memcpy(std.mem.asBytes(&hdr), bytes[0..@sizeOf(FnHeader)]);
    if (hdr.magic != fncode_magic or hdr.version != 1) return null;
    if (hdr.sym_kind > @intFromEnum(SymKind.import)) return null;

    const need = @sizeOf(FnHeader) + @as(usize, hdr.sym_name_len) + @as(usize, hdr.code_len) +
        @as(usize, hdr.reloc_count) * @sizeOf(RelocRec) + @as(usize, hdr.names_len) +
        @as(usize, hdr.lit_count) * @sizeOf(LitRec) + @as(usize, hdr.lits_len);
    if (bytes.len != need) return null;

    var off: usize = @sizeOf(FnHeader);

    // sym name
    const sym_name = try gpa.dupe(u8, bytes[off .. off + hdr.sym_name_len]);
    errdefer gpa.free(sym_name);
    off += hdr.sym_name_len;

    // code
    const code = try gpa.alloc(u8, hdr.code_len);
    errdefer gpa.free(code);
    @memcpy(code, bytes[off .. off + hdr.code_len]);
    off += hdr.code_len;

    // reloc records (followed by the names pool)
    const recs_off = off;
    off += @as(usize, hdr.reloc_count) * @sizeOf(RelocRec);
    const names_pool = bytes[off .. off + hdr.names_len];
    off += hdr.names_len;

    const relocs = try gpa.alloc(Reloc, hdr.reloc_count);
    // On a mid-loop failure, free only the names duped so far + the array.
    var built: usize = 0;
    errdefer {
        for (relocs[0..built]) |r| switch (r.target) {
            .func, .import => |s| gpa.free(s.name),
            .cstr => {},
        };
        gpa.free(relocs);
    }
    for (relocs, 0..) |*out_rl, i| {
        var rec: RelocRec = undefined;
        const rec_at = recs_off + i * @sizeOf(RelocRec);
        @memcpy(std.mem.asBytes(&rec), bytes[rec_at .. rec_at + @sizeOf(RelocRec)]);
        if (rec.kind > @intFromEnum(RelocKind.ldr_lo12)) return null;
        const kind: RelocKind = @enumFromInt(rec.kind);
        const target: SymbolId = switch (rec.tgt_tag) {
            0, 2 => blk: {
                if (rec.tgt_kind > @intFromEnum(SymKind.import)) return null;
                if (@as(usize, rec.name_off) + rec.name_len > names_pool.len) return null;
                const nm = try gpa.dupe(u8, names_pool[rec.name_off .. rec.name_off + rec.name_len]);
                const sn = SymName{ .kind = @enumFromInt(rec.tgt_kind), .name = nm };
                break :blk if (rec.tgt_tag == 0) SymbolId{ .func = sn } else SymbolId{ .import = sn };
            },
            1 => SymbolId{ .cstr = rec.cstr_hash },
            else => return null,
        };
        out_rl.* = .{ .site = rec.site, .target = target, .kind = kind, .addend = rec.addend };
        built += 1;
    }

    // literal records (followed by the lits pool)
    const lits_recs_off = off;
    off += @as(usize, hdr.lit_count) * @sizeOf(LitRec);
    const lits_pool = bytes[off .. off + hdr.lits_len];
    off += hdr.lits_len;

    const literals = try gpa.alloc(Literal, hdr.lit_count);
    var lits_built: usize = 0;
    errdefer {
        for (literals[0..lits_built]) |l| gpa.free(l.bytes);
        gpa.free(literals);
    }
    for (literals, 0..) |*out_lit, i| {
        var rec: LitRec = undefined;
        const rec_at = lits_recs_off + i * @sizeOf(LitRec);
        @memcpy(std.mem.asBytes(&rec), bytes[rec_at .. rec_at + @sizeOf(LitRec)]);
        if (@as(usize, rec.off) + rec.len > lits_pool.len) return null;
        const b = try gpa.dupe(u8, lits_pool[rec.off .. rec.off + rec.len]);
        out_lit.* = .{ .hash = rec.hash, .bytes = b };
        lits_built += 1;
    }

    return FnCode{
        .sym = .{ .kind = @enumFromInt(hdr.sym_kind), .name = sym_name },
        .code = code,
        .relocs = relocs,
        .literals = literals,
    };
}

// ---------------------------------------------------------------------------
// Tests — bl ground truth was assembled on this host (`as -arch arm64` +
// objdump): forward +8 → 0x94000002, backward −12 → 0x97fffffd, self (delta 0)
// → 0x94000000. Each test decodes the patched imm26 back to confirm it points
// at the callee.
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Decode a patched `bl` word's signed imm26 back into a byte delta.
fn decodeBlDelta(word: u32) i64 {
    const imm: i26 = @bitCast(@as(u26, @truncate(word)));
    return @as(i64, imm) * 4;
}

/// A static stable name for test function `idx`: a `user_fn` named "fN". The
/// strings are comptime literals (static lifetime), so test FnCodes can carry
/// them and `freeFns` only frees code/relocs/literals, not these names.
fn fname(comptime idx: u32) SymName {
    return .{ .kind = .user_fn, .name = std.fmt.comptimePrint("f{d}", .{idx}) };
}

/// Make a heap FnCode named "f<sym>" of `nwords` placeholder `bl #0` words plus
/// the given relocs (relocs are duped; targets reference static names, so no
/// name allocation). Freed by `freeFns`.
fn makeFn(gpa: std.mem.Allocator, comptime sym: u32, nwords: usize, relocs: []const Reloc) !FnCode {
    const code = try gpa.alloc(u8, nwords * 4);
    var i: usize = 0;
    while (i < nwords) : (i += 1) {
        std.mem.writeInt(u32, code[i * 4 ..][0..4], Aarch64.bl(0), .little);
    }
    const rl = try gpa.dupe(Reloc, relocs);
    return .{ .sym = fname(sym), .code = code, .relocs = rl, .literals = &.{} };
}

/// Free the test FnCodes' heap parts. The names are comptime statics (see
/// `fname`), so we do NOT free `sym.name` / reloc target names here.
fn freeFns(gpa: std.mem.Allocator, fns: []FnCode) void {
    for (fns) |f| {
        gpa.free(f.code);
        gpa.free(f.relocs);
    }
}

/// Link the test fns with a fresh interner and entry named "f<entry>". Drives the
/// fan-out on a serial (`.limited(0)`) pool so the unit tests stay deterministic
/// and independent of the host's thread count.
fn linkTest(gpa: std.mem.Allocator, fns: []const FnCode, comptime entry: u32) LinkError!Linked {
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer threaded.deinit();
    var si: SymInterner = .{};
    defer si.deinit(gpa);
    return link(threaded.io(), gpa, fns, &si, fname(entry));
}

test "layout offsets follow source order and are contiguous" {
    const gpa = testing.allocator;
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 3, &.{}), // 12 bytes at offset 0
        try makeFn(gpa, 1, 1, &.{}), // 4 bytes at offset 12
        try makeFn(gpa, 2, 2, &.{}), // 8 bytes at offset 16
    };
    defer freeFns(gpa, &fns);

    const linked = try linkTest(gpa, &fns, 0);
    defer gpa.free(linked.text);
    defer gpa.free(linked.data_relocs);

    try testing.expectEqual(@as(usize, 24), linked.text.len);
    try testing.expectEqual(@as(u32, 0), linked.entry_off);
}

test "forward call patched to positive word delta" {
    const gpa = testing.allocator;
    // fn0 (caller, 2 words) has a bl at site 0 targeting fn1 (callee) which lands
    // two words ahead → target_off 8, site_off 0, delta +8 → word 0x94000002.
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 2, &.{.{ .site = 0, .target = .{ .func = fname(1) }, .kind = .call26 }}),
        try makeFn(gpa, 1, 1, &.{}),
    };
    defer freeFns(gpa, &fns);

    var linked = try linkTest(gpa, &fns, 0);
    defer gpa.free(linked.text);
    defer gpa.free(linked.data_relocs);

    const word = std.mem.readInt(u32, linked.text[0..4], .little);
    try testing.expectEqual(@as(u32, 0x94000002), word);
    // Decoded delta lands exactly on the callee's offset.
    try testing.expectEqual(@as(i64, 8), decodeBlDelta(word) + 0);
}

test "backward call patched to negative word delta" {
    const gpa = testing.allocator;
    // fn0 (callee, 1 word) at offset 0; fn1 (caller, 2 words) at offset 4; its bl
    // is the SECOND word (site 4 within fn1 → site_abs 8) targeting fn0 at 0 →
    // delta −8 → word 0x97fffffe. Also assert the −12 ground-truth value.
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 1, &.{}),
        try makeFn(gpa, 1, 2, &.{.{ .site = 4, .target = .{ .func = fname(0) }, .kind = .call26 }}),
    };
    defer freeFns(gpa, &fns);

    var linked = try linkTest(gpa, &fns, 0);
    defer gpa.free(linked.text);
    defer gpa.free(linked.data_relocs);

    const word = std.mem.readInt(u32, linked.text[8..12], .little);
    try testing.expectEqual(@as(u32, 0x97FFFFFE), word);
    // Decoded delta points back at the callee (site_abs 8 + (−8) = 0).
    try testing.expectEqual(@as(i64, 0), 8 + decodeBlDelta(word));
}

test "ground-truth backward -12 and self delta 0" {
    const gpa = testing.allocator;
    // Single fn, 4 words. Word at site 12 calls back to offset 0 → delta −12 →
    // 0x97fffffd. Word at site 0 self-calls (delta 0) → 0x94000000.
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 4, &.{
            .{ .site = 12, .target = .{ .func = fname(0) }, .kind = .call26 },
            .{ .site = 0, .target = .{ .func = fname(0) }, .kind = .call26 },
        }),
    };
    defer freeFns(gpa, &fns);

    var linked = try linkTest(gpa, &fns, 0);
    defer gpa.free(linked.text);
    defer gpa.free(linked.data_relocs);

    try testing.expectEqual(@as(u32, 0x97FFFFFD), std.mem.readInt(u32, linked.text[12..16], .little));
    try testing.expectEqual(@as(u32, 0x94000000), std.mem.readInt(u32, linked.text[0..4], .little));
}

test "entry_off is the entry symbol's offset, not necessarily first" {
    const gpa = testing.allocator;
    // main (f1) is laid out SECOND: helper (f0, 3 words) at 0, main at 12. The
    // entry is resolved BY NAME, so its layout position is irrelevant.
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 3, &.{}),
        try makeFn(gpa, 1, 1, &.{}),
    };
    defer freeFns(gpa, &fns);

    const linked = try linkTest(gpa, &fns, 1);
    defer gpa.free(linked.text);
    defer gpa.free(linked.data_relocs);

    try testing.expectEqual(@as(u32, 12), linked.entry_off);
}

test "reorder does not corrupt a call: same names, different layout order" {
    const gpa = testing.allocator;
    // [f0 calls f1, f1] then [f1, f0 calls f1] — the call resolves by NAME, so the
    // bl delta differs (layout moved) but always points at f1's body. [C4][vi]
    {
        var fns = [_]FnCode{
            try makeFn(gpa, 0, 2, &.{.{ .site = 0, .target = .{ .func = fname(1) }, .kind = .call26 }}),
            try makeFn(gpa, 1, 1, &.{}),
        };
        defer freeFns(gpa, &fns);
        var linked = try linkTest(gpa, &fns, 0);
        defer gpa.free(linked.text);
        defer gpa.free(linked.data_relocs);
        // f0 at 0, f1 at 8: site 0 → +8 → 0x94000002.
        try testing.expectEqual(@as(u32, 0x94000002), std.mem.readInt(u32, linked.text[0..4], .little));
    }
    {
        var fns = [_]FnCode{
            try makeFn(gpa, 1, 1, &.{}),
            try makeFn(gpa, 0, 2, &.{.{ .site = 0, .target = .{ .func = fname(1) }, .kind = .call26 }}),
        };
        defer freeFns(gpa, &fns);
        var linked = try linkTest(gpa, &fns, 0);
        defer gpa.free(linked.text);
        defer gpa.free(linked.data_relocs);
        // Now f1 at 0, f0 at 4: f0's bl at site_abs 4 → f1 at 0 → −4 → 0x97FFFFFF.
        try testing.expectEqual(@as(u32, 0x97FFFFFF), std.mem.readInt(u32, linked.text[4..8], .little));
    }
}

test "stable-sort handles: shuffled input order yields the identical handle map" {
    const gpa = testing.allocator;
    // The SAME fn set, interned in three different source orders. A handle is the
    // stable-sort RANK of the composite key (ties broken by source index), so the
    // handle→name mapping is a pure function of the SET — identical across orders,
    // never arrival/source order. This is the M18 determinism invariant: the
    // parallel tail can intern off worker threads without any thread-arrival order
    // leaking into the handles that key every offset/delta.
    const Fn = struct { kind: SymKind, name: []const u8 };
    // Names chosen so source order (b,a,c) ≠ sorted order (a,b,c) ≠ name length.
    const defs = [_]Fn{
        .{ .kind = .user_fn, .name = "beta" },
        .{ .kind = .user_fn, .name = "alpha" },
        .{ .kind = .import, .name = "alpha" }, // same name, different kind ⇒ distinct
        .{ .kind = .user_fn, .name = "gamma" },
    };

    const internOrder = struct {
        fn run(g: std.mem.Allocator, ds: []const Fn, order: []const usize, out: []u32) !void {
            var fns: [4]FnCode = undefined;
            var built: usize = 0;
            errdefer for (fns[0..built]) |*f| g.free(f.code);
            for (order, 0..) |src, i| {
                fns[i] = .{
                    .sym = .{ .kind = ds[src].kind, .name = ds[src].name },
                    .code = try g.alloc(u8, 4),
                    .relocs = &.{},
                    .literals = &.{},
                };
                built += 1;
            }
            defer for (&fns) |*f| g.free(f.code);

            var si: SymInterner = .{};
            defer si.deinit(g);
            try si.internAll(g, &fns);
            for (ds, 0..) |d, k| out[k] = (try si.get(g, .{ .kind = d.kind, .name = d.name })).?;
        }
    }.run;

    var h_src: [4]u32 = undefined; // interned in declaration order
    var h_rev: [4]u32 = undefined; // reversed
    var h_mix: [4]u32 = undefined; // arbitrary shuffle
    try internOrder(gpa, &defs, &.{ 0, 1, 2, 3 }, &h_src);
    try internOrder(gpa, &defs, &.{ 3, 2, 1, 0 }, &h_rev);
    try internOrder(gpa, &defs, &.{ 2, 0, 3, 1 }, &h_mix);

    try testing.expectEqual(h_src, h_rev);
    try testing.expectEqual(h_src, h_mix);
    // Sorted composite order is: ('f'=user_fn 0)"alpha", ("beta"), ("gamma"),
    // then ('import' tag)"alpha". So handles by def index {beta,alpha-fn,alpha-import,gamma}:
    try testing.expectEqual(@as(u32, 1), h_src[0]); // beta  → rank 1
    try testing.expectEqual(@as(u32, 0), h_src[1]); // alpha (user_fn) → rank 0
    try testing.expectEqual(@as(u32, 3), h_src[2]); // alpha (import) → rank 3
    try testing.expectEqual(@as(u32, 2), h_src[3]); // gamma → rank 2
}

test "prefix-sum text offsets == old cursor accumulation for a randomized length vector" {
    const gpa = testing.allocator;
    // The prefix-sum address barrier (prefixSumTextOffsets) MUST reproduce the old
    // fused `offsets[h]=cursor; cursor+=len` running sum byte-for-byte over a range
    // of fn-length vectors (incl. zero/large words, shuffled handle vs source order).
    var prng = std.Random.DefaultPrng.init(0x18_03_5e_ed);
    const rand = prng.random();

    var trial: usize = 0;
    while (trial < 64) : (trial += 1) {
        const n = rand.intRangeAtMost(usize, 1, 32);
        const lens = try gpa.alloc(usize, n); // words per fn
        defer gpa.free(lens);
        for (lens) |*l| l.* = rand.intRangeAtMost(usize, 0, 64);

        // Build fns named by RANDOM distinct ids so the stable-sort handle order
        // differs from source order — the prefix-sum is keyed by handle, the old
        // running sum walks source order; both must agree.
        const fns = try gpa.alloc(FnCode, n);
        defer {
            for (fns) |*f| {
                gpa.free(f.sym.name);
                gpa.free(f.code);
            }
            gpa.free(fns);
        }
        for (fns, 0..) |*f, i| {
            const name = try std.fmt.allocPrint(gpa, "fn_{d}_{d}", .{ rand.int(u32), i });
            f.* = .{
                .sym = .{ .kind = .user_fn, .name = name },
                .code = try gpa.alloc(u8, lens[i] * 4),
                .relocs = &.{},
                .literals = &.{},
            };
        }

        var si: SymInterner = .{};
        defer si.deinit(gpa);
        try si.internAll(gpa, fns);

        // OLD: fused running sum over source order, keyed by handle.
        const want = try gpa.alloc(u32, n);
        defer gpa.free(want);
        {
            var cursor: u32 = 0;
            for (fns) |f| {
                const h = (try si.get(gpa, f.sym)).?;
                want[h] = cursor;
                cursor += @intCast(f.code.len);
            }
        }

        // NEW: the standalone prefix-sum pass.
        const got = try gpa.alloc(u32, n);
        defer gpa.free(got);
        const total = try prefixSumTextOffsets(gpa, fns, &si, got);

        try testing.expectEqualSlices(u32, want, got);
        var expect_total: u32 = 0;
        for (lens) |l| expect_total += @intCast(l * 4);
        try testing.expectEqual(expect_total, total);
    }
}

test "UnresolvedSymbol on a call to a missing name" {
    const gpa = testing.allocator;
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 1, &.{.{ .site = 0, .target = .{ .func = fname(9) }, .kind = .call26 }}),
    };
    defer freeFns(gpa, &fns);
    try testing.expectError(error.UnresolvedSymbol, linkTest(gpa, &fns, 0));
}

test "self mutual recursion: forward then backward both correct" {
    const gpa = testing.allocator;
    // fn0 (2 words) at 0 calls fn1 (forward); fn1 (2 words) at 8 calls fn0 (back).
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 2, &.{.{ .site = 0, .target = .{ .func = fname(1) }, .kind = .call26 }}),
        try makeFn(gpa, 1, 2, &.{.{ .site = 0, .target = .{ .func = fname(0) }, .kind = .call26 }}),
    };
    defer freeFns(gpa, &fns);

    var linked = try linkTest(gpa, &fns, 0);
    defer gpa.free(linked.text);
    defer gpa.free(linked.data_relocs);

    // fn0's bl at abs 0 → fn1 at 8: delta +8 → 0x94000002.
    const w0 = std.mem.readInt(u32, linked.text[0..4], .little);
    try testing.expectEqual(@as(u32, 0x94000002), w0);
    try testing.expectEqual(@as(i64, 8), 0 + decodeBlDelta(w0));
    // fn1's bl at abs 8 → fn0 at 0: delta −8 → 0x97fffffe.
    const w1 = std.mem.readInt(u32, linked.text[8..12], .little);
    try testing.expectEqual(@as(u32, 0x97FFFFFE), w1);
    try testing.expectEqual(@as(i64, 0), 8 + decodeBlDelta(w1));
}

test "link rebases non-call26 relocs into data_relocs and leaves call26 in place" {
    const gpa = testing.allocator;
    // fn0 (2 words) at 0: a call26 to fn1 at site 0, plus an adrp_page reloc at
    // site 4 targeting a cstring (now a content hash, value-agnostic). fn1 at 8.
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 2, &.{
            .{ .site = 0, .target = .{ .func = fname(1) }, .kind = .call26 },
            .{ .site = 4, .target = .{ .cstr = 0x10 }, .kind = .adrp_page },
        }),
        try makeFn(gpa, 1, 1, &.{}),
    };
    defer freeFns(gpa, &fns);

    var linked = try linkTest(gpa, &fns, 0);
    defer gpa.free(linked.text);
    defer gpa.free(linked.data_relocs);

    // The call26 was patched in place (fn1 is two words ahead → +8 → 0x94000002).
    try testing.expectEqual(@as(u32, 0x94000002), std.mem.readInt(u32, linked.text[0..4], .little));
    // The adrp_page reloc was collected with an absolute __text site (fn0 at 0,
    // site 4 → 4) and is still an unpatched placeholder in the text.
    try testing.expectEqual(@as(usize, 1), linked.data_relocs.len);
    try testing.expectEqual(@as(u32, 4), linked.data_relocs[0].site);
    try testing.expectEqual(RelocKind.adrp_page, linked.data_relocs[0].kind);
    try testing.expectEqual(@as(u64, 0x10), linked.data_relocs[0].target.cstr);
}

test "applyDataRelocs reproduces clang adrp/add/ldr words" {
    // Ground truth from the clang `write` reference on this host:
    //   adrp x8, <cstr>   @ site 0x478, target 0x4b0 → pages 0 → 0x90000008
    //   add  x8, x8,#0x4b0                            → 0x9112c108
    //   adrp x16,<got>    @ site 0x4a4, target 0x4000 → pages 4 → 0x90000030
    //   ldr  x16,[x16]                                → 0xf9400210
    // `link` rebases reloc sites to absolute __text offsets, so set text_vmaddr =
    // clang site for a site-0 reloc and feed each instruction independently.
    const gpa = testing.allocator;
    var threaded = std.Io.Threaded.init(gpa, .{ .concurrent_limit = .limited(0) });
    defer threaded.deinit();
    const io = threaded.io();
    const base: u64 = 0x100000000;
    const cstring_vmaddr: u64 = base + 0x4b0;
    const got_vmaddr: u64 = base + 0x4000;

    // The single import `write` lives in GOT slot 0. The tail already rewrote the
    // `.cstr` hash to offset 0 (the literal sits at the start of __cstring).
    var import_slots: std.StringHashMapUnmanaged(u32) = .empty;
    defer import_slots.deinit(gpa);
    try import_slots.put(gpa, "write", 0);

    {
        var w: [4]u8 = undefined;
        std.mem.writeInt(u32, &w, Aarch64.adrp(8, 0), .little);
        const r = [_]Reloc{.{ .site = 0, .target = .{ .cstr = 0 }, .kind = .adrp_page }};
        applyDataRelocs(io, &w, &r, base + 0x478, cstring_vmaddr, got_vmaddr, &import_slots);
        try testing.expectEqual(@as(u32, 0x90000008), std.mem.readInt(u32, &w, .little));
    }
    {
        var w: [4]u8 = undefined;
        std.mem.writeInt(u32, &w, Aarch64.addImm(8, 8, 0), .little);
        const r = [_]Reloc{.{ .site = 0, .target = .{ .cstr = 0 }, .kind = .add_lo12 }};
        applyDataRelocs(io, &w, &r, base + 0x47c, cstring_vmaddr, got_vmaddr, &import_slots);
        try testing.expectEqual(@as(u32, 0x9112C108), std.mem.readInt(u32, &w, .little));
    }
    {
        var w: [4]u8 = undefined;
        std.mem.writeInt(u32, &w, Aarch64.adrp(16, 0), .little);
        const r = [_]Reloc{.{ .site = 0, .target = .{ .import = .{ .kind = .import, .name = "write" } }, .kind = .adrp_page }};
        applyDataRelocs(io, &w, &r, base + 0x4a4, cstring_vmaddr, got_vmaddr, &import_slots);
        try testing.expectEqual(@as(u32, 0x90000030), std.mem.readInt(u32, &w, .little));
    }
    {
        var w: [4]u8 = undefined;
        std.mem.writeInt(u32, &w, Aarch64.ldrRegUoff(16, 16, 0), .little);
        const r = [_]Reloc{.{ .site = 0, .target = .{ .import = .{ .kind = .import, .name = "write" } }, .kind = .ldr_lo12 }};
        applyDataRelocs(io, &w, &r, base + 0x4a8, cstring_vmaddr, got_vmaddr, &import_slots);
        try testing.expectEqual(@as(u32, 0xF9400210), std.mem.readInt(u32, &w, .little));
    }
}

test "CallTargetTooFar on i26-overflowing displacement" {
    const gpa = testing.allocator;
    // fn0 calls fn1; pad fn0 so the forward delta exceeds (1<<25)-1 words.
    // imm max = (1<<25)-1 = 33554431 words → 134217724 bytes. Make fn0 just over
    // 128 MiB so the delta overflows. Use addend to synthesize without huge alloc:
    // a single-word fn0 with a reloc whose addend pushes target past the limit.
    var fns = [_]FnCode{
        try makeFn(gpa, 0, 1, &.{.{
            .site = 0,
            .target = .{ .func = fname(1) },
            .kind = .call26,
            .addend = (@as(i64, 1) << 27) + 4, // >128 MiB byte delta
        }}),
        try makeFn(gpa, 1, 1, &.{}),
    };
    defer freeFns(gpa, &fns);

    try testing.expectError(error.CallTargetTooFar, linkTest(gpa, &fns, 0));
}

test "FnCode pack/unpack round-trips relocs, names, and literals" {
    const gpa = testing.allocator;
    // A FnCode with one func reloc, one cstr reloc, one import reloc, plus a
    // literal — exercises every target tag and the names/lits pools.
    const code = try gpa.dupe(u8, &[_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 });
    const relocs = try gpa.dupe(Reloc, &.{
        .{ .site = 0, .target = .{ .func = .{ .kind = .user_fn, .name = "add" } }, .kind = .call26 },
        .{ .site = 4, .target = .{ .cstr = 0xDEADBEEF }, .kind = .adrp_page, .addend = 3 },
        .{ .site = 4, .target = .{ .import = .{ .kind = .import, .name = "write" } }, .kind = .ldr_lo12 },
    });
    const lit_bytes = try gpa.dupe(u8, "hi");
    const literals = try gpa.dupe(Literal, &.{.{ .hash = 0x1234, .bytes = lit_bytes }});
    const fc = FnCode{ .sym = .{ .kind = .user_fn, .name = try gpa.dupe(u8, "main") }, .code = code, .relocs = relocs, .literals = literals };
    defer {
        // free the original's owned bits (names here are static literals except
        // sym.name and the duped slices) — use a manual free, not deinit, since
        // the reloc target names are static comptime strings.
        gpa.free(fc.sym.name);
        gpa.free(fc.code);
        gpa.free(fc.relocs);
        for (fc.literals) |l| gpa.free(l.bytes);
        gpa.free(fc.literals);
    }

    const blob = try pack(gpa, fc);
    defer gpa.free(blob);
    var got = (try unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer got.deinit(gpa);

    try testing.expect(got.sym.eql(fc.sym));
    try testing.expectEqualSlices(u8, fc.code, got.code);
    try testing.expectEqual(@as(usize, 3), got.relocs.len);
    try testing.expect(got.relocs[0].target.func.eql(.{ .kind = .user_fn, .name = "add" }));
    try testing.expectEqual(@as(u64, 0xDEADBEEF), got.relocs[1].target.cstr);
    try testing.expectEqual(@as(i64, 3), got.relocs[1].addend);
    try testing.expect(got.relocs[2].target.import.eql(.{ .kind = .import, .name = "write" }));
    try testing.expectEqual(@as(usize, 1), got.literals.len);
    try testing.expectEqual(@as(u64, 0x1234), got.literals[0].hash);
    try testing.expectEqualSlices(u8, "hi", got.literals[0].bytes);
}

test "unpack rejects a foreign or corrupt blob" {
    const gpa = testing.allocator;
    try testing.expect((try unpack(gpa, "not an fncode")) == null);
    try testing.expect((try unpack(gpa, &.{})) == null);
}
