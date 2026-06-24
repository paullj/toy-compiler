//! Lower a whole program (every function) to AArch64 machine code.
//!
//! WHY: this is the one place that turns a typed/resolved AST into bytes. M3
//! generalizes M1's single-`main` lowering to EVERY function in the file, each
//! lowered INDEPENDENTLY into a `Link.FnCode` artifact ({ sym, code, relocs }).
//! Calls name their target SYMBOLICALLY (by function index) via a `.call26`
//! relocation; a later serial `Link.link` pass lays the artifacts out in __text
//! and patches the `bl` placeholders. This per-function {code,relocs} seam is the
//! one M5 (incremental/parallel codegen) will reuse.
//!
//! `main` is the entry: dyld's LC_MAIN glue calls `exit(w0 & 0xFF)` on its return
//! value, so `main` takes no params and returns int (-> exit code) or unit `()`
//! (-> 0). Every other function is a real AAPCS64 function: the first 8 int/bool
//! args arrive in x0..x7 and args 9+ on the stack; the result is returned in x0.
//!
//! The expression evaluator is a constant-offset stack machine: `sp` is allocated
//! once in the prologue and NEVER moves in the body, so every store/load is a
//! constant offset and `sp` stays 16-byte aligned at every `bl`. The frame is:
//!
//!   [sp+0 .. out_base)            outgoing-args scratch (max stack-arg bytes)
//!   [sp+out_base .. +nslots*8)    locals (params spilled here + `:=` locals)
//!   [.. +max_temp_slots*8)             expression temps
//!   fp = sp+frame:                [fp]=saved x29, [fp+8]=saved x30
//!   [fp+16 + k*8]                 OUR incoming stack args (the (8+k)th param)
//!
//! frame = roundUp16(out_base + (nslots + max_temp_slots) * 8). The prologue normalizes
//! every parameter into its local slot (store x0..x7; copy stack args from
//! [x29,#16+k*8]) so the body reads all params uniformly sp-relative.
//!
//! Anything outside the supported subset (a boolean literal, a frame too large,
//! ...) yields exactly one `Diagnostic` at the offending node's source byte
//! offset; the driver turns any diagnostic into a clean failure + exit 1.

const std = @import("std");
const Token = @import("../ast/Token.zig").Token;
const TokenTag = @import("../ast/Token.zig").Tag;
const Ast = @import("../ast/Ast.zig");
const Resolve = @import("../resolve.zig");
const Resolution = @import("../symbols/Resolution.zig").Resolution;
const Typecheck = @import("../types.zig");
const Aarch64 = @import("Aarch64.zig");
const Link = @import("../link/Link.zig");

const Codegen = @This();

/// A construct we can't lower. `byte_offset` points at the offending node's main
/// token so the driver can render `line:col`.
pub const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;

/// Lowering mode (M5). `lower` itself is always pure (`normal`); the cache-or-run
/// wrapper in the Driver interprets the mode:
///   * `normal` — use a cache hit if present, else lower + store.
///   * `force`  — ignore any hit; always re-lower (cold-baseline / byte-identical
///                test).
///   * `verify` — on a hit, ALSO re-lower and assert the fresh pack == the cached
///                blob (the guard against a silent stale-cache miscompile). [C11]
pub const Mode = enum { normal, force, verify };

/// The whole-program lowering output for the SERIAL path (`--emit asm`): one
/// `Link.FnCode` per function plus the asm listing. The print body, string
/// interning, and `uses_write` are derived by the relink tail, not stored here.
/// Owned by the caller; free with `deinit`.
pub const ProgramResult = struct {
    /// One artifact per USER function, in source order. The print body (if any)
    /// is appended by the tail, not here.
    fns: []Link.FnCode,
    /// Human-readable assembly listing for `--emit asm`, or null if not requested.
    listing: ?[]u8,
    /// Lowering diagnostics (unsupported constructs), in discovery order.
    diags: []Diagnostic,
    /// Heap-allocated diagnostic messages; owned so they can be freed.
    owned_msgs: [][]u8,

    pub fn deinit(self: *ProgramResult, gpa: std.mem.Allocator) void {
        for (self.fns) |*f| f.deinit(gpa);
        gpa.free(self.fns);
        if (self.listing) |l| gpa.free(l);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        self.* = undefined;
    }
};

// ---- internal lowering state -----------------------------------------------
//
// ALL FIELDS ARE PER-FUNCTION (M5): `lower(fn)` builds a stack-local `Codegen`
// with only these, reads its frozen inputs, and reads/writes NO shared mutable
// state — so it is a pure, parallel-safe, cacheable query.

gpa: std.mem.Allocator,
tree: Ast.Tree,
tokens: []const Token,
source: []const u8,
resolutions: []const Resolution,
node_types: []const Typecheck.Type,
/// The struct table (M9): one layout per struct id, threaded read-only. Sized
/// from this so a struct local/temp gets exactly its layout size; the ABI walks
/// classify by the struct's byte size. Pure: read-only frozen input.
layouts: []const Typecheck.Layout,
/// Index → stable symbol name, for every user fn (source order) plus the
/// synthetic `print` builtin at index user_fn_count. Precomputed once and shared
/// read-only across the parallel jobs. Used to name a call's `.func` target.
names: []const Link.SymName,

/// Emitted words for the CURRENT function, serialized little-endian.
code: std.ArrayList(u8),
/// Relocations for the current function (patched at link time).
relocs: std.ArrayList(Link.Reloc),
/// This function's decoded string literals, content-hash keyed. The `.cstr`
/// reloc targets the hash; the serial relink tail interns these program-wide and
/// rewrites the targets to global `__cstring` offsets. Owned slices.
literals: std.ArrayList(Link.Literal),
/// Optional parallel assembly listing (one block per fn).
listing: ?std.ArrayList(u8),
diags: std.ArrayList(Diagnostic),
owned_msgs: std.ArrayList([]u8),

/// Outgoing-args region size in bytes (max stack-arg bytes of any call this fn
/// makes, roundUp16). All local/temp offsets are shifted up by this.
out_base: u32,
/// Number of local slots in the current function (params + `:=`). Slots are
/// type-sized: an int/bool slot is 8 bytes, a `str` slot is 16. `slot_byte_off`
/// maps each slot index to its sp-relative byte offset.
nslots: u32,
/// Per-slot sp-relative byte offset for the current function (length `nslots`).
/// Built by `slotSizesFn` before the prologue; replaces the old uniform slot*8.
slot_byte_off: []u32,
/// Total bytes the locals region occupies (sum of slot sizes), for `tempOff`.
locals_bytes: u32,
/// Current expression-temp depth; temp `d` lives at `[sp, #(out_base+locals_bytes+d*16)]`.
/// Every temp is 16 bytes so a `str` temp (ptr,len) fits; int temps under-use it.
depth: u32,
/// High-water mark of `depth`, used to size the frame.
max_temp_slots: u32,
/// The chosen 16-byte-aligned frame size, in bytes.
frame: u32,

/// M9: this fn's return type, and (for a large >16B struct return) the
/// sp-relative offset of the saved incoming x8 (the caller's result address).
/// `ret_type.kind == .invalid` means "not yet set"; `sret_off == NO_SRET` means
/// the fn does not return a large struct.
ret_type: Typecheck.Type = .{ .kind = .invalid },
sret_off: u32 = NO_SRET,

/// Per-function label table: `labels[id]` is the byte offset of the label in the
/// CURRENT function's `code` buffer, or `UNPLACED` until `placeLabel` sets it.
/// Reset (to `.empty`) per function; intra-function only — never a reloc/symbol.
labels: std.ArrayList(u32),
/// Per-function forward-branch fixups: each records a branch instruction's site,
/// the label it targets, and the immediate width to patch. Resolved by
/// `resolveFixups` at the end of `lowerFn`, after every label is placed.
fixups: std.ArrayList(Fixup),
/// Per-function loop-context stack (innermost last). Each break/continue targets
/// `loops.items[len-1]`. Pushed/popped around each loop body's `lowerBlock`.
loops: std.ArrayList(LoopCtx),

/// A control-flow label inside one function. `FALL` is a sentinel meaning "fall
/// through" (never a real label id), used only by `genCond`.
const LabelId = u32;
const FALL: LabelId = std.math.maxInt(LabelId);
/// A label whose code position is not yet known.
const UNPLACED: u32 = std.math.maxInt(u32);
/// A deferred branch: at `site` (a byte offset into `code`) a placeholder branch
/// word was emitted; once `label`'s position is known, patch its `width` imm
/// field to the PC-relative word delta.
const BranchWidth = enum { imm19, imm26 };
const Fixup = struct {
    site: u32,
    label: LabelId,
    width: BranchWidth,
};

/// The lowering context for one enclosing loop or labeled bare block. `break`
/// forward-branches to `break_label`; `continue` branches to `continue_label`
/// (the top for loop/while, the increment for `for`; UNPLACED for a labeled bare
/// block, which has no continue). For a value-yielding context (`loop` or a
/// labeled bare block), a value-break writes its result to `[sp, #result_off]`
/// (8 bytes, or 16 when `result_str`). (M8 generalizes M7's loop-context stack
/// into a label-addressable one.) `label` (borrowed `cg.source` — no free) and
/// `construct_node` let `break @L`/`continue @L` find the NAMED target.
const CtxKind = enum { loop, while_for, labeled_block };
const LoopCtx = struct {
    kind: CtxKind,
    label: ?[]const u8,
    construct_node: Ast.Index,
    break_label: LabelId,
    continue_label: LabelId,
    result_off: u32,
    result_str: bool,
    is_value: bool,
    /// When the context yields a STRUCT value, its type — a value-`break <struct>`
    /// copies the full bytes into `[sp, #result_off]` via genStructToSp (the
    /// reg-pair x0/x1 store only covers <=16B; a >16B break needs the byte copy).
    result_struct: ?Typecheck.Type = null,
};

/// The largest sp/fp scaled-offset the imm12 field can encode (4095 * 8 bytes).
const MAX_SCALED_OFFSET: u32 = 4095 * 8;

/// Sentinel: this fn does NOT return a large (>16B) struct (no x8 sret).
const NO_SRET: u32 = std.math.maxInt(u32);

/// PURE, MEMOIZABLE per-function lowering (the M5 query). The output is a single
/// `Link.FnCode` determined ENTIRELY by the frozen inputs (tree/tokens/source/
/// resolutions/node_types/names) plus the per-fn scalars (`fn_decl`, `sym`,
/// `is_entry`). It reads and writes NO shared mutable state, so it is both
/// parallel-safe and cacheable. Diagnostics for an unsupported construct are
/// surfaced via `out_diags` (the caller fails the build); on a diagnostic the
/// returned FnCode is still well-formed (possibly partial) and frees normally.
///
/// Caller owns the returned `FnCode`. `want_listing` appends a mnemonic block to
/// `out_listing` (the serial asm path only); pass null in the parallel path.
pub fn lower(
    gpa: std.mem.Allocator,
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolution,
    node_types: []const Typecheck.Type,
    layouts: []const Typecheck.Layout,
    names: []const Link.SymName,
    fn_decl: Ast.Index,
    sym: Link.SymName,
    is_entry: bool,
    out_diags: *std.ArrayList(Diagnostic),
    out_owned_msgs: *std.ArrayList([]u8),
    out_listing: ?*std.ArrayList(u8),
) error{OutOfMemory}!Link.FnCode {
    var cg: Codegen = .{
        .gpa = gpa,
        .tree = tree,
        .tokens = tokens,
        .source = source,
        .resolutions = resolutions,
        .node_types = node_types,
        .layouts = layouts,
        .names = names,
        .code = .empty,
        .relocs = .empty,
        .literals = .empty,
        .listing = null, // listing is emitted directly into out_listing below
        .diags = .empty,
        .owned_msgs = .empty,
        .out_base = 0,
        .nslots = 0,
        .slot_byte_off = &.{},
        .locals_bytes = 0,
        .depth = 0,
        .max_temp_slots = 0,
        .frame = 0,
        .labels = .empty,
        .fixups = .empty,
        .loops = .empty,
    };
    // The diag/listing sinks are the caller's (folded in fn source order); point
    // the per-fn state at them so existing helpers keep working unchanged.
    cg.diags = out_diags.*;
    cg.owned_msgs = out_owned_msgs.*;
    if (out_listing) |l| cg.listing = l.*;
    // On any failure, free only what THIS fn owns; the borrowed sinks are written
    // back to the caller below (or, on error, the values are still consistent —
    // we move them back before returning the error).
    errdefer {
        cg.code.deinit(gpa);
        cg.relocs.deinit(gpa);
        for (cg.literals.items) |lit| gpa.free(lit.bytes);
        cg.literals.deinit(gpa);
        if (cg.slot_byte_off.len > 0) gpa.free(cg.slot_byte_off);
        cg.labels.deinit(gpa);
        cg.fixups.deinit(gpa);
        cg.loops.deinit(gpa);
        out_diags.* = cg.diags;
        out_owned_msgs.* = cg.owned_msgs;
        if (out_listing) |l| l.* = cg.listing.?;
    }

    try cg.lowerFn(fn_decl, is_entry);

    if (cg.slot_byte_off.len > 0) gpa.free(cg.slot_byte_off);
    cg.slot_byte_off = &.{};

    const name_copy = try gpa.dupe(u8, sym.name);
    errdefer gpa.free(name_copy);

    const fc = Link.FnCode{
        .sym = .{ .kind = sym.kind, .name = name_copy },
        .code = try cg.code.toOwnedSlice(gpa),
        .relocs = try cg.relocs.toOwnedSlice(gpa),
        .literals = try cg.literals.toOwnedSlice(gpa),
    };

    // Hand the borrowed sinks back to the caller.
    out_diags.* = cg.diags;
    out_owned_msgs.* = cg.owned_msgs;
    if (out_listing) |l| l.* = cg.listing.?;
    return fc;
}

/// The hand-written `print(str)` intrinsic body, built once by the relink tail
/// when any reloc targets `(.builtin,"print")`. Content-free (no frozen inputs),
/// so no fingerprint/cache. `out_listing` (asm path only) gets the mnemonics.
/// Caller owns the result.
pub fn lowerPrint(gpa: std.mem.Allocator, out_listing: ?*std.ArrayList(u8)) error{OutOfMemory}!Link.FnCode {
    var cg: Codegen = .{
        .gpa = gpa,
        .tree = .{ .nodes = &.{}, .extra = &.{} },
        .tokens = &.{},
        .source = &.{},
        .resolutions = &.{},
        .node_types = &.{},
        .layouts = &.{},
        .names = &.{},
        .code = .empty,
        .relocs = .empty,
        .literals = .empty,
        .listing = if (out_listing) |l| l.* else null,
        .diags = .empty,
        .owned_msgs = .empty,
        .out_base = 0,
        .nslots = 0,
        .slot_byte_off = &.{},
        .locals_bytes = 0,
        .depth = 0,
        .max_temp_slots = 0,
        .frame = 0,
        .labels = .empty,
        .fixups = .empty,
        .loops = .empty,
    };
    errdefer {
        cg.code.deinit(gpa);
        cg.relocs.deinit(gpa);
        cg.loops.deinit(gpa);
        if (out_listing) |l| l.* = cg.listing.?;
    }
    try cg.emitPrintBody();
    const name = try gpa.dupe(u8, "print");
    errdefer gpa.free(name);
    const fc = Link.FnCode{
        .sym = .{ .kind = .builtin, .name = name },
        .code = try cg.code.toOwnedSlice(gpa),
        .relocs = try cg.relocs.toOwnedSlice(gpa),
        .literals = &.{},
    };
    if (out_listing) |l| l.* = cg.listing.?;
    return fc;
}

/// Thin SERIAL whole-program lowering for the `--emit asm` path (uncached: a
/// listing dump needs no fingerprint/cache). Lowers every user fn in source
/// order, then — if any fn references `print` — appends the print body. Caller
/// owns the returned `ProgramResult`. The string interning / `uses_write` /
/// data-reloc tail is the Driver's job (not needed for a listing).
pub fn generateProgram(
    gpa: std.mem.Allocator,
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolution,
    node_types: []const Typecheck.Type,
    layouts: []const Typecheck.Layout,
    names: []const Link.SymName,
    fn_nodes: []const Ast.Index,
    entry_fn: u32,
    want_listing: bool,
) !ProgramResult {
    var diags: std.ArrayList(Diagnostic) = .empty;
    var owned_msgs: std.ArrayList([]u8) = .empty;
    var listing: ?std.ArrayList(u8) = if (want_listing) .empty else null;
    errdefer {
        diags.deinit(gpa);
        for (owned_msgs.items) |m| gpa.free(m);
        owned_msgs.deinit(gpa);
        if (listing) |*l| l.deinit(gpa);
    }

    var fns: std.ArrayList(Link.FnCode) = .empty;
    errdefer {
        for (fns.items) |*f| f.deinit(gpa);
        fns.deinit(gpa);
    }

    var uses_print = false;
    for (fn_nodes, 0..) |fn_idx, sym| {
        var fc = try lower(
            gpa,
            tree,
            tokens,
            source,
            resolutions,
            node_types,
            layouts,
            names,
            fn_idx,
            names[sym],
            sym == entry_fn,
            &diags,
            &owned_msgs,
            if (listing) |*l| l else null,
        );
        errdefer fc.deinit(gpa);
        for (fc.relocs) |r| switch (r.target) {
            .func => |s| if (s.kind == .builtin and std.mem.eql(u8, s.name, "print")) {
                uses_print = true;
            },
            else => {},
        };
        try fns.append(gpa, fc);
    }

    if (uses_print) {
        var pf = try lowerPrint(gpa, if (listing) |*l| l else null);
        errdefer pf.deinit(gpa);
        try fns.append(gpa, pf);
    }

    return ProgramResult{
        .fns = try fns.toOwnedSlice(gpa),
        .listing = if (listing) |*l| try l.toOwnedSlice(gpa) else null,
        .diags = try diags.toOwnedSlice(gpa),
        .owned_msgs = try owned_msgs.toOwnedSlice(gpa),
    };
}

fn lowerFn(cg: *Codegen, fn_idx: Ast.Index, is_entry: bool) error{OutOfMemory}!void {
    const decl = cg.tree.nodes[fn_idx];
    const proto = Ast.protoAt(cg.tree, decl.lhs);
    const nparams: u32 = @intCast(proto.params.len);

    // Entry-only contract: main takes no params and returns int or unit `()`.
    if (is_entry) {
        if (nparams > 0) {
            try cg.unsupported(decl.main_token, "parameters on main unsupported in codegen");
            return;
        }
        if (proto.ret_type != Ast.none) {
            const rt = cg.tree.nodes[proto.ret_type];
            const is_unit = rt.tag == .literal_unit;
            if (!is_unit and !std.mem.eql(u8, cg.tokens[rt.main_token].text(cg.source), "int")) {
                try cg.unsupported(decl.main_token, "main must return int or () in codegen");
                return;
            }
        }
    }

    // Frame sizing (BEFORE the prologue, which allocates the frame). nslots is a
    // per-function count; out_base is the max stack-arg bytes of any call here;
    // max_temp_slots comes from a dry expression walk. Slots are TYPE-SIZED: a
    // `str` slot is 16 bytes, int/bool 8 — `slotSizesFn` builds the slot→byte
    // offset table (a prefix sum shifted by out_base) and `locals_bytes`.
    cg.nslots = cg.countSlotsFn(fn_idx, nparams);
    cg.out_base = cg.measureOutgoing(decl.rhs);
    cg.measureDepth(decl.rhs, 0);
    try cg.buildSlotTable(fn_idx, nparams);
    // Determine the return type (for the struct ABI). A large (>16B) struct
    // return uses x8 sret: reserve one 8-byte slot atop the locals to save the
    // incoming x8 (the body's calls clobber x8), then write the result through it.
    cg.ret_type = cg.fnReturnType(proto);
    cg.sret_off = NO_SRET;
    if (cg.ret_type.kind == .@"struct" and abiClass(cg.typeSize(cg.ret_type)) == .indirect) {
        cg.sret_off = cg.out_base + cg.locals_bytes;
        cg.locals_bytes += 8;
    }
    // Each temp is 16 bytes (a str temp must fit ptr,len); int temps under-use.
    cg.frame = roundUp16(cg.out_base + cg.locals_bytes + cg.max_temp_slots * 16);
    std.debug.assert(cg.frame % 16 == 0);
    std.debug.assert(cg.out_base % 16 == 0);

    // The prologue/epilogue adjust sp with a 12-bit immediate; reject an
    // over-budget frame with a clean diagnostic instead of overflowing the imm12.
    if (cg.frame > 4095) {
        try cg.unsupported(decl.main_token, "function frame too large for codegen (too many locals)");
        return;
    }
    // Any individual sp/fp scaled offset must fit the ldr/str imm12 range. The
    // highest local/temp offset is just below `frame` (guarded above), but a
    // function with > 8 params reads incoming stack args at [x29,#16+k*8] which is
    // independent of frame; guard it too.
    if (nparams > 8) {
        const top_in: u32 = 16 + (nparams - 8 - 1) * 8;
        if (top_in > MAX_SCALED_OFFSET) {
            try cg.unsupported(decl.main_token, "too many parameters for codegen (stack-arg offset out of range)");
            return;
        }
    }

    if (cg.listing) |*l| {
        const name = cg.tokens[decl.main_token].text(cg.source);
        try l.print(cg.gpa, "_{s}:\n", .{name});
    }

    // PROLOGUE: save FP/LR, set FP, open the frame.
    try cg.emit(Aarch64.stpFpLrPre, "stp x29, x30, [sp, #-16]!");
    try cg.emit(Aarch64.movFpSp, "mov x29, sp");
    if (cg.frame > 0) {
        try cg.emitFmt(Aarch64.subImm(Aarch64.SP, Aarch64.SP, @intCast(cg.frame)), "sub sp, sp, #{d}", .{cg.frame});
    }
    // A large-struct-returning fn: save the incoming indirect-result reg x8 (the
    // caller's buffer) — body calls clobber x8, so spill it once up front.
    if (cg.sret_off != NO_SRET) {
        try cg.emitFmt(Aarch64.strSp(8, cg.sret_off), "str x8, [sp, #{d}]", .{cg.sret_off});
    }

    // Normalize parameters into local slots (param i occupies slot i). AAPCS64 by
    // NGRN: a scalar takes 1 reg; a str/small-struct an eightbytes-wide reg run;
    // a large (>16B) struct arrives as a POINTER in one reg (copy the bytes in).
    // A param that doesn't fit the remaining regs goes WHOLLY on the incoming
    // stack at [x29,#16+nsaa] and exhausts NGRN.
    {
        var ngrn: u32 = 0;
        var nsaa: u32 = 0; // incoming-stack byte offset past [x29,#16]
        var slot: u32 = 0;
        while (slot < nparams) : (slot += 1) {
            const pty = cg.paramType(proto, slot);
            const off = cg.localOff(slot);
            const size = cg.typeSize(pty);
            const is_agg = (pty.kind == .str or pty.kind == .@"struct");
            if (is_agg and abiClass(size) == .indirect) {
                // A pointer to the caller's copy: dereference and copy bytes in.
                if (ngrn < 8) {
                    try cg.copyStructBytes(Aarch64.SP, off, ngrn, 0, size);
                    ngrn += 1;
                } else {
                    // The pointer is at [x29,#16+nsaa]; load it then copy bytes in.
                    const in_off = 16 + nsaa;
                    try cg.emitFmt(Aarch64.ldrFp(9, in_off), "ldr x9, [x29, #{d}]", .{in_off});
                    try cg.copyStructBytes(Aarch64.SP, off, 9, 0, size);
                    nsaa += 8;
                }
            } else if (is_agg) {
                const ebs = eightbytes(size);
                if (ngrn + ebs <= 8) {
                    var k: u32 = 0;
                    while (k < ebs) : (k += 1) {
                        try cg.emitFmt(Aarch64.strSp(ngrn + k, off + k * 8), "str x{d}, [sp, #{d}]", .{ ngrn + k, off + k * 8 });
                    }
                    ngrn += ebs;
                } else {
                    var k: u32 = 0;
                    while (k < ebs) : (k += 1) {
                        try cg.copyIncomingStack(nsaa + k * 8, off + k * 8);
                    }
                    nsaa += ebs * 8;
                    ngrn = 8;
                }
            } else {
                if (ngrn < 8) {
                    try cg.emitFmt(Aarch64.strSp(ngrn, off), "str x{d}, [sp, #{d}]", .{ ngrn, off });
                    ngrn += 1;
                } else {
                    try cg.copyIncomingStack(nsaa, off);
                    nsaa += 8;
                }
            }
        }
    }

    // BODY: walk statements in order. Each `return` emits its own inline epilogue
    // and lowering CONTINUES (so sibling if/else arms still get code). A non-unit
    // fn lands its body's trailing expression in x0 (M6 — no explicit return
    // needed); a unit fn discards any trailing expression (statement context).
    const ret_is_unit = (proto.ret_type == Ast.none) or (cg.tree.nodes[proto.ret_type].tag == .literal_unit);
    if (ret_is_unit) {
        try cg.lowerBlock(decl.rhs);
    } else if (cg.sret_off != NO_SRET) {
        // Large struct return: the body's trailing expr (M6) is written through the
        // saved incoming x8 (the caller's buffer).
        try cg.lowerBlockValueSret(decl.rhs);
    } else {
        const ret_str = std.mem.eql(u8, cg.tokens[cg.tree.nodes[proto.ret_type].main_token].text(cg.source), "str");
        try cg.lowerBlockValue(decl.rhs, ret_str);
    }

    // ONE trailing fall-through epilogue, ALWAYS emitted. For a UNIT fn it is the
    // reachable exit when the body falls off the end (a unit main returns 0 by
    // contract → force x0=0). For a NON-UNIT fn the body's trailing expression (M6)
    // or an `if`-value join lands the result in x0 right here, and this is the
    // reachable exit — so we must NOT clobber x0. (When every path instead ended in
    // an explicit `return`, definite-return makes this epilogue dead, so leaving x0
    // alone is harmless there too.) Only a UNIT entry forces x0=0.
    if (is_entry and ret_is_unit) {
        try cg.emit(Aarch64.movz(0, 0, 0), "movz x0, #0");
    }
    try cg.emitEpilogue();

    // All labels are now placed (every forward branch has a known target): patch
    // the deferred branch immediates, then release the per-function tables.
    try cg.resolveFixups();
    cg.labels.deinit(cg.gpa);
    cg.fixups.deinit(cg.gpa);
    cg.loops.deinit(cg.gpa);
    // Leave them deinit-safe: deinit set them to `undefined`, but generateProgram's
    // errdefer (and the next loop iteration) must not double-free if a later alloc
    // (e.g. fns.append / toOwnedSlice) fails before they are reset.
    cg.labels = .empty;
    cg.fixups = .empty;
    cg.loops = .empty;
}

// ---- intra-function labels & branch backpatch ------------------------------

/// Allocate a fresh, unplaced label id for the current function.
fn newLabel(cg: *Codegen) error{OutOfMemory}!LabelId {
    const id: LabelId = @intCast(cg.labels.items.len);
    try cg.labels.append(cg.gpa, UNPLACED);
    return id;
}

/// Bind `id` to the CURRENT end of the code buffer (the next instruction lands
/// here). Branches to `id` will be patched to point at this byte offset.
fn placeLabel(cg: *Codegen, id: LabelId) void {
    cg.labels.items[id] = @intCast(cg.code.items.len);
}

/// Select the context a `break`/`continue` node targets, mirroring Typecheck's
/// `targetCtx`: a labeled break/continue (resolved to `.label`) finds the NAMED
/// context by construct node; a bare one finds the innermost LOOP (skipping
/// labeled bare blocks). Typecheck guarantees a match exists, so it never returns
/// null on a well-typed program.
fn targetLoop(cg: *Codegen, stmt_idx: Ast.Index) *LoopCtx {
    const items = cg.loops.items;
    if (cg.resolutions[stmt_idx] == .label) {
        const target = cg.resolutions[stmt_idx].label;
        var i = items.len;
        while (i > 0) {
            i -= 1;
            if (items[i].construct_node == target) return &items[i];
        }
        unreachable; // typecheck-guaranteed
    }
    var i = items.len;
    while (i > 0) {
        i -= 1;
        if (items[i].kind != .labeled_block) return &items[i];
    }
    unreachable; // typecheck-guaranteed
}

/// Emit a branch placeholder `word` (its imm field zero) and record a fixup so
/// `resolveFixups` rewrites the imm once `id`'s position is known. `word` already
/// carries the opcode + identity (cond / rt); only the PC-relative imm is filled.
fn emitBranchToLabel(cg: *Codegen, word: u32, id: LabelId, width: BranchWidth, comptime mnemonic: []const u8) error{OutOfMemory}!void {
    try cg.fixups.append(cg.gpa, .{ .site = @intCast(cg.code.items.len), .label = id, .width = width });
    try cg.emit(word, mnemonic);
}

/// Patch every recorded fixup. The imm encodes `(target - site) / 4` words, two's
/// complement, PC-relative to the branch's OWN site — so it stays correct after
/// `Link` relocates the whole function (all branches move together). An offset
/// that overflows imm19/imm26 yields a clean diagnostic (never in practice: the
/// __TEXT page dwarfs ±1MB).
fn resolveFixups(cg: *Codegen) error{OutOfMemory}!void {
    for (cg.fixups.items) |fx| {
        const target = cg.labels.items[fx.label];
        std.debug.assert(target != UNPLACED);
        const delta_words: i64 = @divExact(@as(i64, target) - @as(i64, fx.site), 4);
        const old = std.mem.readInt(u32, cg.code.items[fx.site..][0..4], .little);
        const new = switch (fx.width) {
            .imm19 => blk: {
                if (delta_words < -(@as(i64, 1) << 18) or delta_words >= (@as(i64, 1) << 18)) {
                    try cg.diags.append(cg.gpa, .{ .byte_offset = 0, .message = "branch target out of range (imm19)" });
                    continue;
                }
                // 0x54xxxxxx is b.cond; 0xB4/0xB5 are cbz/cbnz. Both patch the same imm19 field.
                break :blk if ((old & 0xFF000000) == 0x54000000)
                    Aarch64.patchBCond(old, @intCast(delta_words))
                else
                    Aarch64.patchCbz(old, @intCast(delta_words));
            },
            .imm26 => blk: {
                if (delta_words < -(@as(i64, 1) << 25) or delta_words >= (@as(i64, 1) << 25)) {
                    try cg.diags.append(cg.gpa, .{ .byte_offset = 0, .message = "branch target out of range (imm26)" });
                    continue;
                }
                break :blk Aarch64.patchB(old, @intCast(delta_words));
            },
        };
        std.mem.writeInt(u32, cg.code.items[fx.site..][0..4], new, .little);
    }
}

/// Lower a block's statements in order. Control flow (if/while) and `return` are
/// no longer "first return ends the block": each `return` emits its OWN inline
/// epilogue and lowering CONTINUES to the next statement (so a `return` inside an
/// if-arm doesn't stop codegen of the else/rest). The single trailing fall-through
/// epilogue is emitted once by `lowerFn`; Typecheck's structural definite-return
/// analysis guarantees a non-unit fn can't actually fall off the end with garbage.
fn lowerBlock(cg: *Codegen, block_idx: Ast.Index) error{OutOfMemory}!void {
    const block = cg.tree.nodes[block_idx];
    for (Ast.rangeSlice(cg.tree, block.lhs)) |stmt_idx| {
        try cg.lowerStmt(stmt_idx);
    }
}

/// Lower ONE statement (the per-statement body extracted from `lowerBlock`, so
/// value-context block lowering can reuse it for non-final items). Byte-identical
/// to the old in-line switch.
fn lowerStmt(cg: *Codegen, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = cg.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .var_decl => {
                const slot = cg.localSlot(stmt_idx);
                switch (cg.node_types[stmt_idx].kind) {
                    .str => {
                        try cg.genStrExpr(stmt.lhs); // ptr→x0, len→x1
                        try cg.storeStrSlot(slot);
                    },
                    .@"struct" => {
                        // `q := <struct expr>` copies all bytes into the slot region.
                        try cg.genStructTo(stmt.lhs, Aarch64.SP, cg.localOff(slot));
                    },
                    else => {
                        try cg.genExpr(stmt.lhs);
                        try cg.emitFmt(Aarch64.strSp(0, cg.localOff(slot)), "str x0, [sp, #{d}]", .{cg.localOff(slot)});
                    },
                }
            },
            .assign => {
                const target = cg.tree.nodes[stmt.lhs];
                if (target.tag == .field_access) {
                    try cg.lowerFieldStore(stmt.lhs, stmt.rhs);
                } else switch (cg.node_types[stmt.lhs].kind) {
                    .str => {
                        try cg.genStrExpr(stmt.rhs);
                        try cg.storeStrSlot(cg.localSlot(stmt.lhs));
                    },
                    .@"struct" => {
                        try cg.genStructTo(stmt.rhs, Aarch64.SP, cg.localOff(cg.localSlot(stmt.lhs)));
                    },
                    else => {
                        try cg.genExpr(stmt.rhs);
                        const off = cg.localOff(cg.localSlot(stmt.lhs));
                        try cg.emitFmt(Aarch64.strSp(0, off), "str x0, [sp, #{d}]", .{off});
                    },
                }
            },
            .return_stmt => try cg.lowerReturn(stmt),
            .expr_stmt => try cg.genExpr(stmt.lhs), // evaluate for effect, discard
            .block => try cg.lowerBlock(stmt_idx),
            .if_stmt => try cg.lowerIf(stmt_idx),
            .while_stmt => try cg.lowerWhile(stmt_idx, null),
            .for_stmt => try cg.lowerFor(stmt_idx, null),
            // A labeled construct as a statement (the parser wraps it in an
            // expr_stmt; reached here only if directly present): discard its value.
            .labeled => try cg.lowerLabeledValue(stmt_idx, false),
            .break_stmt => {
                std.debug.assert(cg.loops.items.len > 0); // typecheck guaranteed in-loop
                const ctx = cg.targetLoop(stmt_idx).*;
                if (stmt.lhs != Ast.none and ctx.is_value) {
                    if (ctx.result_struct) |_| {
                        // A struct value-break: copy the full bytes to the result
                        // slot (covers >16B too, unlike the reg-pair store).
                        try cg.genStructToSp(stmt.lhs, ctx.result_off);
                    } else {
                        const two_word = ctx.result_str;
                        try cg.genValue(stmt.lhs, ctx.result_str);
                        try cg.emitFmt(Aarch64.strSp(0, ctx.result_off), "str x0, [sp, #{d}]", .{ctx.result_off});
                        if (two_word) try cg.emitFmt(Aarch64.strSp(1, ctx.result_off + 8), "str x1, [sp, #{d}]", .{ctx.result_off + 8});
                    }
                } else if (stmt.lhs != Ast.none) {
                    // A value expr on a while/for break: evaluate for effect, discard.
                    try cg.genExpr(stmt.lhs);
                }
                try cg.emitBranchToLabel(Aarch64.b(0), ctx.break_label, .imm26, "b Lbreak");
            },
            .continue_stmt => {
                std.debug.assert(cg.loops.items.len > 0);
                const ctx = cg.targetLoop(stmt_idx).*;
                try cg.emitBranchToLabel(Aarch64.b(0), ctx.continue_label, .imm26, "b Lcont");
            },
            else => try cg.unsupported(stmt.main_token, "statement unsupported in codegen"),
    }
}

/// Lower a `return [expr]`, then emit the inline epilogue. A unit return forces
/// x0=0; a scalar/str return leaves x0[,x1]; a small struct returns in (x0[,x1]);
/// a large struct is written through the saved incoming x8 (the caller's buffer).
fn lowerReturn(cg: *Codegen, stmt: Ast.Node) error{OutOfMemory}!void {
    if (stmt.lhs != Ast.none) {
        switch (cg.node_types[stmt.lhs].kind) {
            .str => try cg.genStrExpr(stmt.lhs),
            .@"struct" => {
                const size = cg.typeSize(cg.node_types[stmt.lhs]);
                if (abiClass(size) == .indirect) {
                    try cg.genStructToSret(stmt.lhs); // write through the saved caller buffer
                } else {
                    try cg.genStructExprPair(stmt.lhs); // small struct → (x0[,x1])
                }
            },
            else => try cg.genExpr(stmt.lhs), // result in x0
        }
    } else {
        try cg.emit(Aarch64.movz(0, 0, 0), "movz x0, #0");
    }
    try cg.emitEpilogue();
}

/// Lower a field place-store `p.x = v` / `p.a.b = v`. The place address is a
/// constant sp offset (local-rooted); store the rhs at the field's layout offset.
fn lowerFieldStore(cg: *Codegen, place_idx: Ast.Index, value_idx: Ast.Index) error{OutOfMemory}!void {
    const off = cg.placeByteOff(place_idx);
    switch (cg.node_types[place_idx].kind) {
        .str => {
            try cg.genStrExpr(value_idx);
            try cg.emitFmt(Aarch64.strSp(0, off), "str x0, [sp, #{d}]", .{off});
            try cg.emitFmt(Aarch64.strSp(1, off + 8), "str x1, [sp, #{d}]", .{off + 8});
        },
        .@"struct" => try cg.genStructTo(value_idx, Aarch64.SP, off),
        else => {
            try cg.genExpr(value_idx);
            try cg.emitFmt(Aarch64.strSp(0, off), "str x0, [sp, #{d}]", .{off});
        },
    }
}

/// Lower an `if`/`else if`/`else`. The condition is lowered in CONTROL context via
/// `genCond`: when there is no else, a false condition jumps past the then-block;
/// with an else, a false condition jumps to the else-block and the then-block ends
/// with an unconditional `b` over it. `else if` recurses (the else-node is a nested
/// `if_stmt`). All targets are intra-function labels backpatched by `resolveFixups`.
fn lowerIf(cg: *Codegen, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = cg.tree.nodes[stmt_idx];
    const h = Ast.ifHeaderAt(cg.tree, stmt.rhs);
    if (h.else_node == Ast.none) {
        // cond false → skip the then-block.
        const lend = try cg.newLabel();
        try cg.genCond(stmt.lhs, FALL, lend);
        try cg.lowerBlock(h.then_block);
        cg.placeLabel(lend);
    } else {
        // cond false → else; then-block falls through and `b`s over the else.
        const lelse = try cg.newLabel();
        const lend = try cg.newLabel();
        try cg.genCond(stmt.lhs, FALL, lelse);
        try cg.lowerBlock(h.then_block);
        try cg.emitBranchToLabel(Aarch64.b(0), lend, .imm26, "b Lend");
        cg.placeLabel(lelse);
        if (cg.tree.nodes[h.else_node].tag == .if_stmt) {
            try cg.lowerIf(h.else_node); // else if ...
        } else {
            try cg.lowerBlock(h.else_node);
        }
        cg.placeLabel(lend);
    }
}

/// Evaluate an expression, leaving its value in x0.
fn genExpr(cg: *Codegen, node_idx: Ast.Index) error{OutOfMemory}!void {
    if (node_idx == Ast.none) return;
    const n = cg.tree.nodes[node_idx];
    switch (n.tag) {
        .literal_number => {
            const v = cg.parseInt(n.main_token) orelse {
                try cg.unsupported(n.main_token, "integer literal out of range for codegen (i64)");
                return;
            };
            try cg.emitImm64(0, v);
        },
        .identifier => {
            switch (cg.node_types[node_idx].kind) {
                .str => try cg.genStrExpr(node_idx), // ptr→x0, len→x1
                .@"struct" => try cg.genStructExprPair(node_idx), // small struct → (x0[,x1])
                else => {
                    const slot = cg.localSlot(node_idx);
                    try cg.emitFmt(Aarch64.ldrSp(0, cg.localOff(slot)), "ldr x0, [sp, #{d}]", .{cg.localOff(slot)});
                },
            }
        },
        .unary => {
            const op = cg.tokens[n.main_token].tag;
            switch (op) {
                .minus => {
                    try cg.genExpr(n.lhs);
                    try cg.emit(Aarch64.neg(0, 0), "neg x0, x0");
                },
                .bang => {
                    // Logical NOT of a 0/1 bool: x0 = (x0 == 0).
                    try cg.genExpr(n.lhs);
                    try cg.emit(Aarch64.cmpImm(0, 0), "cmp x0, #0");
                    try cg.emit(Aarch64.cset(0, .eq), "cset x0, eq");
                },
                else => {
                    try cg.unsupported(n.main_token, "unary operator unsupported in codegen");
                    return;
                },
            }
        },
        .binary => try cg.genBinary(node_idx, n),
        .call => try cg.genCall(node_idx, n),
        .literal_bool => {
            // true → 1, false → 0 in x0.
            if (cg.tokens[n.main_token].tag == .kw_true) {
                try cg.emit(Aarch64.movz(0, 1, 0), "movz x0, #1");
            } else {
                try cg.emit(Aarch64.movz(0, 0, 0), "movz x0, #0");
            }
        },
        .literal_string => try cg.genStrExpr(node_idx), // ptr→x0, len→x1 (str value)
        .literal_unit => {}, // zero-sized: materializes to nothing, x0 untouched
        .struct_init => try cg.genStructExprPair(node_idx), // small struct → (x0[,x1])
        .field_access => try cg.genFieldAccessScalar(node_idx), // scalar/str field → x0[,x1]
        .block => try cg.lowerBlockValue(node_idx, false),
        .if_stmt => try cg.lowerIfValue(node_idx, false),
        .loop_expr => try cg.lowerLoopValue(node_idx, false, null),
        .labeled => try cg.lowerLabeledValue(node_idx, false),
        else => try cg.unsupported(n.main_token, "expression unsupported in codegen"),
    }
}

/// Lower a call: evaluate EVERY arg to a frame temp first (so a nested call can't
/// clobber an already-placed arg), then marshal temps into x0..x7 + the outgoing
/// region, then `bl` (a `.call26` reloc patched at link time). Result in x0.
fn genCall(cg: *Codegen, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!void {
    try cg.genCallInner(node_idx, n, null, 0);
}

/// A large-struct-returning call whose result must land at [dst_reg + dst_off]:
/// pass that address in x8 (the indirect-result register) before the `bl`.
fn genCallStructResult(cg: *Codegen, node_idx: Ast.Index, dst_reg: u32, dst_off: u32) error{OutOfMemory}!void {
    try cg.genCallInner(node_idx, cg.tree.nodes[node_idx], dst_reg, dst_off);
}

/// Shared call lowering. `sret_reg`/`sret_off` (non-null) name where a large
/// struct result must be written (passed in x8). Args spill to per-arg temps
/// (variable-sized: a struct arg spans `tempSlots`), then marshal per AAPCS64.
fn genCallInner(cg: *Codegen, node_idx: Ast.Index, n: Ast.Node, sret_reg: ?u32, sret_off: u32) error{OutOfMemory}!void {
    _ = node_idx;
    const callee_sym = cg.resolutions[n.lhs].func;
    const args = Ast.rangeSlice(cg.tree, n.rhs);

    // 1) Evaluate all args left-to-right into per-arg temps spanning tempSlots(ty).
    const base_depth = cg.depth;
    const temp_offs = try cg.gpa.alloc(u32, args.len);
    defer cg.gpa.free(temp_offs);
    for (args, 0..) |arg, i| {
        const ty = cg.node_types[arg];
        const off = cg.tempOff(cg.depth);
        temp_offs[i] = off;
        switch (ty.kind) {
            .str => {
                try cg.genStrExpr(arg); // ptr→x0, len→x1
                try cg.emitFmt(Aarch64.strSp(0, off), "str x0, [sp, #{d}]", .{off});
                try cg.emitFmt(Aarch64.strSp(1, off + 8), "str x1, [sp, #{d}]", .{off + 8});
            },
            .@"struct" => {
                // Advance past THIS arg's own temp span BEFORE materializing it, so a
                // nested call inside a field initializer (genStructInit evaluates field
                // values at cg.depth) spills past these bytes, not onto them.
                cg.depth += cg.tempSlots(ty);
                try cg.genStructTo(arg, Aarch64.SP, off); // full bytes into the temp span
                cg.depth -= cg.tempSlots(ty);
            },
            else => {
                try cg.genExpr(arg); // value in x0
                try cg.emitFmt(Aarch64.strSp(0, off), "str x0, [sp, #{d}]", .{off});
            },
        }
        cg.depth += cg.tempSlots(ty);
    }

    // 2) Marshal per AAPCS64 by NGRN. scalar → 1 reg / 8 nsaa; str/small-struct →
    //    eightbytes regs (ptr,len / 1-2 words) or whole to nsaa; large struct →
    //    pass a POINTER to its temp copy (1 reg / 8 nsaa).
    {
        var ngrn: u32 = 0;
        var nsaa: u32 = 0;
        for (args, 0..) |arg, i| {
            const off = temp_offs[i];
            const ty = cg.node_types[arg];
            const size = cg.typeSize(ty);
            const is_agg = (ty.kind == .str or ty.kind == .@"struct");
            if (is_agg and abiClass(size) == .indirect) {
                // Pass a pointer to the temp copy.
                if (ngrn < 8) {
                    try cg.emitFmt(Aarch64.addImm(ngrn, Aarch64.SP, @intCast(off)), "add x{d}, sp, #{d}", .{ ngrn, off });
                    ngrn += 1;
                } else {
                    try cg.emitFmt(Aarch64.addImm(9, Aarch64.SP, @intCast(off)), "add x9, sp, #{d}", .{off});
                    try cg.emitFmt(Aarch64.strSp(9, nsaa), "str x9, [sp, #{d}]", .{nsaa});
                    nsaa += 8;
                }
            } else if (is_agg) {
                const ebs = eightbytes(size);
                if (ngrn + ebs <= 8) {
                    var k: u32 = 0;
                    while (k < ebs) : (k += 1) {
                        try cg.emitFmt(Aarch64.ldrSp(ngrn + k, off + k * 8), "ldr x{d}, [sp, #{d}]", .{ ngrn + k, off + k * 8 });
                    }
                    ngrn += ebs;
                } else {
                    var k: u32 = 0;
                    while (k < ebs) : (k += 1) {
                        try cg.emitFmt(Aarch64.ldrSp(9, off + k * 8), "ldr x9, [sp, #{d}]", .{off + k * 8});
                        try cg.emitFmt(Aarch64.strSp(9, nsaa + k * 8), "str x9, [sp, #{d}]", .{nsaa + k * 8});
                    }
                    nsaa += ebs * 8;
                    ngrn = 8;
                }
            } else {
                if (ngrn < 8) {
                    try cg.emitFmt(Aarch64.ldrSp(ngrn, off), "ldr x{d}, [sp, #{d}]", .{ ngrn, off });
                    ngrn += 1;
                } else {
                    try cg.emitFmt(Aarch64.ldrSp(9, off), "ldr x9, [sp, #{d}]", .{off});
                    try cg.emitFmt(Aarch64.strSp(9, nsaa), "str x9, [sp, #{d}]", .{nsaa});
                    nsaa += 8;
                }
            }
        }
    }

    // 2b) Indirect-result: point x8 at the caller's result buffer. Emitted AFTER
    //     arg marshalling so the `add x8,...` (which may read sp) is not disturbed.
    if (sret_reg) |sr| {
        if (sr == Aarch64.SP) {
            try cg.emitFmt(Aarch64.addImm(8, Aarch64.SP, @intCast(sret_off)), "add x8, sp, #{d}", .{sret_off});
        } else {
            // dest address is in `sr` (a pointer reg): x8 = sr + off.
            if (sret_off == 0) {
                try cg.emitFmt(Aarch64.movReg(8, sr), "mov x8, x{d}", .{sr});
            } else {
                try cg.emitFmt(Aarch64.addImm(8, sr, @intCast(sret_off)), "add x8, x{d}, #{d}", .{ sr, sret_off });
            }
        }
    }

    // 3) Release the arg temps.
    cg.depth = base_depth;

    // 4) Emit the `bl` placeholder + reloc (callee named by its STABLE symbol).
    const target_name = cg.names[callee_sym];
    const name_copy = try cg.gpa.dupe(u8, target_name.name);
    errdefer cg.gpa.free(name_copy);
    const site: u32 = @intCast(cg.code.items.len);
    try cg.relocs.append(cg.gpa, .{
        .site = site,
        .target = .{ .func = .{ .kind = target_name.kind, .name = name_copy } },
        .kind = .call26,
        .addend = 0,
    });
    try cg.emitFmt(Aarch64.bl(0), "bl _{s}", .{target_name.name});

    // 5) Result is in x0 (scalar/small) or written through x8 (large struct).
}

// ---- str (fat {ptr,len}) values --------------------------------------------

/// Evaluate a `str`-typed expression, leaving ptr in x0 and len in x1.
///   * literal — intern the decoded bytes into `__cstring`, materialize the
///     address via adrp+add (placeholders, patched post-vmaddr) and the length
///     via a compile-time movImm64.
///   * identifier (str local) — two `ldr`s from the slot's 16 bytes.
fn genStrExpr(cg: *Codegen, node_idx: Ast.Index) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    switch (n.tag) {
        .literal_string => {
            const bytes = (try cg.decodeStringLiteral(n.main_token)) orelse return;
            // `addLiteral` TAKES OWNERSHIP of `bytes` (no defer free): the per-fn
            // literal sink owns it until the FnCode is freed. The `.cstr` reloc
            // targets the CONTENT HASH; the relink tail rewrites it to a global
            // `__cstring` offset. No program-wide shared interner → pure query.
            const h = std.hash.Wyhash.hash(lit_seed, bytes);
            try cg.addLiteral(h, bytes);
            const len: i64 = @intCast(bytes.len);

            // adrp x0, <cstr#hash>  (placeholder; .adrp_page reloc to .cstr=hash)
            var site: u32 = @intCast(cg.code.items.len);
            try cg.relocs.append(cg.gpa, .{ .site = site, .target = .{ .cstr = h }, .kind = .adrp_page });
            try cg.emit(Aarch64.adrp(0, 0), "adrp x0, <cstr>");
            // add x0, x0, <cstr#hash>  (placeholder; .add_lo12 reloc, same target)
            site = @intCast(cg.code.items.len);
            try cg.relocs.append(cg.gpa, .{ .site = site, .target = .{ .cstr = h }, .kind = .add_lo12 });
            try cg.emit(Aarch64.addImm(0, 0, 0), "add x0, x0, <cstr>");
            // mov x1, #len  (compile-time immediate; no reloc)
            try cg.emitImm64(1, len);
        },
        .identifier => {
            const slot = cg.localSlot(node_idx);
            const off = cg.localOff(slot);
            try cg.emitFmt(Aarch64.ldrSp(0, off), "ldr x0, [sp, #{d}]", .{off});
            try cg.emitFmt(Aarch64.ldrSp(1, off + 8), "ldr x1, [sp, #{d}]", .{off + 8});
        },
        .if_stmt => try cg.lowerIfValue(node_idx, true),
        .block => try cg.lowerBlockValue(node_idx, true),
        .loop_expr => try cg.lowerLoopValue(node_idx, true, null),
        .labeled => try cg.lowerLabeledValue(node_idx, true),
        // A str-typed field of a struct: load (ptr,len) from the field offset.
        .field_access => try cg.genFieldAccessScalar(node_idx),
        // A str-returning call leaves (ptr,len) in (x0,x1) per AAPCS64 — exactly the
        // str value convention — so no shuffle is needed.
        .call => try cg.genCall(node_idx, n),
        else => try cg.unsupported(n.main_token, "string expression unsupported in codegen"),
    }
}

// ---- struct (value aggregate) codegen (M9) ---------------------------------
//
// A struct VALUE is produced into a destination address `[base_reg + off]`.
// `base_reg` is SP (a frame slot/temp) or a scratch holding a pointer (x8 sret,
// or x9/x10 a place address). Copies move whole eightbytes via x10.

/// Copy `size` bytes (rounded up to 8) from [src_reg+src_off] to [dst_reg+dst_off]
/// via x10. The two base regs must differ from x10.
fn copyStructBytes(cg: *Codegen, dst_reg: u32, dst_off: u32, src_reg: u32, src_off: u32, size: u32) error{OutOfMemory}!void {
    var o: u32 = 0;
    while (o < size) : (o += 8) {
        try cg.emitFmt(Aarch64.ldrRegUoff(10, src_reg, src_off + o), "ldr x10, [x{d}, #{d}]", .{ src_reg, src_off + o });
        try cg.emitFmt(Aarch64.strRegUoff(10, dst_reg, dst_off + o), "str x10, [x{d}, #{d}]", .{ dst_reg, dst_off + o });
    }
}

/// Compute the address of a local-rooted struct PLACE (an identifier, or a field
/// path `p.a.b`) into `dst_reg`: `add dst_reg, sp, #(slot_off + summed offsets)`.
/// The receiver chain must bottom out at a local (Typecheck/Resolve guarantee).
fn genPlaceAddr(cg: *Codegen, node_idx: Ast.Index, dst_reg: u32) error{OutOfMemory}!void {
    const off = cg.placeByteOff(node_idx);
    try cg.emitFmt(Aarch64.addImm(dst_reg, Aarch64.SP, @intCast(off)), "add x{d}, sp, #{d}", .{ dst_reg, off });
}

/// The sp-relative byte offset of a local-rooted place (`p`, `p.x`, `p.a.b`).
fn placeByteOff(cg: *Codegen, node_idx: Ast.Index) u32 {
    const n = cg.tree.nodes[node_idx];
    switch (n.tag) {
        .identifier => return cg.localOff(cg.localSlot(node_idx)),
        .field_access => {
            const base_off = cg.placeByteOff(n.lhs);
            const base_ty = cg.node_types[n.lhs];
            const layout = cg.layouts[base_ty.struct_id];
            const fname = cg.tokens[n.main_token].text(cg.source);
            for (layout.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) return base_off + layout.offsets[j];
            }
            return base_off; // unreachable on a well-typed program
        },
        else => return 0,
    }
}

/// Whether a `field_access`/identifier place is rooted at a local (so its address
/// is a constant sp offset). A call/construction base is NOT local-rooted.
fn isLocalRootedPlace(cg: *Codegen, node_idx: Ast.Index) bool {
    const n = cg.tree.nodes[node_idx];
    return switch (n.tag) {
        .identifier => cg.resolutions[node_idx] == .local,
        .field_access => cg.isLocalRootedPlace(n.lhs),
        else => false,
    };
}

/// Produce the struct VALUE of `node_idx` into [dst_reg + dst_off] (full bytes).
/// `dst_reg` must not be x9 or x10 (used as scratch here).
///
/// CALLER-SAVED-DEST SAFETY: when `dst_reg` is NOT sp (it is a pointer held in a
/// caller-saved reg, e.g. the sret buffer x9, or a nested large-struct field of
/// such a buffer), producing the value may emit `bl`s (a struct-init field
/// initializer with a call, a struct-returning call, a value-control-flow arm)
/// that clobber the pointer reg. So materialize through an sp-relative FRAME TEMP
/// first (call-safe), then copy temp→[dst_reg+dst_off] in a tight no-`bl`
/// sequence. Only the sp-rooted path writes the destination directly.
fn genStructTo(cg: *Codegen, node_idx: Ast.Index, dst_reg: u32, dst_off: u32) error{OutOfMemory}!void {
    const ty = cg.node_types[node_idx];
    const size = cg.typeSize(ty);
    if (dst_reg != Aarch64.SP) {
        const temp = cg.tempOff(cg.depth);
        const saved = cg.depth;
        cg.depth += cg.tempSlots(ty);
        try cg.genStructTo(node_idx, Aarch64.SP, temp);
        cg.depth = saved;
        try cg.copyStructBytes(dst_reg, dst_off, Aarch64.SP, temp, size);
        return;
    }
    return cg.genStructToSp(node_idx, dst_off);
}

/// `genStructTo` for an sp-relative destination (the only path that writes the
/// destination directly — sp survives `bl`s). Splits out so the caller-saved-dest
/// path can re-enter here through a frame temp.
fn genStructToSp(cg: *Codegen, node_idx: Ast.Index, dst_off: u32) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    const ty = cg.node_types[node_idx];
    const size = cg.typeSize(ty);
    const dst_reg = Aarch64.SP;
    switch (n.tag) {
        .struct_init => try cg.genStructInit(node_idx, dst_reg, dst_off),
        // M9: a struct produced by a value-control-flow expression. Lower each arm's
        // value into the SAME sp-relative destination (recursing genStructToSp), so
        // a struct-valued if/block/loop/labeled works in EVERY sink (bind, assign,
        // pass, return) for both small AND large structs.
        .if_stmt => try cg.genStructIfTo(node_idx, dst_off),
        .block => try cg.genStructBlockTo(node_idx, dst_off),
        .loop_expr => try cg.genStructLoopTo(node_idx, dst_off, null),
        .labeled => try cg.genStructLabeledTo(node_idx, dst_off),
        .identifier, .field_access => {
            if (cg.isLocalRootedPlace(node_idx)) {
                const src_off = cg.placeByteOff(node_idx);
                try cg.copyStructBytes(dst_reg, dst_off, Aarch64.SP, src_off, size);
            } else {
                // Rvalue base (e.g. a returned struct's field): materialize the
                // receiver to a temp first, then copy the sub-field.
                try cg.genStructRvalueTo(node_idx, dst_reg, dst_off);
            }
        },
        .call => {
            if (abiClass(size) == .indirect) {
                // A large-struct call writes its result through x8; point x8 at the
                // destination so the result lands there directly.
                try cg.genCallStructResult(node_idx, dst_reg, dst_off);
            } else {
                // Small struct returned in (x0,x1); store both words to dest.
                try cg.genCall(node_idx, n);
                try cg.emitFmt(Aarch64.strRegUoff(0, dst_reg, dst_off), "str x0, [x{d}, #{d}]", .{ dst_reg, dst_off });
                if (size > 8) try cg.emitFmt(Aarch64.strRegUoff(1, dst_reg, dst_off + 8), "str x1, [x{d}, #{d}]", .{ dst_reg, dst_off + 8 });
            }
        },
        else => try cg.unsupported(n.main_token, "struct expression unsupported in codegen"),
    }
}

/// VALUE-context if/else producing a STRUCT into [sp + dst_off]. Mirrors
/// `lowerIfValue` but routes each arm's trailing value through genStructToSp
/// (the byte-copy producer) so both small AND large struct results work.
fn genStructIfTo(cg: *Codegen, node_idx: Ast.Index, dst_off: u32) error{OutOfMemory}!void {
    const stmt = cg.tree.nodes[node_idx];
    const h = Ast.ifHeaderAt(cg.tree, stmt.rhs);
    std.debug.assert(h.else_node != Ast.none);
    const lelse = try cg.newLabel();
    const lend = try cg.newLabel();
    try cg.genCond(stmt.lhs, FALL, lelse);
    try cg.genStructBlockTo(h.then_block, dst_off);
    try cg.emitBranchToLabel(Aarch64.b(0), lend, .imm26, "b Lend");
    cg.placeLabel(lelse);
    if (cg.tree.nodes[h.else_node].tag == .if_stmt)
        try cg.genStructIfTo(h.else_node, dst_off)
    else
        try cg.genStructBlockTo(h.else_node, dst_off);
    cg.placeLabel(lend);
}

/// VALUE-context block producing a STRUCT into [sp + dst_off]: non-final items run
/// for effect, the trailing expression is copied into the destination. A trailing
/// non-expr statement (e.g. a `return`) is lowered as a statement (no value).
fn genStructBlockTo(cg: *Codegen, block_idx: Ast.Index, dst_off: u32) error{OutOfMemory}!void {
    const stmts = Ast.rangeSlice(cg.tree, cg.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return;
    for (stmts[0 .. stmts.len - 1]) |s| try cg.lowerStmt(s);
    const last = stmts[stmts.len - 1];
    if (cg.tree.nodes[last].tag == .expr_stmt) {
        try cg.genStructToSp(cg.tree.nodes[last].lhs, dst_off);
    } else {
        try cg.lowerStmt(last);
    }
}

/// VALUE-context `loop` producing a STRUCT into [sp + dst_off]. The destination IS
/// the result slot a `break <struct>` writes (via genStructToSp); the body lowers
/// at the SAME depth (the destination is the caller's slot, not a fresh temp).
fn genStructLoopTo(cg: *Codegen, node_idx: Ast.Index, dst_off: u32, label: ?[]const u8) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    const top = try cg.newLabel();
    cg.placeLabel(top);
    const exit = try cg.newLabel();
    try cg.loops.append(cg.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .break_label = exit, .continue_label = top, .result_off = dst_off, .result_str = false, .is_value = true, .result_struct = cg.node_types[node_idx] });
    try cg.lowerBlock(n.lhs);
    _ = cg.loops.pop();
    try cg.emitBranchToLabel(Aarch64.b(0), top, .imm26, "b Ltop");
    cg.placeLabel(exit);
}

/// VALUE-context labeled construct producing a STRUCT into [sp + dst_off].
fn genStructLabeledTo(cg: *Codegen, node_idx: Ast.Index, dst_off: u32) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    const label = cg.tokens[n.main_token].text(cg.source);
    const inner = cg.tree.nodes[n.lhs];
    switch (inner.tag) {
        .loop_expr => try cg.genStructLoopTo(n.lhs, dst_off, label),
        .block => try cg.genStructLabeledBlockTo(n.lhs, dst_off, label),
        else => try cg.unsupported(n.main_token, "labeled struct construct unsupported in codegen"),
    }
}

/// VALUE-context labeled BARE BLOCK producing a STRUCT into [sp + dst_off]: both a
/// `break @L <struct>` and the trailing expression write the destination directly.
fn genStructLabeledBlockTo(cg: *Codegen, block_idx: Ast.Index, dst_off: u32, label: []const u8) error{OutOfMemory}!void {
    const exit = try cg.newLabel();
    try cg.loops.append(cg.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .break_label = exit, .continue_label = UNPLACED, .result_off = dst_off, .result_str = false, .is_value = true, .result_struct = cg.node_types[block_idx] });
    try cg.genStructBlockTo(block_idx, dst_off);
    _ = cg.loops.pop();
    cg.placeLabel(exit);
}

/// Field-of-rvalue (`area(p).x`-style) — currently only field_access whose base
/// is itself a non-local struct value. Materialize the base into a temp, then
/// copy the requested sub-range.
fn genStructRvalueTo(cg: *Codegen, node_idx: Ast.Index, dst_reg: u32, dst_off: u32) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    // Only field_access reaches here (identifier rvalue isn't possible).
    std.debug.assert(n.tag == .field_access);
    const base_ty = cg.node_types[n.lhs];
    const base_temp = cg.tempOff(cg.depth);
    const saved = cg.depth;
    cg.depth += cg.tempSlots(base_ty);
    try cg.genStructTo(n.lhs, Aarch64.SP, base_temp);
    cg.depth = saved;
    const layout = cg.layouts[base_ty.struct_id];
    const fname = cg.tokens[n.main_token].text(cg.source);
    var foff: u32 = 0;
    var fsize: u32 = 8;
    for (layout.field_names, 0..) |dn, j| {
        if (std.mem.eql(u8, dn, fname)) {
            foff = layout.offsets[j];
            fsize = cg.typeSize(layout.field_types[j]);
            break;
        }
    }
    try cg.copyStructBytes(dst_reg, dst_off, Aarch64.SP, base_temp + foff, fsize);
}

/// Write a `Name { field: value, ... }` construction into [dst_reg + dst_off].
fn genStructInit(cg: *Codegen, node_idx: Ast.Index, dst_reg: u32, dst_off: u32) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    const id = cg.structIdOf(node_idx);
    const layout = cg.layouts[id];
    for (Ast.rangeSlice(cg.tree, n.rhs)) |fi_idx| {
        const fi = cg.tree.nodes[fi_idx];
        const fname = cg.tokens[fi.main_token].text(cg.source);
        // Find the declared field offset/type.
        var foff: u32 = 0;
        var fty = Typecheck.Type.int;
        for (layout.field_names, 0..) |dn, j| {
            if (std.mem.eql(u8, dn, fname)) {
                foff = layout.offsets[j];
                fty = layout.field_types[j];
                break;
            }
        }
        switch (fty.kind) {
            .str => {
                try cg.genStrExpr(fi.lhs); // ptr→x0, len→x1
                try cg.emitFmt(Aarch64.strRegUoff(0, dst_reg, dst_off + foff), "str x0, [x{d}, #{d}]", .{ dst_reg, dst_off + foff });
                try cg.emitFmt(Aarch64.strRegUoff(1, dst_reg, dst_off + foff + 8), "str x1, [x{d}, #{d}]", .{ dst_reg, dst_off + foff + 8 });
            },
            .@"struct" => try cg.genStructTo(fi.lhs, dst_reg, dst_off + foff),
            else => {
                try cg.genExpr(fi.lhs); // scalar → x0
                try cg.emitFmt(Aarch64.strRegUoff(0, dst_reg, dst_off + foff), "str x0, [x{d}, #{d}]", .{ dst_reg, dst_off + foff });
            },
        }
    }
}

/// A struct expression in scalar/reg-pair value position (small struct ≤16B):
/// produce it into a temp, then load (x0[,x1]).
fn genStructExprPair(cg: *Codegen, node_idx: Ast.Index) error{OutOfMemory}!void {
    const ty = cg.node_types[node_idx];
    const size = cg.typeSize(ty);
    const temp = cg.tempOff(cg.depth);
    const saved = cg.depth;
    cg.depth += cg.tempSlots(ty);
    try cg.genStructTo(node_idx, Aarch64.SP, temp);
    cg.depth = saved;
    try cg.emitFmt(Aarch64.ldrSp(0, temp), "ldr x0, [sp, #{d}]", .{temp});
    if (size > 8) try cg.emitFmt(Aarch64.ldrSp(1, temp + 8), "ldr x1, [sp, #{d}]", .{temp + 8});
}

/// Produce a value into x0 (and x1 for a 2-word value). Routes a struct-typed
/// expression to the reg-pair producer; a str (or `want_str`) to genStrExpr; a
/// scalar to genExpr. Unifies the value sinks (return/break/block-value/loop).
fn genValue(cg: *Codegen, node_idx: Ast.Index, want_str: bool) error{OutOfMemory}!void {
    if (cg.node_types[node_idx].kind == .@"struct") {
        try cg.genStructExprPair(node_idx);
    } else if (want_str) {
        try cg.genStrExpr(node_idx);
    } else {
        try cg.genExpr(node_idx);
    }
}

/// Read a field access in value position. A scalar field → x0; a str field →
/// (x0,x1); a small-struct field → reg pair (via a temp). Local-rooted places
/// read directly; rvalue bases materialize first.
fn genFieldAccessScalar(cg: *Codegen, node_idx: Ast.Index) error{OutOfMemory}!void {
    const ty = cg.node_types[node_idx];
    if (ty.kind == .@"struct") {
        try cg.genStructExprPair(node_idx);
        return;
    }
    if (cg.isLocalRootedPlace(node_idx)) {
        const off = cg.placeByteOff(node_idx);
        try cg.emitFmt(Aarch64.ldrSp(0, off), "ldr x0, [sp, #{d}]", .{off});
        if (ty.kind == .str) try cg.emitFmt(Aarch64.ldrSp(1, off + 8), "ldr x1, [sp, #{d}]", .{off + 8});
        return;
    }
    // Rvalue base: materialize the receiver, then load the field.
    const n = cg.tree.nodes[node_idx];
    const base_ty = cg.node_types[n.lhs];
    const base_temp = cg.tempOff(cg.depth);
    const saved = cg.depth;
    cg.depth += cg.tempSlots(base_ty);
    try cg.genStructTo(n.lhs, Aarch64.SP, base_temp);
    cg.depth = saved;
    const layout = cg.layouts[base_ty.struct_id];
    const fname = cg.tokens[n.main_token].text(cg.source);
    var foff: u32 = 0;
    for (layout.field_names, 0..) |dn, j| {
        if (std.mem.eql(u8, dn, fname)) {
            foff = layout.offsets[j];
            break;
        }
    }
    try cg.emitFmt(Aarch64.ldrSp(0, base_temp + foff), "ldr x0, [sp, #{d}]", .{base_temp + foff});
    if (ty.kind == .str) try cg.emitFmt(Aarch64.ldrSp(1, base_temp + foff + 8), "ldr x1, [sp, #{d}]", .{base_temp + foff + 8});
}

/// VALUE-context block (M6): non-final items run for effect (via `lowerStmt`),
/// and the trailing expression (if any) is the block's value (x0, or x0/x1 for
/// str). A block with no trailing expression is `()` → materializes to nothing.
/// Pure: reads only frozen inputs + per-fn state.
fn lowerBlockValue(cg: *Codegen, block_idx: Ast.Index, want_str: bool) error{OutOfMemory}!void {
    const stmts = Ast.rangeSlice(cg.tree, cg.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return; // unit value: nothing
    for (stmts[0 .. stmts.len - 1]) |s| try cg.lowerStmt(s);
    const last = stmts[stmts.len - 1];
    if (cg.tree.nodes[last].tag == .expr_stmt) {
        try cg.genValue(cg.tree.nodes[last].lhs, want_str);
    } else {
        try cg.lowerStmt(last); // var_decl/assign/return/if-stmt/while → unit value
    }
}

/// VALUE-context block whose value is a LARGE struct, written through the saved
/// incoming x8 (the caller's result buffer). The trailing expression is produced
/// directly into [x8]. Non-final items run for effect. A trailing `return` (which
/// itself writes through x8) is lowered as a statement.
fn lowerBlockValueSret(cg: *Codegen, block_idx: Ast.Index) error{OutOfMemory}!void {
    const stmts = Ast.rangeSlice(cg.tree, cg.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return;
    for (stmts[0 .. stmts.len - 1]) |s| try cg.lowerStmt(s);
    const last = stmts[stmts.len - 1];
    const lt = cg.tree.nodes[last].tag;
    if (lt == .expr_stmt) {
        // The trailing expression is the fn's result: write it through the saved
        // caller buffer. genStructToSret produces into a frame temp first (so a
        // value-if/block/loop/large-call body is handled, and no `bl` clobbers the
        // buffer pointer) then copies into [x8].
        try cg.genStructToSret(cg.tree.nodes[last].lhs);
    } else if (lt == .if_stmt or lt == .block or lt == .loop_expr or lt == .labeled) {
        // A trailing value-producing control-flow expression (NOT wrapped in an
        // expr_stmt by the parser) is the fn's struct result — route it through
        // sret. Fixes the value-if-body sret hole (was lowered as a bare statement,
        // never writing x8).
        try cg.genStructToSret(last);
    } else {
        try cg.lowerStmt(last); // a trailing `return` writes through x8 itself
    }
}

/// Write a LARGE (>16B) struct value `node_idx` through the saved incoming x8 (the
/// caller's result buffer). Produce into an sp-relative FRAME TEMP first (call-safe;
/// the value production may emit `bl`s), THEN load the saved buffer pointer and copy
/// — the pointer load is adjacent to the copy with NO intervening `bl`, so a
/// caller-saved-reg clobber cannot corrupt it. Handles any value-producing
/// expression (struct-init, identifier/field copy, call, value-control-flow).
fn genStructToSret(cg: *Codegen, node_idx: Ast.Index) error{OutOfMemory}!void {
    const ty = cg.node_types[node_idx];
    const size = cg.typeSize(ty);
    const temp = cg.tempOff(cg.depth);
    const saved = cg.depth;
    cg.depth += cg.tempSlots(ty);
    try cg.genStructToSp(node_idx, temp);
    cg.depth = saved;
    try cg.emitFmt(Aarch64.ldrSp(9, cg.sret_off), "ldr x9, [sp, #{d}]", .{cg.sret_off});
    try cg.copyStructBytes(9, 0, Aarch64.SP, temp, size);
}

/// The function's declared return type (resolving a struct name to its layout).
fn fnReturnType(cg: *const Codegen, proto: Ast.FnProto) Typecheck.Type {
    if (proto.ret_type == Ast.none) return Typecheck.Type.unit;
    if (cg.tree.nodes[proto.ret_type].tag == .literal_unit) return Typecheck.Type.unit;
    const name = cg.tokens[cg.tree.nodes[proto.ret_type].main_token].text(cg.source);
    if (std.mem.eql(u8, name, "str")) return Typecheck.Type.str;
    if (std.mem.eql(u8, name, "bool")) return Typecheck.Type.@"bool";
    if (std.mem.eql(u8, name, "int")) return Typecheck.Type.int;
    for (cg.layouts, 0..) |l, id| {
        if (std.mem.eql(u8, l.name, name)) return Typecheck.Type.structT(@intCast(id));
    }
    return Typecheck.Type.int;
}

/// VALUE-context if/else (M6): BOTH arms write the ONE result location (x0, or
/// x0/x1 for str); control joins at `lend`. Reuses `genCond` exactly like
/// `lowerIf` — no parallel branch machine. `else` is guaranteed present
/// (typecheck enforces "value-if requires else"). A fully-divergent arm emits its
/// inline epilogue inside `lowerStmt` and never reaches the join; the live arm's
/// value reaches `lend`.
fn lowerIfValue(cg: *Codegen, node_idx: Ast.Index, want_str: bool) error{OutOfMemory}!void {
    const stmt = cg.tree.nodes[node_idx];
    const h = Ast.ifHeaderAt(cg.tree, stmt.rhs);
    std.debug.assert(h.else_node != Ast.none);
    const lelse = try cg.newLabel();
    const lend = try cg.newLabel();
    try cg.genCond(stmt.lhs, FALL, lelse);
    try cg.lowerBlockValue(h.then_block, want_str);
    try cg.emitBranchToLabel(Aarch64.b(0), lend, .imm26, "b Lend");
    cg.placeLabel(lelse);
    if (cg.tree.nodes[h.else_node].tag == .if_stmt)
        try cg.lowerIfValue(h.else_node, want_str) // else-if
    else
        try cg.lowerBlockValue(h.else_node, want_str);
    cg.placeLabel(lend);
}

/// VALUE-context `loop` (M7): an infinite loop yielding via `break <expr>`. The
/// single result location is a frame temp at `cg.depth`; the body lowers ONE
/// deeper. Each value-break (handled in `lowerStmt`) writes that slot and
/// forward-branches to `exit`; the back-edge re-enters `top`. After the loop the
/// result is loaded into x0 (and x1 for str). A break-less (`never`) loop never
/// reaches `exit`; the trailing load is dead but harmless. Mirrors the
/// frame-sizing in `measureExpr` exactly (result@depth, body@depth+1).
fn lowerLoopValue(cg: *Codegen, node_idx: Ast.Index, want_str: bool, label: ?[]const u8) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    const result_off = cg.tempOff(cg.depth);
    cg.depth += 1; // body lowers one deeper than the result slot
    const top = try cg.newLabel();
    cg.placeLabel(top);
    const exit = try cg.newLabel();
    try cg.loops.append(cg.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .break_label = exit, .continue_label = top, .result_off = result_off, .result_str = want_str, .is_value = true });
    try cg.lowerBlock(n.lhs);
    _ = cg.loops.pop();
    try cg.emitBranchToLabel(Aarch64.b(0), top, .imm26, "b Ltop"); // back-edge
    cg.placeLabel(exit);
    cg.depth -= 1;
    try cg.emitFmt(Aarch64.ldrSp(0, result_off), "ldr x0, [sp, #{d}]", .{result_off});
    if (want_str) try cg.emitFmt(Aarch64.ldrSp(1, result_off + 8), "ldr x1, [sp, #{d}]", .{result_off + 8});
}

/// `while cond { body }` (extracted from the inline lowerStmt arm so a labeled
/// while can thread its label). top: cond (genCond false → done; true falls into
/// body); body; b top; done:. A `()` statement.
fn lowerWhile(cg: *Codegen, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = cg.tree.nodes[stmt_idx];
    const top = try cg.newLabel();
    cg.placeLabel(top);
    const done = try cg.newLabel();
    try cg.genCond(stmt.lhs, FALL, done);
    try cg.loops.append(cg.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .break_label = done, .continue_label = top, .result_off = 0, .result_str = false, .is_value = false });
    try cg.lowerBlock(stmt.rhs);
    _ = cg.loops.pop();
    try cg.emitBranchToLabel(Aarch64.b(0), top, .imm26, "b Ltop");
    cg.placeLabel(done);
}

/// Lower a `labeled` wrapper (M8). Dispatches to the inner construct's lowering,
/// threading the label name. A labeled loop/block is value-yielding (`want_str`
/// flows through); a labeled while/for is a `()` statement (value discarded).
fn lowerLabeledValue(cg: *Codegen, node_idx: Ast.Index, want_str: bool) error{OutOfMemory}!void {
    const n = cg.tree.nodes[node_idx];
    const label = cg.tokens[n.main_token].text(cg.source);
    const inner = cg.tree.nodes[n.lhs];
    switch (inner.tag) {
        .loop_expr => try cg.lowerLoopValue(n.lhs, want_str, label),
        .block => try cg.lowerLabeledBlock(n.lhs, want_str, label),
        .while_stmt => try cg.lowerWhile(n.lhs, label),
        .for_stmt => try cg.lowerFor(n.lhs, label),
        else => try cg.unsupported(n.main_token, "labeled construct unsupported in codegen"),
    }
}

/// VALUE-context labeled BARE BLOCK (M8): like `lowerLoopValue` MINUS the
/// back-edge PLUS a trailing-value store. The result location is a frame temp at
/// `cg.depth`; the body lowers ONE deeper (the SIGBUS-critical depth that the four
/// frame-sizing walkers MUST mirror). `break @label <expr>` inside writes the slot
/// and forward-branches to `exit` (reusing the break body in `lowerStmt`). The
/// block's trailing expression also writes the slot, then falls into `exit`.
fn lowerLabeledBlock(cg: *Codegen, block_idx: Ast.Index, want_str: bool, label: []const u8) error{OutOfMemory}!void {
    const result_off = cg.tempOff(cg.depth);
    cg.depth += 1; // body lowers one deeper than the result slot
    const exit = try cg.newLabel();
    try cg.loops.append(cg.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .break_label = exit, .continue_label = UNPLACED, .result_off = result_off, .result_str = want_str, .is_value = true });
    // The trailing expression is the block's fall-through value; store it into the
    // SAME result slot the breaks target, then fall into `exit`.
    try cg.lowerBlockValueInto(block_idx, want_str, result_off);
    _ = cg.loops.pop();
    cg.placeLabel(exit);
    cg.depth -= 1;
    try cg.emitFmt(Aarch64.ldrSp(0, result_off), "ldr x0, [sp, #{d}]", .{result_off});
    if (want_str) try cg.emitFmt(Aarch64.ldrSp(1, result_off + 8), "ldr x1, [sp, #{d}]", .{result_off + 8});
}

/// Like `lowerBlockValue` but stores the trailing-expr value into the result slot
/// at `result_off` (instead of leaving it in x0/x1), so a labeled bare block's
/// fall-through value lands where the breaks also write. A block with no trailing
/// expression (a `()` value) leaves the slot untouched — only reached when the
/// block always diverges or its value is unit (never observed).
fn lowerBlockValueInto(cg: *Codegen, block_idx: Ast.Index, want_str: bool, result_off: u32) error{OutOfMemory}!void {
    const stmts = Ast.rangeSlice(cg.tree, cg.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return;
    for (stmts[0 .. stmts.len - 1]) |s| try cg.lowerStmt(s);
    const last = stmts[stmts.len - 1];
    if (cg.tree.nodes[last].tag == .expr_stmt) {
        const e = cg.tree.nodes[last].lhs;
        const two_word = want_str or cg.node_types[e].kind == .@"struct";
        try cg.genValue(e, want_str);
        try cg.emitFmt(Aarch64.strSp(0, result_off), "str x0, [sp, #{d}]", .{result_off});
        if (two_word) try cg.emitFmt(Aarch64.strSp(1, result_off + 8), "str x1, [sp, #{d}]", .{result_off + 8});
    } else {
        try cg.lowerStmt(last); // var_decl/assign/return/if-stmt/while → unit value
    }
}

/// `for i in lo..hi { body }` desugared to: `i := lo`; `top:` re-eval `hi` into a
/// temp, `cmp i, hi`, `b.ge exit`; body; `inc:` `i = i + 1`; `b top`; `exit:`.
/// Half-open `[lo, hi)`; `continue` targets `inc` so the increment always runs.
/// `i` is the `.local` bound on the `for_stmt` node (an int slot).
fn lowerFor(cg: *Codegen, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = cg.tree.nodes[stmt_idx];
    const h = Ast.forHeaderAt(cg.tree, stmt.rhs);
    const ioff = cg.localOff(cg.localSlot(stmt_idx));

    try cg.genExpr(h.lo); // i := lo
    try cg.emitFmt(Aarch64.strSp(0, ioff), "str x0, [sp, #{d}]", .{ioff});

    const top = try cg.newLabel();
    cg.placeLabel(top);
    const exit = try cg.newLabel();
    const inc = try cg.newLabel();

    // Re-evaluate `hi` each iteration into a temp at the base depth (so a
    // non-trivial `hi` expression is sized; see measureDepth's bumpTempSlots).
    const hi_off = cg.tempOff(cg.depth);
    cg.depth += 1;
    try cg.genExpr(h.hi);
    try cg.emitFmt(Aarch64.strSp(0, hi_off), "str x0, [sp, #{d}]", .{hi_off});
    try cg.emitFmt(Aarch64.ldrSp(0, hi_off), "ldr x0, [sp, #{d}]", .{hi_off}); // hi → x0
    cg.depth -= 1;
    try cg.emitFmt(Aarch64.ldrSp(1, ioff), "ldr x1, [sp, #{d}]", .{ioff}); // i → x1
    try cg.emit(Aarch64.cmpReg(1, 0), "cmp x1, x0"); // i - hi (signed)
    try cg.emitBranchToLabel(Aarch64.bCond(.ge, 0), exit, .imm19, "b.ge Lexit");

    try cg.loops.append(cg.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .break_label = exit, .continue_label = inc, .result_off = 0, .result_str = false, .is_value = false });
    try cg.lowerBlock(stmt.lhs);
    _ = cg.loops.pop();

    cg.placeLabel(inc); // i = i + 1
    try cg.emitFmt(Aarch64.ldrSp(0, ioff), "ldr x0, [sp, #{d}]", .{ioff});
    try cg.emit(Aarch64.addImm(0, 0, 1), "add x0, x0, #1");
    try cg.emitFmt(Aarch64.strSp(0, ioff), "str x0, [sp, #{d}]", .{ioff});
    try cg.emitBranchToLabel(Aarch64.b(0), top, .imm26, "b Ltop");
    cg.placeLabel(exit);
}

/// Store a str value (ptr in x0, len in x1) into local `slot`'s 16 bytes.
fn storeStrSlot(cg: *Codegen, slot: u32) !void {
    const off = cg.localOff(slot);
    try cg.emitFmt(Aarch64.strSp(0, off), "str x0, [sp, #{d}]", .{off});
    try cg.emitFmt(Aarch64.strSp(1, off + 8), "str x1, [sp, #{d}]", .{off + 8});
}

/// Seed for the literal content hash. Distinct from other Wyhash uses; the value
/// is part of the on-disk reloc target, so it must stay stable across versions.
const lit_seed: u64 = 0x10c5_7e87;

/// Record one decoded string literal in this function's literal sink, keyed by
/// content hash. TAKES OWNERSHIP of `bytes`. Dedups within the function so a
/// literal used twice contributes one entry (and one program-wide cstring after
/// the tail interns). On a within-fn dup the new copy is freed.
fn addLiteral(cg: *Codegen, hash: u64, bytes: []u8) error{OutOfMemory}!void {
    for (cg.literals.items) |lit| {
        if (lit.hash == hash and std.mem.eql(u8, lit.bytes, bytes)) {
            cg.gpa.free(bytes);
            return;
        }
    }
    cg.literals.append(cg.gpa, .{ .hash = hash, .bytes = bytes }) catch |e| {
        cg.gpa.free(bytes);
        return e;
    };
}

/// Emit the hand-written `print(str)` intrinsic body. It receives the str in
/// (x0=ptr, x1=len) and calls `write(fd=1, buf=ptr, len=len)` via the imported
/// `_write` (reached through its __got slot). No locals → no `sub sp`.
fn emitPrintBody(cg: *Codegen) error{OutOfMemory}!void {
    if (cg.listing) |*l| try l.print(cg.gpa, "_print:\n", .{});
    try cg.emit(Aarch64.stpFpLrPre, "stp x29, x30, [sp, #-16]!");
    try cg.emit(Aarch64.movFpSp, "mov x29, sp");
    // Shuffle (ptr,len) into write's (buf,len) = (x1,x2), then fd=1 in w0. Order
    // matters: move len (x1→x2) BEFORE overwriting x1 with ptr (x0→x1).
    try cg.emit(Aarch64.movReg(2, 1), "mov x2, x1"); // len → x2
    try cg.emit(Aarch64.movReg(1, 0), "mov x1, x0"); // ptr → x1
    try cg.emit(Aarch64.movz(0, 1, 0), "movz w0, #1"); // fd = 1
    // adrp x16, _write@GOT  (placeholder; .adrp_page reloc to import "write")
    var site: u32 = @intCast(cg.code.items.len);
    const wname1 = try cg.gpa.dupe(u8, "write");
    errdefer cg.gpa.free(wname1);
    try cg.relocs.append(cg.gpa, .{ .site = site, .target = .{ .import = .{ .kind = .import, .name = wname1 } }, .kind = .adrp_page });
    try cg.emit(Aarch64.adrp(16, 0), "adrp x16, _write@GOT");
    // ldr x16, [x16]  (placeholder; .ldr_lo12 reloc, same import)
    site = @intCast(cg.code.items.len);
    const wname2 = try cg.gpa.dupe(u8, "write");
    errdefer cg.gpa.free(wname2);
    try cg.relocs.append(cg.gpa, .{ .site = site, .target = .{ .import = .{ .kind = .import, .name = wname2 } }, .kind = .ldr_lo12 });
    try cg.emit(Aarch64.ldrRegUoff(16, 16, 0), "ldr x16, [x16]");
    try cg.emit(Aarch64.blr(16), "blr x16");
    try cg.emit(Aarch64.ldpFpLrPost, "ldp x29, x30, [sp], #16");
    try cg.emit(Aarch64.ret, "ret");
}

fn genBinary(cg: *Codegen, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!void {
    _ = node_idx;
    const op = cg.tokens[n.main_token].tag;
    switch (op) {
        .plus, .minus, .star, .slash => {
            // Arithmetic: evaluate lhs into x0, spill, evaluate rhs into x0, reload
            // lhs into x1, combine (operand order matters: lhs is minuend).
            try cg.genArithOperands(n); // → x1 = lhs, x0 = rhs
            switch (op) {
                .plus => try cg.emit(Aarch64.addReg(0, 1, 0), "add x0, x1, x0"),
                .minus => try cg.emit(Aarch64.subReg(0, 1, 0), "sub x0, x1, x0"),
                .star => try cg.emit(Aarch64.mul(0, 1, 0), "mul x0, x1, x0"),
                .slash => try cg.emit(Aarch64.sdiv(0, 1, 0), "sdiv x0, x1, x0"),
                else => unreachable,
            }
        },
        .lt, .lt_eq, .gt, .gt_eq, .eq_eq, .bang_eq => {
            // Comparison as a VALUE: cmp lhs,rhs then materialize 0/1 via cset.
            try cg.genCmpOperands(n);
            try cg.emit(Aarch64.cset(0, condFromToken(op)), "cset x0, <cc>");
        },
        .amp_amp => {
            // `a && b` as a VALUE: short-circuit to 0/1 in x0, reusing genCond's
            // branch machinery. If either operand tests false, jump to lfalse and
            // load 0; only if both fall through (true) do we load 1.
            const lfalse = try cg.newLabel();
            const lend = try cg.newLabel();
            try cg.genCond(n.lhs, FALL, lfalse);
            try cg.genCond(n.rhs, FALL, lfalse);
            try cg.emit(Aarch64.movz(0, 1, 0), "movz x0, #1");
            try cg.emitBranchToLabel(Aarch64.b(0), lend, .imm26, "b Lend");
            cg.placeLabel(lfalse);
            try cg.emit(Aarch64.movz(0, 0, 0), "movz x0, #0");
            cg.placeLabel(lend);
        },
        .pipe_pipe => {
            // `a || b` as a VALUE: if either tests true, jump to ltrue and load 1;
            // only if both fall through (false) do we load 0.
            const ltrue = try cg.newLabel();
            const lend = try cg.newLabel();
            try cg.genCond(n.lhs, ltrue, FALL);
            try cg.genCond(n.rhs, ltrue, FALL);
            try cg.emit(Aarch64.movz(0, 0, 0), "movz x0, #0");
            try cg.emitBranchToLabel(Aarch64.b(0), lend, .imm26, "b Lend");
            cg.placeLabel(ltrue);
            try cg.emit(Aarch64.movz(0, 1, 0), "movz x0, #1");
            cg.placeLabel(lend);
        },
        else => {
            try cg.unsupported(n.main_token, "binary operator unsupported in codegen");
            return;
        },
    }
}

/// Evaluate `n`'s two operands for an arithmetic op, leaving lhs in x1 and rhs in
/// x0 (via a frame temp spill so a nested expr can't clobber lhs).
fn genArithOperands(cg: *Codegen, n: Ast.Node) error{OutOfMemory}!void {
    try cg.genExpr(n.lhs);
    const spill_off = cg.tempOff(cg.depth);
    try cg.emitFmt(Aarch64.strSp(0, spill_off), "str x0, [sp, #{d}]", .{spill_off});
    cg.depth += 1;
    try cg.genExpr(n.rhs);
    cg.depth -= 1;
    try cg.emitFmt(Aarch64.ldrSp(1, spill_off), "ldr x1, [sp, #{d}]", .{spill_off});
}

/// Evaluate `n`'s two operands for a comparison and emit `cmp x1, x0` (lhs in x1,
/// rhs in x0) — the SUBS that sets NZCV for a following b.cond or cset. Same spill
/// discipline as `genArithOperands`, shared by both the VALUE (cset) and CONTROL
/// (b.cond) comparison paths so frame sizing stays identical.
fn genCmpOperands(cg: *Codegen, n: Ast.Node) error{OutOfMemory}!void {
    try cg.genArithOperands(n); // → x1 = lhs, x0 = rhs
    try cg.emit(Aarch64.cmpReg(1, 0), "cmp x1, x0");
}

/// Map a comparison token to its signed AArch64 condition. The condition is the
/// LOGICAL one (cset/bCond encode the inverse internally where needed).
fn condFromToken(tag: TokenTag) Aarch64.Cond {
    return switch (tag) {
        .lt => .lt,
        .lt_eq => .le,
        .gt => .gt,
        .gt_eq => .ge,
        .eq_eq => .eq,
        .bang_eq => .ne,
        else => unreachable,
    };
}

// ---- CONTROL-context bool lowering (destination-passing branches) ----------

/// Lower a bool expression in CONTROL context: instead of materializing 0/1, emit
/// branches so control reaches `true_dest` when the expression is true and
/// `false_dest` when false. EXACTLY ONE of the two dests is `FALL` (fall through
/// to the next instruction); the other is a real label. This is the shared core
/// used by if/while conditions (Stage D) and by &&/|| VALUE lowering.
///
///   * comparison  — `cmp` + ONE `b.<cc>` (or `b.<!cc>`), no cset.
///   * `!a`        — recurse on `a` with the two dests SWAPPED.
///   * `a && b`    — a-false jumps to false_dest; a-true falls into b.
///   * `a || b`    — a-true jumps to true_dest;  a-false falls into b.
///   * bare bool   — `cbnz`/`cbz` on the materialized 0/1.
fn genCond(cg: *Codegen, node_idx: Ast.Index, true_dest: LabelId, false_dest: LabelId) error{OutOfMemory}!void {
    std.debug.assert((true_dest == FALL) != (false_dest == FALL)); // exactly one FALL
    const n = cg.tree.nodes[node_idx];
    switch (n.tag) {
        .binary => {
            const op = cg.tokens[n.main_token].tag;
            switch (op) {
                .lt, .lt_eq, .gt, .gt_eq, .eq_eq, .bang_eq => {
                    try cg.genCmpOperands(n);
                    const cc = condFromToken(op);
                    if (false_dest == FALL) {
                        // Branch to true_dest when the comparison holds; else fall through.
                        try cg.emitBranchToLabel(Aarch64.bCond(cc, 0), true_dest, .imm19, "b.<cc> Ltrue");
                    } else {
                        // Branch to false_dest when it does NOT hold; else fall through (to true).
                        try cg.emitBranchToLabel(Aarch64.bCond(Aarch64.invert(cc), 0), false_dest, .imm19, "b.<!cc> Lfalse");
                    }
                },
                .amp_amp => {
                    // a false → false_dest; a true → fall into b.
                    const a_false = if (false_dest == FALL) try cg.newLabel() else false_dest;
                    try cg.genCond(n.lhs, FALL, a_false);
                    try cg.genCond(n.rhs, true_dest, false_dest);
                    if (false_dest == FALL) cg.placeLabel(a_false);
                },
                .pipe_pipe => {
                    // a true → true_dest; a false → fall into b.
                    const a_true = if (true_dest == FALL) try cg.newLabel() else true_dest;
                    try cg.genCond(n.lhs, a_true, FALL);
                    try cg.genCond(n.rhs, true_dest, false_dest);
                    if (true_dest == FALL) cg.placeLabel(a_true);
                },
                else => try cg.genCondBareBool(node_idx, true_dest, false_dest),
            }
        },
        .unary => {
            if (cg.tokens[n.main_token].tag == .bang) {
                try cg.genCond(n.lhs, false_dest, true_dest); // SWAP dests
            } else {
                try cg.genCondBareBool(node_idx, true_dest, false_dest);
            }
        },
        else => try cg.genCondBareBool(node_idx, true_dest, false_dest), // identifier/call/literal_bool
    }
}

/// Fallback for a bool value with no special control structure: materialize it to
/// x0 (0/1) then test it. `cbnz` jumps when nonzero (true); `cbz` when zero (false).
fn genCondBareBool(cg: *Codegen, node_idx: Ast.Index, true_dest: LabelId, false_dest: LabelId) error{OutOfMemory}!void {
    try cg.genExpr(node_idx); // x0 = 0/1
    if (false_dest == FALL) {
        try cg.emitBranchToLabel(Aarch64.cbnz(0, 0), true_dest, .imm19, "cbnz x0, Ltrue");
    } else {
        try cg.emitBranchToLabel(Aarch64.cbz(0, 0), false_dest, .imm19, "cbz x0, Lfalse");
    }
}

fn emitEpilogue(cg: *Codegen) !void {
    if (cg.frame > 0) {
        try cg.emitFmt(Aarch64.addImm(Aarch64.SP, Aarch64.SP, @intCast(cg.frame)), "add sp, sp, #{d}", .{cg.frame});
    }
    try cg.emit(Aarch64.ldpFpLrPost, "ldp x29, x30, [sp], #16");
    try cg.emit(Aarch64.ret, "ret");
}

// ---- offset helpers (single source of truth for the +out_base shift) -------

/// Byte offset (sp-relative) of local slot `slot`, from the type-sized table.
fn localOff(cg: *const Codegen, slot: u32) u32 {
    return cg.slot_byte_off[slot];
}

/// Byte offset (sp-relative) of expression temp `d`. Temps sit above the locals
/// region; each is 16 bytes so a `str` temp (ptr@d, len@d+8) fits. A struct temp
/// > 16 bytes spans `tempSlots(ty)` consecutive 16-byte slots.
fn tempOff(cg: *const Codegen, d: u32) u32 {
    return cg.out_base + cg.locals_bytes + d * 16;
}

// ---- struct ABI (M9) -------------------------------------------------------
//
// Reuse the str (16-byte, 2-register) precedent generalized to N eightbytes.
// A type's size/align come from the struct table (int/bool 8, str 16). Aggregate
// ABI class: <=16 bytes → register pair (1 or 2 x-regs); >16 → indirect (a
// pointer arg) + x8 sret return.

const AbiClass = enum { reg_pair, indirect };

/// Byte size of a type (int/bool 8, str 16, struct → its layout size).
fn typeSize(cg: *const Codegen, ty: Typecheck.Type) u32 {
    return switch (ty.kind) {
        .int, .bool => 8,
        .str => 16,
        .@"struct" => cg.layouts[ty.struct_id].size,
        else => 0,
    };
}

/// Natural alignment of a type (8 for scalars/str; a struct's max field align).
fn typeAlign(cg: *const Codegen, ty: Typecheck.Type) u32 {
    return switch (ty.kind) {
        .int, .bool, .str => 8,
        .@"struct" => cg.layouts[ty.struct_id].@"align",
        else => 1,
    };
}

/// AAPCS64 aggregate class: <=16 bytes passes in a register pair; >16 indirect.
fn abiClass(size: u32) AbiClass {
    return if (size <= 16) .reg_pair else .indirect;
}

/// How many 8-byte "eightbytes" a <=16-byte reg-pair value occupies (1 or 2).
fn eightbytes(size: u32) u32 {
    return (size + 7) / 8;
}

/// Number of 16-byte temp slots a value of this type occupies when spilled to a
/// frame temp. Scalars/str → 1; a struct rounds its size up to 16 then /16.
fn tempSlots(cg: *const Codegen, ty: Typecheck.Type) u32 {
    const sz = cg.typeSize(ty);
    if (sz <= 16) return 1;
    return roundUp16(sz) / 16;
}

/// The struct id of a struct-typed node (asserts it is a struct type).
fn structIdOf(cg: *const Codegen, node_idx: Ast.Index) u32 {
    return cg.node_types[node_idx].struct_id;
}

// ---- type-sized slot table -------------------------------------------------

/// The type held by local slot `slot`. A param's type comes from its declared
/// type-ref; a `:=`/`for` local's from the bound type recorded on its node by
/// Typecheck. Defaults to int (a plain scalar) when not found.
fn slotType(cg: *Codegen, fn_idx: Ast.Index, nparams: u32, slot: u32) Typecheck.Type {
    const decl = cg.tree.nodes[fn_idx];
    const proto = Ast.protoAt(cg.tree, decl.lhs);
    if (slot < nparams) return cg.paramType(proto, slot);
    var ty: Typecheck.Type = Typecheck.Type.int; // default scalar (8 bytes)
    cg.findLocalSlotType(decl.rhs, slot, &ty);
    return ty;
}

/// Resolve a parameter's declared type (handles a struct name → its layout).
fn paramType(cg: *const Codegen, proto: Ast.FnProto, slot: u32) Typecheck.Type {
    if (slot >= proto.params.len) return Typecheck.Type.int;
    const param = cg.tree.nodes[proto.params[slot]];
    if (param.lhs == Ast.none) return Typecheck.Type.int;
    const tok = cg.tree.nodes[param.lhs].main_token;
    const name = cg.tokens[tok].text(cg.source);
    if (std.mem.eql(u8, name, "str")) return Typecheck.Type.str;
    if (std.mem.eql(u8, name, "bool")) return Typecheck.Type.@"bool";
    // A struct param: find its layout by name (the struct table is order-stable).
    for (cg.layouts, 0..) |l, id| {
        if (std.mem.eql(u8, l.name, name)) return Typecheck.Type.structT(@intCast(id));
    }
    return Typecheck.Type.int;
}

fn findLocalSlotType(cg: *Codegen, block_idx: Ast.Index, slot: u32, out: *Typecheck.Type) void {
    const block = cg.tree.nodes[block_idx];
    for (Ast.rangeSlice(cg.tree, block.lhs)) |stmt_idx| {
        const stmt = cg.tree.nodes[stmt_idx];
        switch (stmt.tag) {
            .var_decl => {
                const res = cg.resolutions[stmt_idx];
                if (res == .local and res.local == slot) {
                    out.* = cg.node_types[stmt_idx];
                }
                // M6: the initializer may be a value-if/block declaring nested locals.
                cg.findLocalSlotTypeExpr(stmt.lhs, slot, out);
            },
            .assign => cg.findLocalSlotTypeExpr(stmt.rhs, slot, out),
            .return_stmt => if (stmt.lhs != Ast.none) cg.findLocalSlotTypeExpr(stmt.lhs, slot, out),
            .expr_stmt => cg.findLocalSlotTypeExpr(stmt.lhs, slot, out),
            .block => cg.findLocalSlotType(stmt_idx, slot, out),
            .if_stmt, .while_stmt => cg.findLocalSlotTypeStmt(stmt_idx, slot, out),
            .for_stmt => {
                // The `for` loop variable is an int slot.
                const res = cg.resolutions[stmt_idx];
                if (res == .local and res.local == slot) out.* = Typecheck.Type.int;
                const h = Ast.forHeaderAt(cg.tree, stmt.rhs);
                cg.findLocalSlotTypeExpr(h.lo, slot, out);
                cg.findLocalSlotTypeExpr(h.hi, slot, out);
                cg.findLocalSlotType(stmt.lhs, slot, out);
            },
            .break_stmt => if (stmt.lhs != Ast.none) cg.findLocalSlotTypeExpr(stmt.lhs, slot, out),
            .continue_stmt => {},
            .labeled => cg.findLocalSlotTypeExpr(stmt_idx, slot, out),
            else => {},
        }
    }
}

/// Descend an expression that may be a value-position if/block/loop (M6/M7) for
/// `findLocalSlotType`, so a local declared inside an arm is sized correctly.
fn findLocalSlotTypeExpr(cg: *Codegen, node_idx: Ast.Index, slot: u32, out: *Typecheck.Type) void {
    if (node_idx == Ast.none) return;
    switch (cg.tree.nodes[node_idx].tag) {
        .block => cg.findLocalSlotType(node_idx, slot, out),
        .if_stmt => cg.findLocalSlotTypeStmt(node_idx, slot, out),
        .loop_expr => cg.findLocalSlotType(cg.tree.nodes[node_idx].lhs, slot, out),
        .labeled => cg.findLocalSlotTypeLabeled(node_idx, slot, out),
        else => {},
    }
}

/// Descend a labeled wrapper's inner construct for `findLocalSlotType`.
fn findLocalSlotTypeLabeled(cg: *Codegen, node_idx: Ast.Index, slot: u32, out: *Typecheck.Type) void {
    const inner_idx = cg.tree.nodes[node_idx].lhs;
    switch (cg.tree.nodes[inner_idx].tag) {
        .block => cg.findLocalSlotType(inner_idx, slot, out),
        .loop_expr => cg.findLocalSlotType(cg.tree.nodes[inner_idx].lhs, slot, out),
        .while_stmt => cg.findLocalSlotType(cg.tree.nodes[inner_idx].rhs, slot, out),
        .for_stmt => {
            const res = cg.resolutions[inner_idx];
            if (res == .local and res.local == slot) out.* = Typecheck.Type.int;
            const h = Ast.forHeaderAt(cg.tree, cg.tree.nodes[inner_idx].rhs);
            cg.findLocalSlotTypeExpr(h.lo, slot, out);
            cg.findLocalSlotTypeExpr(h.hi, slot, out);
            cg.findLocalSlotType(cg.tree.nodes[inner_idx].lhs, slot, out);
        },
        else => {},
    }
}

/// Recurse a control-flow statement (if/while) for `findLocalSlotType`, descending
/// into then/else/body blocks. Handles `else if` (else-node is a nested if_stmt).
fn findLocalSlotTypeStmt(cg: *Codegen, stmt_idx: Ast.Index, slot: u32, out: *Typecheck.Type) void {
    const stmt = cg.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .if_stmt => {
            const h = Ast.ifHeaderAt(cg.tree, stmt.rhs);
            cg.findLocalSlotType(h.then_block, slot, out);
            if (h.else_node != Ast.none) {
                if (cg.tree.nodes[h.else_node].tag == .if_stmt) {
                    cg.findLocalSlotTypeStmt(h.else_node, slot, out);
                } else {
                    cg.findLocalSlotType(h.else_node, slot, out);
                }
            }
        },
        .while_stmt => cg.findLocalSlotType(stmt.rhs, slot, out),
        else => {},
    }
}

/// Build `slot_byte_off` (a prefix sum of type-sized slots, shifted by out_base)
/// and `locals_bytes`. Must run AFTER nslots/out_base are known.
fn buildSlotTable(cg: *Codegen, fn_idx: Ast.Index, nparams: u32) error{OutOfMemory}!void {
    const table = try cg.gpa.alloc(u32, cg.nslots);
    var running: u32 = 0;
    var slot: u32 = 0;
    while (slot < cg.nslots) : (slot += 1) {
        const ty = cg.slotType(fn_idx, nparams, slot);
        // A struct local occupies its layout size (rounded to 8 so following
        // slots stay 8-aligned; struct align is <=8 today). Scalars 8, str 16.
        const sz: u32 = switch (ty.kind) {
            .str => 16,
            .@"struct" => roundUp8(cg.typeSize(ty)),
            else => 8,
        };
        running = roundUp8(running);
        table[slot] = cg.out_base + running;
        running += sz;
    }
    cg.slot_byte_off = table;
    cg.locals_bytes = running;
}

fn roundUp8(n: u32) u32 {
    return (n + 7) / 8 * 8;
}

/// Copy one 8-byte incoming stack word at [x29,#16+nsaa] into the local at
/// sp-relative `off` (via x9).
fn copyIncomingStack(cg: *Codegen, nsaa: u32, off: u32) !void {
    const in_off = 16 + nsaa;
    try cg.emitFmt(Aarch64.ldrFp(9, in_off), "ldr x9, [x29, #{d}]", .{in_off});
    try cg.emitFmt(Aarch64.strSp(9, off), "str x9, [sp, #{d}]", .{off});
}

// ---- frame sizing (read-only AST walks, no emission) -----------------------

/// Per-function local slot count: params occupy slots 0..nparams-1, and `:=`
/// locals carry their `.local` resolution. The count is max(slot) + 1.
fn countSlotsFn(cg: *Codegen, fn_idx: Ast.Index, nparams: u32) u32 {
    const decl = cg.tree.nodes[fn_idx];
    var max_slot: i64 = @as(i64, nparams) - 1; // params always occupy 0..nparams-1
    cg.collectSlots(decl.rhs, &max_slot);
    return if (max_slot < 0) 0 else @intCast(max_slot + 1);
}

/// Walk a block collecting the highest `.local` slot of any var_decl or identifier
/// reachable from it. Mirrors lowerBlock's statement set.
fn collectSlots(cg: *Codegen, block_idx: Ast.Index, max_slot: *i64) void {
    const block = cg.tree.nodes[block_idx];
    for (Ast.rangeSlice(cg.tree, block.lhs)) |stmt_idx| {
        const stmt = cg.tree.nodes[stmt_idx];
        switch (stmt.tag) {
            .var_decl => {
                cg.noteSlot(stmt_idx, max_slot);
                cg.collectSlotsExpr(stmt.lhs, max_slot);
            },
            .assign => {
                cg.noteSlot(stmt.lhs, max_slot);
                cg.collectSlotsExpr(stmt.rhs, max_slot);
            },
            .return_stmt => if (stmt.lhs != Ast.none) cg.collectSlotsExpr(stmt.lhs, max_slot),
            .expr_stmt => cg.collectSlotsExpr(stmt.lhs, max_slot),
            .block => cg.collectSlots(stmt_idx, max_slot),
            .if_stmt, .while_stmt => cg.collectSlotsStmt(stmt_idx, max_slot),
            .for_stmt => {
                cg.noteSlot(stmt_idx, max_slot); // the loop variable's slot
                const h = Ast.forHeaderAt(cg.tree, stmt.rhs);
                cg.collectSlotsExpr(h.lo, max_slot);
                cg.collectSlotsExpr(h.hi, max_slot);
                cg.collectSlots(stmt.lhs, max_slot);
            },
            .break_stmt => if (stmt.lhs != Ast.none) cg.collectSlotsExpr(stmt.lhs, max_slot),
            .continue_stmt => {},
            .labeled => cg.collectSlotsExpr(stmt_idx, max_slot),
            else => {},
        }
    }
}

/// Recurse a control-flow statement (if/while) for `collectSlots`: cond locals +
/// then/else/body locals. Handles `else if` (else-node is a nested if_stmt).
fn collectSlotsStmt(cg: *Codegen, stmt_idx: Ast.Index, max_slot: *i64) void {
    const stmt = cg.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .if_stmt => {
            cg.collectSlotsExpr(stmt.lhs, max_slot);
            const h = Ast.ifHeaderAt(cg.tree, stmt.rhs);
            cg.collectSlots(h.then_block, max_slot);
            if (h.else_node != Ast.none) {
                if (cg.tree.nodes[h.else_node].tag == .if_stmt) {
                    cg.collectSlotsStmt(h.else_node, max_slot);
                } else {
                    cg.collectSlots(h.else_node, max_slot);
                }
            }
        },
        .while_stmt => {
            cg.collectSlotsExpr(stmt.lhs, max_slot);
            cg.collectSlots(stmt.rhs, max_slot);
        },
        else => {},
    }
}

fn collectSlotsExpr(cg: *Codegen, node_idx: Ast.Index, max_slot: *i64) void {
    if (node_idx == Ast.none) return;
    const n = cg.tree.nodes[node_idx];
    switch (n.tag) {
        .identifier => cg.noteSlot(node_idx, max_slot),
        .unary => cg.collectSlotsExpr(n.lhs, max_slot),
        .binary => {
            cg.collectSlotsExpr(n.lhs, max_slot);
            cg.collectSlotsExpr(n.rhs, max_slot);
        },
        .call => for (Ast.rangeSlice(cg.tree, n.rhs)) |arg| cg.collectSlotsExpr(arg, max_slot),
        // M6: an if/block in expression position (var_decl/return/arg) declares its
        // own locals; descend exactly where genExpr lowers them.
        .if_stmt => cg.collectSlotsStmt(node_idx, max_slot),
        .block => cg.collectSlots(node_idx, max_slot),
        .loop_expr => cg.collectSlots(n.lhs, max_slot),
        .labeled => cg.collectSlotsLabeled(node_idx, max_slot),
        .literal_unit => {},
        // M9: a struct literal's field values and a field access's receiver may
        // reference locals (e.g. punning `P { x }` → identifier `x`).
        .struct_init => for (Ast.rangeSlice(cg.tree, n.rhs)) |fi| cg.collectSlotsExpr(cg.tree.nodes[fi].lhs, max_slot),
        .field_access => cg.collectSlotsExpr(n.lhs, max_slot),
        else => {},
    }
}

/// Collect slots under a labeled wrapper's inner construct (depth-independent).
fn collectSlotsLabeled(cg: *Codegen, node_idx: Ast.Index, max_slot: *i64) void {
    const inner_idx = cg.tree.nodes[node_idx].lhs;
    const inner = cg.tree.nodes[inner_idx];
    switch (inner.tag) {
        .block => cg.collectSlots(inner_idx, max_slot),
        .loop_expr => cg.collectSlots(inner.lhs, max_slot),
        .while_stmt => cg.collectSlotsStmt(inner_idx, max_slot),
        .for_stmt => {
            cg.noteSlot(inner_idx, max_slot); // the loop variable's slot
            const h = Ast.forHeaderAt(cg.tree, inner.rhs);
            cg.collectSlotsExpr(h.lo, max_slot);
            cg.collectSlotsExpr(h.hi, max_slot);
            cg.collectSlots(inner.lhs, max_slot);
        },
        else => {},
    }
}

fn noteSlot(cg: *Codegen, node_idx: Ast.Index, max_slot: *i64) void {
    const res = cg.resolutions[node_idx];
    if (res == .local) {
        const s: i64 = res.local;
        if (s > max_slot.*) max_slot.* = s;
    }
}

/// Max outgoing-args bytes of any call in the function (args 9+ go on the stack),
/// rounded up to 16. 0 if no call passes more than 8 args.
fn measureOutgoing(cg: *Codegen, block_idx: Ast.Index) u32 {
    var max_bytes: u32 = 0;
    cg.measureOutgoingBlock(block_idx, &max_bytes);
    return roundUp16(max_bytes);
}

fn measureOutgoingBlock(cg: *Codegen, block_idx: Ast.Index, max_bytes: *u32) void {
    const block = cg.tree.nodes[block_idx];
    for (Ast.rangeSlice(cg.tree, block.lhs)) |stmt_idx| {
        const stmt = cg.tree.nodes[stmt_idx];
        switch (stmt.tag) {
            .var_decl => cg.measureOutgoingExpr(stmt.lhs, max_bytes),
            .assign => cg.measureOutgoingExpr(stmt.rhs, max_bytes),
            .return_stmt => if (stmt.lhs != Ast.none) cg.measureOutgoingExpr(stmt.lhs, max_bytes),
            .expr_stmt => cg.measureOutgoingExpr(stmt.lhs, max_bytes),
            .block => cg.measureOutgoingBlock(stmt_idx, max_bytes),
            .if_stmt, .while_stmt => cg.measureOutgoingStmt(stmt_idx, max_bytes),
            .for_stmt => {
                const h = Ast.forHeaderAt(cg.tree, stmt.rhs);
                cg.measureOutgoingExpr(h.lo, max_bytes);
                cg.measureOutgoingExpr(h.hi, max_bytes);
                cg.measureOutgoingBlock(stmt.lhs, max_bytes);
            },
            .break_stmt => if (stmt.lhs != Ast.none) cg.measureOutgoingExpr(stmt.lhs, max_bytes),
            .continue_stmt => {},
            .labeled => cg.measureOutgoingExpr(stmt_idx, max_bytes),
            else => {},
        }
    }
}

/// Recurse a control-flow statement (if/while) for `measureOutgoing`: a call in a
/// cond or in a then/else/body arm contributes its outgoing-args bytes too.
fn measureOutgoingStmt(cg: *Codegen, stmt_idx: Ast.Index, max_bytes: *u32) void {
    const stmt = cg.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .if_stmt => {
            cg.measureOutgoingExpr(stmt.lhs, max_bytes);
            const h = Ast.ifHeaderAt(cg.tree, stmt.rhs);
            cg.measureOutgoingBlock(h.then_block, max_bytes);
            if (h.else_node != Ast.none) {
                if (cg.tree.nodes[h.else_node].tag == .if_stmt) {
                    cg.measureOutgoingStmt(h.else_node, max_bytes);
                } else {
                    cg.measureOutgoingBlock(h.else_node, max_bytes);
                }
            }
        },
        .while_stmt => {
            cg.measureOutgoingExpr(stmt.lhs, max_bytes);
            cg.measureOutgoingBlock(stmt.rhs, max_bytes);
        },
        else => {},
    }
}

fn measureOutgoingExpr(cg: *Codegen, node_idx: Ast.Index, max_bytes: *u32) void {
    if (node_idx == Ast.none) return;
    const n = cg.tree.nodes[node_idx];
    switch (n.tag) {
        .unary => cg.measureOutgoingExpr(n.lhs, max_bytes),
        .binary => {
            cg.measureOutgoingExpr(n.lhs, max_bytes);
            cg.measureOutgoingExpr(n.rhs, max_bytes);
        },
        .call => {
            const args = Ast.rangeSlice(cg.tree, n.rhs);
            // Same NGRN/nsaa walk as genCallInner's marshal phase: scalar → 1 reg
            // or 8 nsaa; str/small-struct → eightbytes regs or nsaa; large struct →
            // a POINTER (1 reg or 8 nsaa).
            var ngrn: u32 = 0;
            var nsaa: u32 = 0;
            for (args) |arg| {
                const ty = cg.node_types[arg];
                const size = cg.typeSize(ty);
                const is_agg = (ty.kind == .str or ty.kind == .@"struct");
                if (is_agg and abiClass(size) == .indirect) {
                    if (ngrn < 8) ngrn += 1 else nsaa += 8;
                } else if (is_agg) {
                    const ebs = eightbytes(size);
                    if (ngrn + ebs <= 8) ngrn += ebs else {
                        nsaa += ebs * 8;
                        ngrn = 8;
                    }
                } else {
                    if (ngrn < 8) ngrn += 1 else nsaa += 8;
                }
            }
            if (nsaa > max_bytes.*) max_bytes.* = nsaa;
            // Nested calls in argument position contribute their own outgoing too.
            for (args) |arg| cg.measureOutgoingExpr(arg, max_bytes);
        },
        // M6: a call inside an expression-position if/block contributes too.
        .if_stmt => cg.measureOutgoingStmt(node_idx, max_bytes),
        .block => cg.measureOutgoingBlock(node_idx, max_bytes),
        // M7 gap (pre-existing): a call inside a value-`loop` body contributes its
        // outgoing args too; this arm was missing, a latent cache/frame hole.
        .loop_expr => cg.measureOutgoingBlock(n.lhs, max_bytes),
        .labeled => cg.measureOutgoingLabeled(node_idx, max_bytes),
        // M9: a struct literal's field values / a field access receiver may hold a
        // call whose outgoing args contribute.
        .struct_init => {
            cg.measureOutgoingExpr(n.lhs, max_bytes);
            for (Ast.rangeSlice(cg.tree, n.rhs)) |fi| cg.measureOutgoingExpr(cg.tree.nodes[fi].lhs, max_bytes);
        },
        .field_access => cg.measureOutgoingExpr(n.lhs, max_bytes),
        else => {},
    }
}

/// Measure outgoing-args bytes under a labeled wrapper's inner construct.
fn measureOutgoingLabeled(cg: *Codegen, node_idx: Ast.Index, max_bytes: *u32) void {
    const inner_idx = cg.tree.nodes[node_idx].lhs;
    const inner = cg.tree.nodes[inner_idx];
    switch (inner.tag) {
        .block => cg.measureOutgoingBlock(inner_idx, max_bytes),
        .loop_expr => cg.measureOutgoingBlock(inner.lhs, max_bytes),
        .while_stmt => cg.measureOutgoingStmt(inner_idx, max_bytes),
        .for_stmt => {
            const h = Ast.forHeaderAt(cg.tree, inner.rhs);
            cg.measureOutgoingExpr(h.lo, max_bytes);
            cg.measureOutgoingExpr(h.hi, max_bytes);
            cg.measureOutgoingBlock(inner.lhs, max_bytes);
        },
        else => {},
    }
}

/// Walk a block's statements to size their temp spills (feeding `max_temp_slots`,
/// the count of simultaneously-live spill slots); mirrors genBinary/genCall
/// exactly. `base` is the temp depth the block's statements lower from: 0 for a
/// statement-context block (a fn body, an if-stmt/while arm — each statement starts
/// fresh), but the INHERITED `cg.depth` for an expression-context block (a value
/// `{ ... }` whose statements lower at the same depth the block was entered at, see
/// `lowerBlockValue`). Threading `base` keeps the measure walk in lockstep with
/// lowering so an arm at depth N reserves slots at N.. (not 0..), otherwise the top
/// spill lands past `[sp,#frame]` and clobbers the saved x29/x30 — a SIGBUS in
/// non-leaf fns.
fn measureDepth(cg: *Codegen, block_idx: Ast.Index, base: u32) void {
    const block = cg.tree.nodes[block_idx];
    for (Ast.rangeSlice(cg.tree, block.lhs)) |stmt_idx| {
        const stmt = cg.tree.nodes[stmt_idx];
        switch (stmt.tag) {
            .var_decl => cg.measureExpr(stmt.lhs, base),
            .assign => cg.measureExpr(stmt.rhs, base),
            .return_stmt => if (stmt.lhs != Ast.none) cg.measureExpr(stmt.lhs, base),
            .expr_stmt => cg.measureExpr(stmt.lhs, base),
            .block => cg.measureDepth(stmt_idx, base),
            .if_stmt, .while_stmt => cg.measureDepthStmt(stmt_idx, base),
            .for_stmt => {
                const h = Ast.forHeaderAt(cg.tree, stmt.rhs);
                cg.measureExpr(h.lo, base);
                // lowerFor reserves the hi re-eval temp at `base` (cg.depth) then
                // bumps cg.depth before genExpr(h.hi), so the hi expression's
                // interior spills at `base + 1`. Size it there or a non-trivial
                // hi (binary/call) overruns the frame onto saved x29/x30 (SIGBUS).
                cg.bumpTempSlots(base + 1);
                cg.measureExpr(h.hi, base + 1);
                cg.measureDepth(stmt.lhs, base); // body at the same base depth
            },
            .break_stmt => if (stmt.lhs != Ast.none) cg.measureExpr(stmt.lhs, base),
            .continue_stmt => {},
            // A labeled construct as a statement is parsed wrapped in an expr_stmt
            // (handled above via measureExpr); this direct arm mirrors that for the
            // defensive direct-`.labeled` lowerStmt path, sizing its body at `base`.
            .labeled => cg.measureExpr(stmt_idx, base),
            else => {},
        }
    }
}

/// Recurse a control-flow statement (if/while) for `measureDepth`: a comparison/
/// &&/|| in a cond is a `.binary` so `measureExpr` sizes its temp (depth+1); then/
/// else/body arms recurse as their own blocks at the same `base`.
fn measureDepthStmt(cg: *Codegen, stmt_idx: Ast.Index, base: u32) void {
    const stmt = cg.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .if_stmt => {
            cg.measureExpr(stmt.lhs, base);
            const h = Ast.ifHeaderAt(cg.tree, stmt.rhs);
            cg.measureDepth(h.then_block, base);
            if (h.else_node != Ast.none) {
                if (cg.tree.nodes[h.else_node].tag == .if_stmt) {
                    cg.measureDepthStmt(h.else_node, base);
                } else {
                    cg.measureDepth(h.else_node, base);
                }
            }
        },
        .while_stmt => {
            cg.measureExpr(stmt.lhs, base);
            cg.measureDepth(stmt.rhs, base);
        },
        else => {},
    }
}

fn measureExpr(cg: *Codegen, node_idx: Ast.Index, depth: u32) void {
    if (node_idx == Ast.none) return;
    const n = cg.tree.nodes[node_idx];
    // M9 (SIGBUS-critical): a STRUCT-valued control-flow expression (value-if /
    // block / loop / labeled) in a reg-pair/value sink is materialized into a temp
    // span at `depth` by genStructExprPair, and genStructIfTo/…/BlockTo route each
    // arm's value into that temp; arm INTERIORS then spill at `depth + tempSlots`.
    // Reserve the temp and size the arms one span deeper. (A struct loop/labeled
    // body lowers at the SAME depth as the result slot — but sizing one deeper only
    // OVER-reserves, which is safe.)
    switch (n.tag) {
        .if_stmt, .block, .loop_expr, .labeled => {
            if (cg.node_types[node_idx].kind == .@"struct") {
                const span = cg.tempSlots(cg.node_types[node_idx]);
                cg.bumpTempSlots(depth + span);
                cg.measureStructCtrlArms(node_idx, depth + span);
                return;
            }
        },
        else => {},
    }
    switch (n.tag) {
        .unary => cg.measureExpr(n.lhs, depth),
        .binary => {
            // genBinary spills the lhs result at slot `depth`, so this needs
            // `depth + 1` temp slots. Size by the highest slot actually STORED,
            // not the highest depth visited — otherwise the top store lands at
            // [fp] and clobbers the saved x29 (a silent ABI violation).
            cg.bumpTempSlots(depth + 1);
            cg.measureExpr(n.lhs, depth);
            cg.measureExpr(n.rhs, depth + 1);
        },
        .call => {
            // genCallInner spills each arg at the running depth, advancing by
            // tempSlots(argTy) (a struct >16B spans multiple 16-byte slots). Mirror
            // that EXACTLY — a fixed +1 per arg under-counts a struct arg and the
            // top spill overruns the frame onto saved x29/x30 (SIGBUS).
            const args = Ast.rangeSlice(cg.tree, n.rhs);
            var d = depth;
            for (args) |arg| {
                const span = cg.tempSlots(cg.node_types[arg]);
                cg.bumpTempSlots(d + span); // the arg's temp span ends at d+span
                cg.measureExpr(arg, d); // the arg expression's interior spills at d
                d += span;
            }
        },
        // M6: an if/block in expression position. lowerIfValue/lowerBlockValue run
        // the arms at the INHERITED `cg.depth` (= this `depth`), so the arm internals
        // must be sized from `depth`, not 0 — otherwise a nested-binary arm in a deep
        // sub-operand (e.g. the Nth call arg) under-counts slots and the top spill
        // overruns the frame. The cond is a `.binary`/etc. sized at the current `depth`.
        .if_stmt => {
            const h = Ast.ifHeaderAt(cg.tree, cg.tree.nodes[node_idx].rhs);
            cg.measureExpr(cg.tree.nodes[node_idx].lhs, depth); // cond
            cg.measureDepth(h.then_block, depth);
            if (h.else_node != Ast.none) {
                if (cg.tree.nodes[h.else_node].tag == .if_stmt)
                    cg.measureExpr(h.else_node, depth)
                else
                    cg.measureDepth(h.else_node, depth);
            }
        },
        .block => cg.measureDepth(node_idx, depth),
        // M7: a `loop` in expression position. lowerLoopValue puts the result slot
        // at `depth` and lowers the body ONE deeper (= depth+1), so mirror that
        // exactly — a `break <deeply-nested-expr>` inside the body is then sized
        // via measureDepth(body, depth+1) → its break arm at depth+1.
        .loop_expr => {
            cg.bumpTempSlots(depth + 1); // the result slot at `depth`
            cg.measureDepth(cg.tree.nodes[node_idx].lhs, depth + 1); // body one deeper
        },
        // M8 (SIGBUS-critical): a labeled construct in expression position. A
        // labeled value LOOP and a labeled value BARE BLOCK both put their result
        // slot at `depth` and lower the body ONE deeper (lowerLoopValue /
        // lowerLabeledBlock), so mirror `.loop_expr` EXACTLY — a `break @L
        // <deeply-nested-expr>` or the block's trailing expr is then sized via
        // measureDepth(body, depth+1). A labeled while/for is a `()` statement: its
        // body and (for `for`) hi re-eval temp lower at `depth` (no result slot).
        .labeled => {
            const inner_idx = cg.tree.nodes[node_idx].lhs;
            const inner = cg.tree.nodes[inner_idx];
            switch (inner.tag) {
                .loop_expr => {
                    cg.bumpTempSlots(depth + 1);
                    cg.measureDepth(inner.lhs, depth + 1);
                },
                .block => {
                    cg.bumpTempSlots(depth + 1);
                    cg.measureDepth(inner_idx, depth + 1);
                },
                .while_stmt => cg.measureDepthStmt(inner_idx, depth),
                .for_stmt => {
                    const h = Ast.forHeaderAt(cg.tree, inner.rhs);
                    cg.measureExpr(h.lo, depth);
                    cg.bumpTempSlots(depth + 1); // hi re-eval temp (see lowerFor)
                    cg.measureExpr(h.hi, depth + 1);
                    cg.measureDepth(inner.lhs, depth);
                },
                else => {},
            }
        },
        // M9 (SIGBUS-critical): a struct construction in value position is
        // materialized into a temp span at `depth` (genStructExprPair /
        // genStructTo), so reserve `depth + tempSlots(self)`. genStructInit / the
        // struct-arg path advance cg.depth PAST the struct's own span before
        // evaluating each field VALUE (so a nested call in a field initializer
        // spills past these bytes, not onto them) — mirror that: measure each field
        // value at `depth + tempSlots(self)`. A nested struct field value recurses
        // into the SAME destination span but its OWN .struct_init arm re-advances.
        .struct_init => {
            const self_ty = cg.node_types[node_idx];
            const span = cg.tempSlots(self_ty);
            cg.bumpTempSlots(depth + span);
            for (Ast.rangeSlice(cg.tree, n.rhs)) |fi| cg.measureExpr(cg.tree.nodes[fi].lhs, depth + span);
        },
        // A field access: an rvalue base (e.g. a returned struct's field)
        // materializes the receiver into a temp span at `depth`, so reserve
        // `depth + tempSlots(base)` and size it. A LOCAL-rooted place reads
        // directly with NO temp UNLESS the field itself is a struct VALUE in a
        // reg-pair sink (genStructExprPair / genFieldAccessScalar materialize it
        // into a temp span at `depth` — e.g. `return o.i` where `i` is a struct).
        .field_access => {
            if (!cg.isLocalRootedPlace(node_idx)) {
                const base_ty = cg.node_types[n.lhs];
                cg.bumpTempSlots(depth + cg.tempSlots(base_ty));
                cg.measureExpr(n.lhs, depth);
            } else if (cg.node_types[node_idx].kind == .@"struct") {
                cg.bumpTempSlots(depth + cg.tempSlots(cg.node_types[node_idx]));
            }
        },
        // A bare struct-typed identifier in a reg-pair value sink (e.g. `return p`,
        // `q := p` is handled as a direct copy, but a return/break/arg of a bare
        // small-struct local) is materialized into a temp span at `depth` by
        // genStructExprPair. Reserve it or the return-copy store overruns the frame.
        .identifier => {
            if (cg.node_types[node_idx].kind == .@"struct")
                cg.bumpTempSlots(depth + cg.tempSlots(cg.node_types[node_idx]));
        },
        else => {}, // leaf: value goes to x0, occupies no temp slot itself
    }
}

/// Size the interiors of a struct-valued control-flow construct (if/block/loop/
/// labeled) at temp depth `d` — mirrors genStructIfTo/…/Labeled lowering. The
/// result slot itself is reserved by the caller; this sizes the arm bodies.
fn measureStructCtrlArms(cg: *Codegen, node_idx: Ast.Index, d: u32) void {
    const n = cg.tree.nodes[node_idx];
    switch (n.tag) {
        .if_stmt => {
            const h = Ast.ifHeaderAt(cg.tree, n.rhs);
            cg.measureExpr(n.lhs, d); // cond
            cg.measureDepth(h.then_block, d);
            if (h.else_node != Ast.none) {
                if (cg.tree.nodes[h.else_node].tag == .if_stmt)
                    cg.measureStructCtrlArms(h.else_node, d)
                else
                    cg.measureDepth(h.else_node, d);
            }
        },
        .block => cg.measureDepth(node_idx, d),
        .loop_expr => cg.measureDepth(n.lhs, d),
        .labeled => {
            const inner_idx = n.lhs;
            const inner = cg.tree.nodes[inner_idx];
            switch (inner.tag) {
                .loop_expr => cg.measureDepth(inner.lhs, d),
                .block => cg.measureDepth(inner_idx, d),
                else => {},
            }
        },
        else => {},
    }
}

fn bumpTempSlots(cg: *Codegen, n: u32) void {
    if (n > cg.max_temp_slots) cg.max_temp_slots = n;
}

// ---- emit helpers ----------------------------------------------------------

/// Append one instruction word (little-endian) and, if a listing is requested,
/// its mnemonic text.
fn emit(cg: *Codegen, word: u32, comptime mnemonic: []const u8) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, word, .little);
    try cg.code.appendSlice(cg.gpa, &buf);
    if (cg.listing) |*l| {
        try l.appendSlice(cg.gpa, "  " ++ mnemonic ++ "\n");
    }
}

/// `emit` with a formatted listing line.
fn emitFmt(cg: *Codegen, word: u32, comptime fmt: []const u8, args: anytype) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, word, .little);
    try cg.code.appendSlice(cg.gpa, &buf);
    if (cg.listing) |*l| {
        try l.appendSlice(cg.gpa, "  ");
        try l.print(cg.gpa, fmt ++ "\n", args);
    }
}

/// Materialize an i64 into `rd` (1..4 words) and append them + a listing line.
fn emitImm64(cg: *Codegen, rd: u32, value: i64) !void {
    var imm_words: [16]u8 = undefined;
    var len: usize = 0;
    Aarch64.movImm64(&imm_words, &len, rd, value);
    try cg.code.appendSlice(cg.gpa, imm_words[0..len]);
    if (cg.listing) |*l| {
        try l.print(cg.gpa, "  mov x{d}, #{d}\n", .{ rd, value });
    }
}

// ---- AST / token helpers ---------------------------------------------------

/// The local stack slot a `.local` resolution names. Callers only reach this for
/// nodes the resolver bound to a local, so a non-local is a compiler bug.
fn localSlot(cg: *Codegen, node_idx: Ast.Index) u32 {
    const res = cg.resolutions[node_idx];
    std.debug.assert(res == .local);
    return res.local;
}

/// Parse a number-literal token into i64, stripping `_` digit separators.
/// Returns null if the literal does not fit in i64 (the front-end accepts the
/// lexical form but never range-checks it, so codegen must — otherwise an
/// out-of-range literal would silently lower to a wrong value).
fn parseInt(cg: *Codegen, tok: u32) ?i64 {
    const raw = cg.tokens[tok].text(cg.source);
    // i64 has at most 19 decimal digits; anything longer can't fit. The buffer
    // is sized so we never silently truncate a too-long literal into range.
    var buf: [24]u8 = undefined;
    var n: usize = 0;
    for (raw) |c| {
        if (c == '_') continue;
        if (n >= buf.len) return null;
        buf[n] = c;
        n += 1;
    }
    return std.fmt.parseInt(i64, buf[0..n], 10) catch null;
}

/// Decode a string-literal token into its runtime bytes. The lexer stores the
/// raw token text INCLUDING the surrounding quotes and WITHOUT decoding escapes;
/// here we strip the quotes and decode `\n \t \\ \"`. The returned slice is owned
/// by the caller (allocated via `cg.gpa`) and is NOT NUL-terminated — interning
/// appends the NUL so `str.len` excludes it. An unknown escape (or a malformed,
/// unquoted token) yields one clean Diagnostic at the token and returns null.
fn decodeStringLiteral(cg: *Codegen, tok: u32) error{OutOfMemory}!?[]u8 {
    const raw = cg.tokens[tok].text(cg.source);
    // Defensive: the lexer only produces literal_string tokens that open and
    // close with a quote, but never trust that — a bad token must not panic.
    if (raw.len < 2 or raw[0] != '"' or raw[raw.len - 1] != '"') {
        try cg.unsupported(tok, "malformed string literal");
        return null;
    }
    const body = raw[1 .. raw.len - 1];

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(cg.gpa);

    var i: usize = 0;
    while (i < body.len) : (i += 1) {
        const c = body[i];
        if (c != '\\') {
            try out.append(cg.gpa, c);
            continue;
        }
        // Escape: consume the next byte.
        i += 1;
        if (i >= body.len) {
            out.deinit(cg.gpa);
            try cg.unsupported(tok, "string literal ends with a dangling backslash");
            return null;
        }
        const decoded: u8 = switch (body[i]) {
            'n' => 0x0A,
            't' => 0x09,
            '\\' => 0x5C,
            '"' => 0x22,
            else => {
                out.deinit(cg.gpa);
                try cg.unsupported(tok, "unsupported escape sequence in string literal");
                return null;
            },
        };
        try out.append(cg.gpa, decoded);
    }

    return try out.toOwnedSlice(cg.gpa);
}

/// Record an "<feature> unsupported in codegen" diagnostic at a node's token.
fn unsupported(cg: *Codegen, tok: u32, message: []const u8) !void {
    try cg.diags.append(cg.gpa, .{ .byte_offset = cg.tokens[tok].start, .message = message });
}

fn roundUp16(n: u32) u32 {
    return (n + 15) & ~@as(u32, 15);
}

// ---------------------------------------------------------------------------
// Tests — run the real front-end pipeline, then assert on the emitted bytes.
// ---------------------------------------------------------------------------

const testing = std.testing;
const Lexer = @import("../lex.zig");
const Parser = @import("../parse.zig");

const Lowered = struct {
    tokens: []Token,
    tree: Ast.Tree,
    resolve: Resolve.Result,
    typecheck: Typecheck.Result,
    result: ProgramResult,
    source: []const u8,
    names: []Link.SymName,
    entry_name: []const u8,

    fn deinit(self: *Lowered, gpa: std.mem.Allocator) void {
        self.result.deinit(gpa);
        self.typecheck.deinit(gpa);
        self.resolve.deinit(gpa);
        for (self.names) |nm| gpa.free(nm.name);
        gpa.free(self.names);
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }

    /// The FnCode whose `sym` is the entry (main).
    fn entry(self: *const Lowered) Link.FnCode {
        for (self.result.fns) |f| {
            if (f.sym.kind == .user_fn and std.mem.eql(u8, f.sym.name, self.entry_name)) return f;
        }
        unreachable;
    }

    /// The FnCode for the user fn at source index `idx`.
    fn fnAt(self: *const Lowered, idx: usize) Link.FnCode {
        return self.result.fns[idx];
    }
};

/// Build the index→SymName table: user fns named by their source spelling, plus
/// the synthetic `print` builtin at user_fn_count. Caller owns the names.
fn buildNames(gpa: std.mem.Allocator, tree: Ast.Tree, tokens: []const Token, source: []const u8, fn_nodes: []const Ast.Index) ![]Link.SymName {
    const names = try gpa.alloc(Link.SymName, fn_nodes.len + 1);
    for (fn_nodes, 0..) |fn_idx, i| {
        const nm = tokens[tree.nodes[fn_idx].main_token].text(source);
        names[i] = .{ .kind = .user_fn, .name = try gpa.dupe(u8, nm) };
    }
    names[fn_nodes.len] = .{ .kind = .builtin, .name = try gpa.dupe(u8, "print") };
    return names;
}

/// Run lex→parse→resolve→typecheck, collect all fns, find `main`, and lower all.
fn lowerAll(source: []const u8, want_listing: bool) !Lowered {
    const gpa = testing.allocator;
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    var diag: ?Parser.Diagnostic = null;
    const tree = (try Parser.parse(gpa, tokens, &diag)) orelse return error.UnexpectedParseFailure;
    errdefer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }
    var res = try Resolve.resolve(gpa, tree, tokens, source);
    errdefer res.deinit(gpa);
    var tc = try Typecheck.check(gpa, tree, tokens, source, res.resolutions);
    errdefer tc.deinit(gpa);

    // Collect all fn_decl indices in source order; find `main`'s position.
    const prog = tree.nodes[Ast.root(tree.nodes)];
    const fn_nodes = Ast.rangeSlice(tree, prog.lhs);
    var entry_fn: u32 = std.math.maxInt(u32);
    for (fn_nodes, 0..) |fn_idx, i| {
        const d = tree.nodes[fn_idx];
        if (std.mem.eql(u8, tokens[d.main_token].text(source), "main")) entry_fn = @intCast(i);
    }
    if (entry_fn == std.math.maxInt(u32)) return error.NoMain;

    const names = try buildNames(gpa, tree, tokens, source, fn_nodes);
    errdefer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    const result = try generateProgram(gpa, tree, tokens, source, res.resolutions, tc.node_types, tc.layouts, names, fn_nodes, entry_fn, want_listing);
    return .{ .tokens = tokens, .tree = tree, .resolve = res, .typecheck = tc, .result = result, .source = source, .names = names, .entry_name = "main" };
}

fn words(code: []const u8) []align(1) const u32 {
    return std.mem.bytesAsSlice(u32, code);
}

/// Whether `target` appears among the (possibly unaligned) instruction words.
fn hasWord(w: []align(1) const u32, target: u32) bool {
    for (w) |x| if (x == target) return true;
    return false;
}

test "main with locals: prologue, frame, returns x0" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n x := 40\n y := 2\n return x + y\n}\n", false);
    defer lo.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const code = lo.entry().code;
    try testing.expect(code.len % 4 == 0);

    const w = words(code);
    // 2 locals + max temp depth 1 = 3 slots → frame roundUp16(24) = 32. No calls
    // so out_base == 0: byte-identical to M1.
    try testing.expectEqual(@as(u32, 0xA9BF7BFD), w[0]); // stp x29, x30, [sp, #-16]!
    try testing.expectEqual(@as(u32, 0x910003FD), w[1]); // mov x29, sp
    try testing.expectEqual(Aarch64.subImm(Aarch64.SP, Aarch64.SP, 32), w[2]); // sub sp, sp, #32
    try testing.expectEqual(@as(usize, 0), lo.entry().relocs.len);
    // Last two words are the frame restore + ret.
    try testing.expectEqual(@as(u32, 0xA8C17BFD), w[w.len - 2]);
    try testing.expectEqual(@as(u32, 0xD65F03C0), w[w.len - 1]);
}

test "return -5 materializes then negates" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n return -5\n}\n", false);
    defer lo.deinit(gpa);

    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    // No locals, no temps → frame 0, no sub sp. Prologue is 2 words, then the
    // expression: movz x0,#5 ; neg x0,x0 ; then epilogue ldp ; ret.
    try testing.expectEqual(@as(u32, 0xA9BF7BFD), w[0]);
    try testing.expectEqual(@as(u32, 0x910003FD), w[1]);
    try testing.expectEqual(Aarch64.movz(0, 5, 0), w[2]); // movz x0, #5
    try testing.expectEqual(Aarch64.neg(0, 0), w[3]); // neg x0, x0
    try testing.expectEqual(@as(u32, 0xA8C17BFD), w[4]);
    try testing.expectEqual(@as(u32, 0xD65F03C0), w[5]);
}

test "() main falls off the end returning 0" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() {\n x := 1\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    // Ends with movz x0,#0 ; (add sp) ; ldp ; ret.
    try testing.expectEqual(@as(u32, 0xD65F03C0), w[w.len - 1]);
    try testing.expectEqual(@as(u32, 0xA8C17BFD), w[w.len - 2]);
}

test "asm listing contains the prologue mnemonics" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n x := 40\n y := 2\n return x + y\n}\n", true);
    defer lo.deinit(gpa);
    const listing = lo.result.listing orelse return error.NoListing;
    try testing.expect(std.mem.indexOf(u8, listing, "_main:") != null);
    try testing.expect(std.mem.indexOf(u8, listing, "stp x29, x30, [sp, #-16]!") != null);
    try testing.expect(std.mem.indexOf(u8, listing, "ret") != null);
}

test "integer literal out of range is rejected, not silently zeroed" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n return 99999999999999999999999\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lo.result.diags.len);
    try testing.expectEqualStrings("integer literal out of range for codegen (i64)", lo.result.diags[0].message);
}

test "main returning bool is rejected with a clear message" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> bool {\n return true\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), lo.result.diags.len);
    try testing.expectEqualStrings("main must return int or () in codegen", lo.result.diags[0].message);
}

test "an over-budget frame is a diagnostic, not a panic" {
    const gpa = testing.allocator;
    // > 511 8-byte locals overflows the 12-bit sp-adjust immediate. Build such a
    // main programmatically and confirm it yields a clean diagnostic.
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "fn main() -> int {\n");
    var i: usize = 0;
    while (i < 600) : (i += 1) try src.print(gpa, " v{d} := {d}\n", .{ i, i });
    try src.appendSlice(gpa, " return v0\n}\n");

    var lo = try lowerAll(src.items, false);
    defer lo.deinit(gpa);
    try testing.expect(lo.result.diags.len >= 1);
    try testing.expectEqualStrings("function frame too large for codegen (too many locals)", lo.result.diags[0].message);
}

// ---- M3 tests --------------------------------------------------------------

test "two functions produce two FnCode artifacts" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn helper() -> int {\n return 7\n}\nfn main() -> int {\n return 1\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    try testing.expectEqual(@as(usize, 2), lo.result.fns.len);
    // Source order: helper sym 0, main sym 1 (entry). Symbols are named, not indexed.
    try testing.expect(lo.result.fns[0].sym.eql(.{ .kind = .user_fn, .name = "helper" }));
    try testing.expect(lo.result.fns[1].sym.eql(.{ .kind = .user_fn, .name = "main" }));
    try testing.expect(lo.entry().sym.eql(.{ .kind = .user_fn, .name = "main" }));
}

test "a call records one call26 reloc at the bl site targeting the callee" {
    const gpa = testing.allocator;
    // add is sym 0, main is sym 1. main: `return add(40, 2)`.
    var lo = try lowerAll(
        "fn add(a: int, b: int) -> int {\n return a + b\n}\nfn main() -> int {\n return add(40, 2)\n}\n",
        false,
    );
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);

    const m = lo.entry();
    try testing.expectEqual(@as(usize, 1), m.relocs.len);
    const rl = m.relocs[0];
    try testing.expectEqual(Link.RelocKind.call26, rl.kind);
    try testing.expect(rl.target.func.eql(.{ .kind = .user_fn, .name = "add" }));
    try testing.expectEqual(@as(i64, 0), rl.addend);
    // The word at the reloc site is the unresolved bl placeholder.
    const w = words(m.code);
    try testing.expectEqual(@as(u32, 0x94000000), w[rl.site / 4]);
    try testing.expectEqual(@as(u32, 0), rl.site % 4);
}

test "more than 8 args sizes out_base and stores stack args at [sp+(i-8)*8]" {
    const gpa = testing.allocator;
    // A 10-param sum fn called from main with 1..10. Stack args = 2 → 16 bytes.
    const src =
        "fn sum(a: int, b: int, c: int, d: int, e: int, f: int, g: int, h: int, i: int, j: int) -> int {\n" ++
        " return a + b\n}\n" ++
        "fn main() -> int {\n return sum(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)\n}\n";
    var lo = try lowerAll(src, false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);

    const m = lo.entry();
    const w = words(m.code);
    // main makes one call with 2 stack args → out_base = 16. The marshaling must
    // store the 9th/10th args to [sp,#0] and [sp,#8]. Find those store words.
    try testing.expect(hasWord(w, Aarch64.strSp(9, 0)));
    try testing.expect(hasWord(w, Aarch64.strSp(9, 8)));
    // Exactly one call reloc.
    try testing.expectEqual(@as(usize, 1), m.relocs.len);

    // The callee `sum` reads its 9th/10th params from the incoming stack region
    // [x29,#16] / [x29,#24] in its prologue.
    const s = lo.result.fns[0]; // sum is sym 0
    const sw = words(s.code);
    try testing.expect(hasWord(sw, Aarch64.ldrFp(9, 16)));
    try testing.expect(hasWord(sw, Aarch64.ldrFp(9, 24)));
}

test "call-arg temps stay below the saved frame pointer (x29 not clobbered)" {
    // Regression: measureExpr sized the frame by the max depth VISITED, so the
    // top arg temp of `1 + add(40,2)` was stored at [fp], clobbering the saved
    // x29. With M2's 16-byte temps the 3 simultaneously-live temp slots need
    // 3*16 = 48 bytes, so the top store still sits strictly below fp.
    const gpa = testing.allocator;
    var lo = try lowerAll("fn add(a:int,b:int)->int{ return a+b }\nfn main()->int{ return 1 + add(40,2) }\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const m = lo.entry();
    const w = words(m.code);
    // Prologue: stp; mov x29,sp; sub sp,sp,#frame — frame must be 48 (3 16B temps).
    try testing.expectEqual(Aarch64.subImm(Aarch64.SP, Aarch64.SP, 48), w[2]);
}

test "nested call f(g(1)) keeps args in temps (no clobber) and sizes temps" {
    const gpa = testing.allocator;
    // add(id(40), id(2)): the result of id(40) must be spilled to a temp before
    // id(2) runs, or the first arg would be clobbered.
    const src =
        "fn id(x: int) -> int {\n return x\n}\n" ++
        "fn add(a: int, b: int) -> int {\n return a + b\n}\n" ++
        "fn main() -> int {\n return add(id(40), id(2))\n}\n";
    var lo = try lowerAll(src, false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);

    const m = lo.entry();
    // main makes 3 calls (id, id, add): three call26 relocs.
    try testing.expectEqual(@as(usize, 3), m.relocs.len);
    // Targets: id is sym 0, add is sym 1. The outer add reloc is last (emitted
    // after both arg calls).
    try testing.expect(m.relocs[2].target.func.eql(.{ .kind = .user_fn, .name = "add" }));
    try testing.expect(m.relocs[0].target.func.eql(.{ .kind = .user_fn, .name = "id" }));
    try testing.expect(m.relocs[1].target.func.eql(.{ .kind = .user_fn, .name = "id" }));
    // No diagnostics and the code is word-aligned.
    try testing.expect(m.code.len % 4 == 0);
}

test "a () call used as an expression statement discards the result" {
    const gpa = testing.allocator;
    const src =
        "fn noop() {\n return\n}\n" ++
        "fn main() -> int {\n noop()\n return 5\n}\n";
    var lo = try lowerAll(src, false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);

    const m = lo.entry();
    // One call reloc to noop (sym 0); no spill of x0 after the call (it's a stmt).
    try testing.expectEqual(@as(usize, 1), m.relocs.len);
    try testing.expect(m.relocs[0].target.func.eql(.{ .kind = .user_fn, .name = "noop" }));
}

test "non-entry function with params spills x0..x7 to local slots in prologue" {
    const gpa = testing.allocator;
    var lo = try lowerAll(
        "fn add(a: int, b: int) -> int {\n return a + b\n}\nfn main() -> int {\n return add(1, 2)\n}\n",
        false,
    );
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);

    // add is sym 0. nslots = 2 (params a,b), max_temp_slots = 1 (a+b spill) → frame
    // roundUp16((2+1)*8) = 32, out_base = 0. Prologue must store x0→slot0, x1→slot1.
    const add = lo.result.fns[0];
    const w = words(add.code);
    try testing.expectEqual(@as(u32, 0xA9BF7BFD), w[0]); // stp
    try testing.expectEqual(@as(u32, 0x910003FD), w[1]); // mov x29, sp
    try testing.expectEqual(Aarch64.subImm(Aarch64.SP, Aarch64.SP, 32), w[2]); // sub sp,#32
    try testing.expectEqual(Aarch64.strSp(0, 0), w[3]); // str x0, [sp, #0]  (slot 0)
    try testing.expectEqual(Aarch64.strSp(1, 8), w[4]); // str x1, [sp, #8]  (slot 1)
}

test "per-function frames are independent (no program-wide max)" {
    const gpa = testing.allocator;
    // big has many locals; main has one. main's frame must be small, not big's.
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "fn big() -> int {\n");
    var i: usize = 0;
    while (i < 50) : (i += 1) try src.print(gpa, " v{d} := {d}\n", .{ i, i });
    try src.appendSlice(gpa, " return v0\n}\n");
    try src.appendSlice(gpa, "fn main() -> int {\n z := 9\n return z\n}\n");

    var lo = try lowerAll(src.items, false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);

    // main has 1 local, 0 temp depth → frame roundUp16(8) = 16. The sub sp word
    // must be #16, proving main is NOT sized to big's 50-local frame.
    const m = lo.entry();
    const w = words(m.code);
    try testing.expectEqual(Aarch64.subImm(Aarch64.SP, Aarch64.SP, 16), w[2]);
}

// ---- M2 str tests ----------------------------------------------------------

test "print of a string literal: literal carried, str materialized, print body emitted" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() {\n print(\"hello world\\n\")\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);

    // main carries its decoded literal (12 bytes, NO trailing NUL — the tail adds
    // it when interning program-wide).
    const m = lo.entry();
    try testing.expectEqual(@as(usize, 1), m.literals.len);
    try testing.expectEqualSlices(u8, "hello world\n", m.literals[0].bytes);

    // main's reloc set: the str literal records [adrp_page, add_lo12] to the SAME
    // content hash, and the print call records a .call26 to the print builtin.
    // Order: adrp, add, then bl (after marshaling).
    try testing.expectEqual(@as(usize, 3), m.relocs.len);
    try testing.expectEqual(Link.RelocKind.adrp_page, m.relocs[0].kind);
    try testing.expectEqual(Link.RelocKind.add_lo12, m.relocs[1].kind);
    // Both cstr relocs share one content hash (== the literal's hash).
    try testing.expectEqual(m.relocs[0].target.cstr, m.relocs[1].target.cstr);
    try testing.expectEqual(m.literals[0].hash, m.relocs[0].target.cstr);
    try testing.expectEqual(Link.RelocKind.call26, m.relocs[2].kind);
    try testing.expect(m.relocs[2].target.func.eql(.{ .kind = .builtin, .name = "print" }));

    // The str length is materialized as a compile-time immediate (movz x1,#12).
    try testing.expect(hasWord(words(m.code), Aarch64.movz(1, 12, 0)));

    // The print body is a 2nd FnCode named (.builtin,"print"), with the write
    // shuffle + GOT load + blr and [adrp_page, ldr_lo12] import relocs.
    try testing.expectEqual(@as(usize, 2), lo.result.fns.len); // main + print
    var print_fn: ?Link.FnCode = null;
    for (lo.result.fns) |f| if (f.sym.kind == .builtin) {
        print_fn = f;
    };
    const pf = print_fn.?;
    const pw = words(pf.code);
    try testing.expect(hasWord(pw, Aarch64.movReg(2, 1))); // mov x2, x1 (len)
    try testing.expect(hasWord(pw, Aarch64.movReg(1, 0))); // mov x1, x0 (ptr)
    try testing.expect(hasWord(pw, Aarch64.movz(0, 1, 0))); // mov w0, #1 (fd)
    try testing.expect(hasWord(pw, Aarch64.blr(16))); // blr x16
    try testing.expectEqual(@as(usize, 2), pf.relocs.len);
    try testing.expectEqual(Link.RelocKind.adrp_page, pf.relocs[0].kind);
    try testing.expect(pf.relocs[0].target.import.eql(.{ .kind = .import, .name = "write" }));
    try testing.expectEqual(Link.RelocKind.ldr_lo12, pf.relocs[1].kind);
    try testing.expect(pf.relocs[1].target.import.eql(.{ .kind = .import, .name = "write" }));
}

test "print(str) marshals the one str arg into x0,x1 with no outgoing stack" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() {\n print(\"hi\")\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const m = lo.entry();
    const w = words(m.code);
    // The arg temp is stored as two words (ptr@off, len@off+8) then loaded into
    // x0,x1 — never spilled to the outgoing region [sp+0..]. No call has >register
    // args, so out_base is 0: the marshaled loads target x0 and x1.
    // The str temp store uses str x0 and str x1; the marshal uses ldr x0 / ldr x1.
    try testing.expect(hasWord(w, Aarch64.movz(1, 2, 0))); // len of "hi" = 2
}

test "a str local occupies 16 bytes (frame grows, two str/two ldr)" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() {\n s := \"hi\"\n print(s)\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const m = lo.entry();
    const w = words(m.code);
    // One str local (16B) at out_base 0 → off 0/8. Stored as str x0,[sp,#0] and
    // str x1,[sp,#8]; reloaded as ldr x0,[sp,#0] / ldr x1,[sp,#8] for the print.
    try testing.expect(hasWord(w, Aarch64.strSp(0, 0)));
    try testing.expect(hasWord(w, Aarch64.strSp(1, 8)));
    try testing.expect(hasWord(w, Aarch64.ldrSp(0, 0)));
    try testing.expect(hasWord(w, Aarch64.ldrSp(1, 8)));
}

test "string literals dedup within a function to one literal entry" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() {\n print(\"hi\")\n print(\"hi\")\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    // "hi" carried once (per-fn content-hash dedup), even though referenced twice.
    const m = lo.entry();
    try testing.expectEqual(@as(usize, 1), m.literals.len);
    try testing.expectEqualSlices(u8, "hi", m.literals[0].bytes);
}

test "many distinct literals: all carried, a within-fn duplicate dedups" {
    const gpa = testing.allocator;
    var src: std.ArrayList(u8) = .empty;
    defer src.deinit(gpa);
    try src.appendSlice(gpa, "fn main() {\n");
    var i: usize = 0;
    while (i < 40) : (i += 1) try src.print(gpa, " print(\"literal_padding_text_{d:0>2}\\n\")\n", .{i});
    // A duplicate of literal #0 must reuse the existing entry, not add a 41st.
    try src.appendSlice(gpa, " print(\"literal_padding_text_00\\n\")\n}\n");

    var lo = try lowerAll(src.items, false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    // 40 distinct literals; the 41st print reuses #0 → exactly 40 entries.
    try testing.expectEqual(@as(usize, 40), lo.entry().literals.len);
}

test "decodeStringLiteral decodes escapes and strips quotes" {
    const gpa = testing.allocator;
    // Source whose only string token is the literal under test. We tokenize,
    // find the `.string` token, and run the decoder against a minimal Codegen.
    const source = "\"hello\\n\"";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);

    var cg: Codegen = .{
        .gpa = gpa,
        .tree = .{ .nodes = &.{}, .extra = &.{} },
        .tokens = tokens,
        .source = source,
        .resolutions = &.{},
        .node_types = &.{},
        .layouts = &.{},
        .names = &.{},
        .code = .empty,
        .relocs = .empty,
        .literals = .empty,
        .listing = null,
        .diags = .empty,
        .owned_msgs = .empty,
        .out_base = 0,
        .nslots = 0,
        .slot_byte_off = &.{},
        .locals_bytes = 0,
        .depth = 0,
        .max_temp_slots = 0,
        .frame = 0,
        .labels = .empty,
        .fixups = .empty,
        .loops = .empty,
    };
    defer {
        cg.diags.deinit(gpa);
        for (cg.owned_msgs.items) |m| gpa.free(m);
        cg.owned_msgs.deinit(gpa);
    }

    var str_tok: u32 = std.math.maxInt(u32);
    for (tokens, 0..) |tk, i| if (tk.tag == .string) {
        str_tok = @intCast(i);
    };
    try testing.expect(str_tok != std.math.maxInt(u32));

    const bytes = (try cg.decodeStringLiteral(str_tok)).?;
    defer gpa.free(bytes);
    try testing.expectEqualSlices(u8, &.{ 'h', 'e', 'l', 'l', 'o', 0x0A }, bytes);
    try testing.expectEqual(@as(usize, 0), cg.diags.items.len);
}

test "decodeStringLiteral rejects an unknown escape with one diagnostic" {
    const gpa = testing.allocator;
    const source = "\"\\q\"";
    const tokens = try Lexer.tokenize(gpa, source);
    defer gpa.free(tokens);

    var cg: Codegen = .{
        .gpa = gpa,
        .tree = .{ .nodes = &.{}, .extra = &.{} },
        .tokens = tokens,
        .source = source,
        .resolutions = &.{},
        .node_types = &.{},
        .layouts = &.{},
        .names = &.{},
        .code = .empty,
        .relocs = .empty,
        .literals = .empty,
        .listing = null,
        .diags = .empty,
        .owned_msgs = .empty,
        .out_base = 0,
        .nslots = 0,
        .slot_byte_off = &.{},
        .locals_bytes = 0,
        .depth = 0,
        .max_temp_slots = 0,
        .frame = 0,
        .labels = .empty,
        .fixups = .empty,
        .loops = .empty,
    };
    defer {
        cg.diags.deinit(gpa);
        for (cg.owned_msgs.items) |m| gpa.free(m);
        cg.owned_msgs.deinit(gpa);
    }

    var str_tok: u32 = std.math.maxInt(u32);
    for (tokens, 0..) |tk, i| if (tk.tag == .string) {
        str_tok = @intCast(i);
    };
    try testing.expect(str_tok != std.math.maxInt(u32));

    const res = try cg.decodeStringLiteral(str_tok);
    try testing.expect(res == null);
    try testing.expectEqual(@as(usize, 1), cg.diags.items.len);
}

// ---- M4 Stage C: bool VALUE codegen + backpatch ----------------------------

/// Decode the imm19 field (bits[23:5], signed) of a b.cond / cbz / cbnz word.
fn imm19Of(word: u32) i32 {
    const raw: u19 = @truncate(word >> 5);
    return @as(i32, @as(i19, @bitCast(raw)));
}

/// Decode the imm26 field (bits[25:0], signed) of a `b` word.
fn imm26Of(word: u32) i32 {
    const raw: u26 = @truncate(word);
    return @as(i32, @as(i26, @bitCast(raw)));
}

/// Index of the first word equal to `target`, or null.
fn indexOfWord(w: []align(1) const u32, target: u32) ?usize {
    for (w, 0..) |x, i| if (x == target) return i;
    return null;
}

test "comparison in value context lowers to cmp + cset (no branch)" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n b := 1 < 2\n return 0\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    // cmp x1, x0 (0xEB00003F) then cset x0, lt (0x9A9FA7E0). No branch instructions.
    try testing.expect(hasWord(w, Aarch64.cmpReg(1, 0)));
    try testing.expect(hasWord(w, Aarch64.cset(0, .lt)));
    for (w) |x| {
        try testing.expect((x & 0xFF000000) != 0x54000000); // no b.cond
        try testing.expect((x & 0xFC000000) != 0x14000000); // no b
    }
}

test "literal_bool true/false lower to movz #1 / #0" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n t := true\n f := false\n return 0\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    try testing.expect(hasWord(w, Aarch64.movz(0, 1, 0)));
    try testing.expect(hasWord(w, Aarch64.movz(0, 0, 0)));
}

test "unary ! lowers to cmp x0,#0 + cset x0,eq" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n b := !(1 < 2)\n return 0\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    try testing.expect(hasWord(w, Aarch64.cmpImm(0, 0))); // cmp x0, #0
    try testing.expect(hasWord(w, Aarch64.cset(0, .eq))); // cset x0, eq
}

test "&& value: short-circuit branches backpatched to correct forward offsets" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n b := (1 < 2) && (3 < 4)\n return 0\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);

    // Two comparisons → two cmp x1,x0. The && value path uses genCond(FALL,lfalse)
    // for each operand, so each comparison emits a b.<!cc> (b.ge, inverse of lt)
    // to lfalse, then movz #1, b Lend, lfalse: movz #0, Lend:.
    // Find the unconditional `b` (top 6 bits 0b000101 → 0x14000000) to Lend.
    var b_idx: ?usize = null;
    for (w, 0..) |x, i| if ((x & 0xFC000000) == 0x14000000) {
        b_idx = i;
    };
    try testing.expect(b_idx != null);

    // Find a b.cond (b.ge = inverse of lt). Its imm19 must be a POSITIVE forward
    // offset (jumping to lfalse, which is after it).
    var found_bcond = false;
    for (w, 0..) |x, i| {
        if ((x & 0xFF000000) == 0x54000000) {
            found_bcond = true;
            const target = @as(i64, @intCast(i)) + imm19Of(x);
            try testing.expect(target > @as(i64, @intCast(i))); // forward jump
            try testing.expect(target <= @as(i64, @intCast(w.len)));
        }
    }
    try testing.expect(found_bcond);

    // The unconditional b jumps forward to Lend (past the movz #0).
    const bw = w[b_idx.?];
    const btarget = @as(i64, @intCast(b_idx.?)) + imm26Of(bw);
    try testing.expect(btarget > @as(i64, @intCast(b_idx.?)));
    try testing.expect(btarget <= @as(i64, @intCast(w.len)));

    // Both materialization movz #1 and movz #0 are present.
    try testing.expect(hasWord(w, Aarch64.movz(0, 1, 0)));
    try testing.expect(hasWord(w, Aarch64.movz(0, 0, 0)));
}

test "|| value: forward branches backpatched, no UNPLACED placeholders left" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n b := (1 < 2) || (3 < 4)\n return 0\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    // Every branch word's imm must be nonzero-resolved to a real in-range target
    // (no leftover .+0 placeholder, which would mean an unpatched fixup).
    for (w, 0..) |x, i| {
        if ((x & 0xFF000000) == 0x54000000) { // b.cond
            const t = @as(i64, @intCast(i)) + imm19Of(x);
            try testing.expect(t >= 0 and t <= @as(i64, @intCast(w.len)));
            try testing.expect(t != @as(i64, @intCast(i))); // not a self-loop placeholder
        }
        if ((x & 0xFC000000) == 0x14000000) { // b
            const t = @as(i64, @intCast(i)) + imm26Of(x);
            try testing.expect(t >= 0 and t <= @as(i64, @intCast(w.len)));
            try testing.expect(t != @as(i64, @intCast(i)));
        }
    }
}

// ---- M4 Stage D: control-flow statement codegen ----------------------------

test "if/else lowers cmp + forward b.cond + a b over the else, two inline epilogues + trailing" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n if 1 < 2 {\n return 10\n } else {\n return 20\n }\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);

    // A cmp must be present (the condition `1 < 2`).
    try testing.expect(hasWord(w, Aarch64.cmpReg(1, 0)));

    // Exactly one b.cond (the inverted-lt = ge that skips to the else when false),
    // jumping FORWARD to lelse.
    var bcond_count: usize = 0;
    var bcond_fwd = false;
    for (w, 0..) |x, i| {
        if ((x & 0xFF000000) == 0x54000000) {
            bcond_count += 1;
            const target = @as(i64, @intCast(i)) + imm19Of(x);
            if (target > @as(i64, @intCast(i))) bcond_fwd = true;
            // b.ge = inverse of b.lt: cond field (low 4 bits) == ge (0xA).
            try testing.expectEqual(@as(u32, @intFromEnum(Aarch64.Cond.ge)), x & 0xF);
        }
    }
    try testing.expectEqual(@as(usize, 1), bcond_count);
    try testing.expect(bcond_fwd);

    // Exactly one unconditional `b` (the then-block's jump over the else), forward.
    var b_count: usize = 0;
    var b_fwd = false;
    for (w, 0..) |x, i| {
        if ((x & 0xFC000000) == 0x14000000) {
            b_count += 1;
            const target = @as(i64, @intCast(i)) + imm26Of(x);
            if (target > @as(i64, @intCast(i))) b_fwd = true;
        }
    }
    try testing.expectEqual(@as(usize, 1), b_count);
    try testing.expect(b_fwd);

    // `ret` count: two inline (then/else) + one trailing fall-through = 3.
    var ret_count: usize = 0;
    for (w) |x| if (x == Aarch64.ret) {
        ret_count += 1;
    };
    try testing.expectEqual(@as(usize, 3), ret_count);
}

test "while lowers a BACKWARD unconditional b to the loop top (negative imm26)" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n i := 0\n while i < 5 {\n i = i + 1\n }\n return i\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);

    // The loop's trailing `b` jumps BACKWARD to the condition (top). Its imm26 is
    // negative — encoded as 0x17xxxxxx (sign bit of the 26-bit field set).
    var found_back = false;
    for (w, 0..) |x, i| {
        if ((x & 0xFC000000) == 0x14000000) {
            const off = imm26Of(x);
            if (off < 0) {
                found_back = true;
                const target = @as(i64, @intCast(i)) + off;
                try testing.expect(target >= 0); // lands inside the function
            }
        }
    }
    try testing.expect(found_back);
}

test "an if with no else has no trailing b over an else (only the skip b.cond)" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n if 1 < 2 {\n return 9\n }\n return 0\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    // The else-less if uses genCond(FALL, lend): one b.cond skip, NO unconditional b.
    var b_count: usize = 0;
    for (w) |x| if ((x & 0xFC000000) == 0x14000000) {
        b_count += 1;
    };
    try testing.expectEqual(@as(usize, 0), b_count);
    try testing.expect(hasWord(w, Aarch64.cmpReg(1, 0)));
}

test "return inside an if-arm does not stop codegen of the trailing statement" {
    const gpa = testing.allocator;
    // The then-arm returns 42; the code after the if (return 7) must still be
    // lowered (its own movz + epilogue present).
    var lo = try lowerAll("fn main() -> int {\n if 1 < 2 {\n return 42\n }\n return 7\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    // Both return values are materialized: movz x0,#42 and movz x0,#7 (both < 65536,
    // so single-word movz).
    try testing.expect(hasWord(w, Aarch64.movz(0, 42, 0)));
    try testing.expect(hasWord(w, Aarch64.movz(0, 7, 0)));
}

// ---- M6 expression-orientation codegen -------------------------------------

test "value-if lowers a cond + two arms + a b over the else + a join" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n c := 1\n x := if c > 0 { 7 } else { 9 }\n return x\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    // The condition `c > 0` lowers to a cmp; both arm values are materialized.
    try testing.expect(hasWord(w, Aarch64.cmpReg(1, 0)));
    try testing.expect(hasWord(w, Aarch64.movz(0, 7, 0)));
    try testing.expect(hasWord(w, Aarch64.movz(0, 9, 0)));
    // Exactly one unconditional `b` (the then-arm's jump over the else), forward.
    var b_count: usize = 0;
    for (w, 0..) |x, i| if ((x & 0xFC000000) == 0x14000000) {
        b_count += 1;
        const t = @as(i64, @intCast(i)) + imm26Of(x);
        try testing.expect(t > @as(i64, @intCast(i)));
    };
    try testing.expectEqual(@as(usize, 1), b_count);
}

test "a fn whose body is a trailing expression returns its value" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn answer() -> int { 41 + 1 }\nfn main() -> int {\n return answer()\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    // answer is sym 0: it materializes 41, spills, materializes 1, adds → x0.
    const a = lo.result.fns[0];
    const w = words(a.code);
    try testing.expect(hasWord(w, Aarch64.movz(0, 41, 0)));
    try testing.expect(hasWord(w, Aarch64.addReg(0, 1, 0)));
}

test "a bare block expression yields its trailing value" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn main() -> int {\n x := { a := 1\n a + 1 }\n return x\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    try testing.expect(hasWord(w, Aarch64.addReg(0, 1, 0)));
}

test "a () -> () fn lowers and falls into the epilogue" {
    const gpa = testing.allocator;
    var lo = try lowerAll("fn noop() -> () {\n }\nfn main() -> int {\n noop()\n return 0\n}\n", false);
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    // noop is sym 0: prologue + epilogue, ends in ret.
    const n = lo.result.fns[0];
    const w = words(n.code);
    try testing.expectEqual(@as(u32, 0xD65F03C0), w[w.len - 1]); // ret
}

test "no UNPLACED placeholders survive: every branch resolves to an in-range target" {
    const gpa = testing.allocator;
    // An else-if ladder + a while + a nested if: exercises many labels at once.
    var lo = try lowerAll(
        "fn main() -> int {\n i := 0\n while i < 3 {\n if i == 0 {\n i = i + 1\n } else if i == 1 {\n i = i + 1\n } else {\n i = i + 1\n }\n }\n return i\n}\n",
        false,
    );
    defer lo.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), lo.result.diags.len);
    const w = words(lo.entry().code);
    for (w, 0..) |x, i| {
        if ((x & 0xFF000000) == 0x54000000) { // b.cond
            const t = @as(i64, @intCast(i)) + imm19Of(x);
            try testing.expect(t >= 0 and t <= @as(i64, @intCast(w.len)));
            try testing.expect(t != @as(i64, @intCast(i)));
        }
        if ((x & 0xFC000000) == 0x14000000) { // b (forward over-else OR backward loop)
            const t = @as(i64, @intCast(i)) + imm26Of(x);
            try testing.expect(t >= 0 and t <= @as(i64, @intCast(w.len)));
            try testing.expect(t != @as(i64, @intCast(i)));
        }
    }
}
