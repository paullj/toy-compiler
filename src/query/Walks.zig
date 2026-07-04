//! Thin shims that collect the inputs `Fingerprint.fingerprint` folds, by riding
//! the ONE AST walk in `AstWalk`:
//!
//!   * `walkCalls` — the callee signatures, in body-walk order (the order the
//!     fingerprint's (b) component expects).
//!   * `walkTouchedSig` — the touched-type layout descriptors (the (c) component).
//!     `freeTouched` frees the layout slices a result owns.
//!
//! AST visit order is defined ONCE in `AstWalk.walk`; these build the matching
//! `CallVisitor`/`TouchedVisitor` over it, so there is no second hand-mirrored
//! walk to drift from the fingerprint fold. `frozen` is `anytype` so the
//! single-file and graph callers can share it (both pass `*const Driver.Frozen`).
//!
//! The visitor reads (`frozen.sigs`/`frozen.names` in `CallVisitor`; `frozen.layouts`/
//! `frozen.enum_layouts` in `typeRefToType`/`structLayoutBytes`/`enumLayoutBytes`) fold
//! the SAME callee-signature and touched-type-layout dependencies the typecheck
//! `body->signature`/`body->layout` edges record. The fingerprint these walks feed IS
//! the codegen node's own fp, which records a single `codegen(gid)->body(gid)` edge and
//! reaches every signature/layout dependency transitively through the body subtree — so
//! the per-dependency `codegen(fn)->signature(callee)`/`codegen(fn)->layout(type)` edges
//! are deliberately NOT recorded, not forgotten. The reads are SAFE to leave direct:
//! invalidation is still content-fingerprint, and the fingerprint folds these
//! dependencies correctly.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Fingerprint = @import("Fingerprint.zig");
const AstWalk = @import("AstWalk.zig");
const Mono = @import("../symbols/Mono.zig");

/// Collect the signatures of every function this fn calls, in body walk order
/// (matching the fingerprint's walk) so the fingerprint's (b) component lines up.
pub fn walkCalls(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, out: *std.ArrayList(Fingerprint.Sig)) !void {
    const Frozen = @TypeOf(frozen.*);
    var v = AstWalk.CallVisitor(Frozen){ .gpa = gpa, .frozen = frozen, .out = out };
    try AstWalk.walk(.{ .tree = frozen.tree, .tokens = frozen.tokens, .source = frozen.source }, idx, &v);
}

/// Collect the types this fn touches (its node types under the subtree, in walk
/// order), each as a `TouchedType` carrying — for an aggregate — an index-free
/// layout descriptor so a field-layout edit flips every using fn's hash. Caller
/// frees each `layout` slice (see `freeTouched`).
///
/// `fn_sig` threads in the OWNING fn's signature so the fn_decl case folds the
/// ABI-correct param/return types. It is the fn's typecheck Sig (`frozen.sigs[idx]`
/// single-file / `gf.sigs[gid]` graph) when known; null elsewhere. The sig's
/// `params`/`ret` carry the GLOBAL struct/enum ids the typechecker resolved —
/// including a CROSS-MODULE qualified type-ref `b: rect.Rect` which a bare-name scan
/// would otherwise mis-resolve to the FIRST same-named type in the program-wide
/// layout table (the cross-module hole). Folding the sig types
/// makes a pub-type LAYOUT edit reach EXACTLY the importers that name it.
pub fn walkTouchedSig(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, fn_sig: ?Fingerprint.Sig, out: *std.ArrayList(Fingerprint.TouchedType)) error{OutOfMemory}!void {
    const Frozen = @TypeOf(frozen.*);
    var v = AstWalk.TouchedVisitor(Frozen){ .gpa = gpa, .frozen = frozen, .fn_sig = fn_sig, .out = out };
    try AstWalk.walk(.{ .tree = frozen.tree, .tokens = frozen.tokens, .source = frozen.source }, idx, &v);
}

/// Free the layout slices owned by a `walkTouchedSig` result.
pub fn freeTouched(gpa: std.mem.Allocator, items: []const Fingerprint.TouchedType) void {
    for (items) |t| if (t.layout.len > 0) gpa.free(t.layout);
}

/// Build the ordered `TouchedType` list for a monomorphized instance's concrete
/// type-args (M2), each carrying its full index-free layout descriptor via the
/// same `appendTouched` the (c) touched fold uses. Fed into `fingerprint`'s (d)
/// component so `id[int]` and `id[Point]` diverge and a struct-layout edit to a
/// type-arg invalidates exactly the dependent instance. Empty for a non-generic
/// fn (no args). Caller frees each `layout` via `freeTouched`.
pub fn walkTypeArgs(gpa: std.mem.Allocator, frozen: anytype, args: []const @import("../layout/Engine.zig").Type, out: *std.ArrayList(Fingerprint.TouchedType)) error{OutOfMemory}!void {
    for (args) |a| try AstWalk.appendTouched(gpa, frozen, a, out);
}

/// Build the ordered `Fingerprint.ResolvedConformance` list for a monomorphized
/// instance's resolved `[T has P]` bounds (M13), fed into `fingerprint`'s (e)
/// component. Each entry's `conform` is the conforming type's index-free layout
/// descriptor built by the SAME `appendTouched` the type-arg (d) fold uses (reading
/// `frozen.layouts`, the codegen-time concrete reified layout); `protocol_name` and
/// `witness_syms` are borrowed verbatim from the `Mono.ResolvedConformance` (both
/// outlive codegen). Empty for a non-bounded instance. Caller frees via `freeConformances`.
pub fn walkConformances(gpa: std.mem.Allocator, frozen: anytype, conformances: []const Mono.ResolvedConformance, out: *std.ArrayList(Fingerprint.ResolvedConformance)) error{OutOfMemory}!void {
    for (conformances) |rc| {
        var tmp: std.ArrayList(Fingerprint.TouchedType) = .empty;
        defer tmp.deinit(gpa);
        try AstWalk.appendTouched(gpa, frozen, rc.conform_ty, &tmp);
        try out.append(gpa, .{ .protocol_name = rc.protocol_name, .conform = tmp.items[0], .witness_syms = rc.witness_syms });
    }
}

/// Free the conform-layout slices owned by a `walkConformances` result (the outer
/// list is caller-owned; `protocol_name`/`witness_syms` are borrowed, never freed here).
pub fn freeConformances(gpa: std.mem.Allocator, items: []const Fingerprint.ResolvedConformance) void {
    for (items) |c| if (c.conform.layout.len > 0) gpa.free(c.conform.layout);
}
