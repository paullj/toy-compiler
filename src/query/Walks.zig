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
//! M16 BACK-DOOR-READ AUDIT — codegen-level MIRROR sites (deferred to a later
//! codegen-decompose stage; NOT a gap in the M16 typecheck DAG). The reads inside
//! the visitors (`frozen.sigs`/`frozen.names` in `CallVisitor`; `frozen.layouts`/
//! `frozen.enum_layouts` in `typeRefToType`/`structLayoutBytes`/`enumLayoutBytes`)
//! are the CODEGEN-level mirrors of the typecheck firewall reads — they fold the
//! SAME callee-signature and touched-type-layout dependencies the typecheck
//! `body->signature`/`body->layout` edges already record, but at the codegen
//! fingerprint level. The codegen NodeKey is still COARSE this milestone
//! (Engine.nodeKeyFor maps the single `check` Cache.Phase and codegen records its
//! own already-computed fingerprint as the node fp), so the plan's
//! `codegen(fn)->signature(callee)`/`codegen(fn)->layout(type)` edges are NOT yet
//! recorded — only the typecheck-level edges are. These reads stay DIRECT here
//! because the fingerprint these walks feed is itself the codegen node's fp;
//! routing them through `signature(callee)`/`layout(type)` query calls is the
//! deferred codegen-decompose work. They are SAFE to leave in M16: invalidation is
//! still content-fingerprint, and the fingerprint folds these dependencies
//! correctly today.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Fingerprint = @import("Fingerprint.zig");
const AstWalk = @import("AstWalk.zig");

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
/// layout table (the cross-module M9 / TOP-RISK-#1 hole). Folding the sig types
/// makes a pub-type LAYOUT edit reach EXACTLY the importers that name it. [design 10]
pub fn walkTouchedSig(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, fn_sig: ?Fingerprint.Sig, out: *std.ArrayList(Fingerprint.TouchedType)) error{OutOfMemory}!void {
    const Frozen = @TypeOf(frozen.*);
    var v = AstWalk.TouchedVisitor(Frozen){ .gpa = gpa, .frozen = frozen, .fn_sig = fn_sig, .out = out };
    try AstWalk.walk(.{ .tree = frozen.tree, .tokens = frozen.tokens, .source = frozen.source }, idx, &v);
}

/// Free the layout slices owned by a `walkTouchedSig` result.
pub fn freeTouched(gpa: std.mem.Allocator, items: []const Fingerprint.TouchedType) void {
    for (items) |t| if (t.layout.len > 0) gpa.free(t.layout);
}
