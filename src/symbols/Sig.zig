//! A callee's identity-and-signature. A peer data module: types produces it,
//! the fingerprint folds it. Depends on `Sym` (for the symbol kind) and the
//! type checker's `Type`, not sideways on the link backend.

const Sym = @import("Sym.zig");
const types = @import("../types.zig");

/// A callee's identity-and-signature, folded into a caller's fingerprint so a
/// signature OR symbol-identity change (but not a body change) invalidates the
/// caller. [C1][C2]
pub const Sig = struct {
    /// Load-bearing, not just decorative: folded into the fingerprint alongside
    /// the sig (see the `(b)` fold site in `query/Fingerprint.zig` for why a
    /// shadow/unshadow of an identical sig must still flip the caller's hash). [C6]
    kind: Sym.SymKind,
    /// BORROWED from the resolve fns table (`GraphResult.fns[i].name`); its
    /// lifetime is tied to the sibling resolve result. NEVER freed through a Sig.
    name: []const u8,
    params: []const types.Type,
    ret: types.Type,
};
