//! A callee's identity-and-signature. A peer data module: types produces it,
//! the fingerprint folds it. Depends on `Sym` (for the symbol kind) and the
//! type checker's `Type`, not sideways on the link backend.

const Sym = @import("Sym.zig");
const types = @import("../types.zig");

/// A callee's identity-and-signature, folded into a caller's fingerprint so a
/// signature OR symbol-identity change (but not a body change) invalidates the
/// caller.
pub const Sig = struct {
    /// Load-bearing, not just decorative: folded into the fingerprint alongside
    /// the sig (see the `(b)` fold site in `query/Fingerprint.zig` for why a
    /// shadow/unshadow of an identical sig must still flip the caller's hash).
    kind: Sym.SymKind,
    /// BORROWED from the resolve fns table (`GraphResult.fns[i].name`); its
    /// lifetime is tied to the sibling resolve result. NEVER freed through a Sig.
    name: []const u8,
    params: []const types.Type,
    ret: types.Type,

    /// True when some param is a check-time `type_var` — i.e. this sig is a generic
    /// template. The single home for the test the bare-inferred-generic-call sites
    /// (`lower`, `AstWalk.CallVisitor`, `Mono.callInstanceRef`) gate on.
    pub fn hasTypeVar(sig: Sig) bool {
        for (sig.params) |p| if (p.isTypeVar()) return true;
        return false;
    }

    /// `1 + max type_var ordinal` over the sig's params — the generic-param count
    /// `Infer.infer` needs when only the template's sig (not its `FnSym`) is on hand.
    /// Equals `generic_params.len` for any call that survived Pass C (every var of a
    /// surviving call is bound, hence appears in a value param).
    pub fn genericParamCount(sig: Sig) u32 {
        var m: u32 = 0;
        for (sig.params) |p| if (p.isTypeVar() and p.typeVarOrd() > m) {
            m = p.typeVarOrd();
        };
        return m + 1;
    }
};
