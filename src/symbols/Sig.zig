//! A callee's identity-and-signature. A peer data module: types produces it,
//! the fingerprint folds it. Depends on `Sym` (for the symbol kind) and the
//! type checker's `Type`, not sideways on the link backend.

const Sym = @import("Sym.zig");
const types = @import("../types.zig");

/// A callee's identity-and-signature, folded into a caller's fingerprint so a
/// signature OR symbol-identity change (but not a body change) invalidates the
/// caller. [C1][C2]
///
/// The callee's resolved KIND (user_fn vs builtin vs import) is load-bearing,
/// not just its sig: a user `fn print(s: str)` has the SAME sig {[str],void} as
/// the builtin `print`, but lowers to a DIFFERENT `.func` reloc target
/// (`{user_fn,"print"}` vs `{builtin,"print"}`). Folding only the sig would let
/// a shadow-and-unshadow edit keep an identical fingerprint while the call binds
/// to a different symbol — a stale-cache miscompile. So we fold the full
/// `SymName{kind,name}` (which IS what enters the lowered bytes) too. [Cx]
pub const Sig = struct {
    kind: Sym.SymKind,
    name: []const u8,
    params: []const types.Type,
    ret: types.Type,
};
