//! What an identifier expression resolves to. A peer data module: resolve
//! produces it, types/codegen consume it — each depends on this DATA, not on the
//! resolve stage sideways.

const Ast = @import("../ast/Ast.zig");

/// What an identifier expression refers to, once resolved.
pub const Resolution = union(enum) {
    /// Not (yet) resolved, or resolution failed (an error was reported).
    unresolved,
    /// A parameter or `:=` local, named by its per-function slot index.
    local: u32,
    /// A top-level function, named by its index into the function table.
    func: u32,
    /// On a `break`/`continue` node carrying a `@label`: the target's INNER
    /// construct node (the `loop_expr`/`while_stmt`/`for_stmt`/`block` that the
    /// `labeled` wrapper wraps). Typecheck/Codegen match this against their
    /// context stacks to pick the named (not innermost) target.
    label: Ast.Index,
};
