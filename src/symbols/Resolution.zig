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
    /// A top-level function, named by its index into the function table. In a
    /// single-file resolve this is the per-file fn index; in a whole-graph resolve
    /// (`resolve_graph`) it is the GRAPH-GLOBAL fn id (an index into the merged
    /// program-wide function table). Either way Typecheck/Codegen index their fn
    /// table by it — the table is built in the same id space the resolver used.
    func: u32,
    /// An imported module namespace bound by an `import` decl (the `.` 4th
    /// meaning). Carries the GRAPH-GLOBAL module id of the imported module.
    /// Written onto the *receiver* identifier of a `mod.member` field-access; the
    /// member itself (e.g. the callee `mod.fn`) is resolved on the field_access
    /// node to a `.func` (cross-module) or stays quiet for a qualified type/enum
    /// reference (resolved later by Typecheck against that module's tables).
    module: u32,
    /// On a `break`/`continue` node carrying a `@label`: the target's INNER
    /// construct node (the `loop_expr`/`while_stmt`/`for_stmt`/`block` that the
    /// `labeled` wrapper wraps). Typecheck/Codegen match this against their
    /// context stacks to pick the named (not innermost) target.
    label: Ast.Index,
};
