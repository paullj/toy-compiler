//! Name resolution over a parsed `Ast.Tree`.
//!
//! This is the first semantic pass after parsing. It walks the program once and
//! answers, for every identifier *expression*, "what does this name refer to?":
//! either a local slot (a parameter or a `:=` declaration) or a top-level
//! function symbol. Functions are gathered first (a whole-program pass), so a
//! function may call another declared later in the file — forward references
//! resolve fine.
//!
//! Scoping is lexical: each function opens a scope for its parameters, and each
//! `{ ... }` block opens a nested scope. A name lookup walks scopes innermost
//! first, so an inner declaration may shadow an outer one. Only a *duplicate in
//! the same scope* is an error. Local slot indices are assigned per function and
//! are NOT reclaimed when a block scope closes — every local in a function gets a
//! distinct slot, which a later codegen pass can map straight onto a stack frame.
//!
//! Output is flat and index-based to stay cache-friendly later: `resolutions` is
//! a `[]Resolution` parallel to the node array (indexed by node index). The pass
//! never stops at the first error — it records a `Diagnostic` and keeps going so
//! one run surfaces as many problems as possible. The result is in-memory only
//! for now (not a cache phase yet).

const std = @import("std");
const Token = @import("ast/Token.zig").Token;
const Ast = @import("ast/Ast.zig");

const Resolve = @This();

/// What an identifier expression refers to, once resolved. Definition lives in
/// the `symbols/` peer data module; re-exported so `Resolve.Resolution` keeps
/// working for the resolver's own internals.
pub const Resolution = @import("symbols/Resolution.zig").Resolution;

/// A reported problem. Same shape as `Parser.Diagnostic` so the driver/CLI can
/// render either uniformly (byte offset → line:col).
pub const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;

/// The pass output. Owned by the caller; free with `deinit`.
pub const Result = struct {
    /// One entry per AST node (indexed by node index). Only identifier-expression
    /// nodes carry a meaningful value; others stay `.unresolved`.
    resolutions: []Resolution,
    /// Resolution diagnostics, in discovery order.
    diags: []Diagnostic,
    /// Heap-allocated diagnostic messages (the name-bearing ones); owned so they
    /// can be freed. Static-literal messages are not in here.
    owned_msgs: [][]u8,

    pub fn deinit(self: *Result, gpa: std.mem.Allocator) void {
        gpa.free(self.resolutions);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        self.* = undefined;
    }
};

// ---- internal pass state ---------------------------------------------------

/// A top-level function symbol.
const FnSym = struct {
    /// Token index of the function's name (for diagnostics, if ever needed).
    name_tok: u32,
    /// The `fn_decl` node.
    decl_node: Ast.Index,
};

/// A declared local (parameter or `:=`). Slot is the index into the current
/// function's local list, i.e. `slot == index of this Local within `locals``.
const Local = struct {
    name_tok: u32,
    slot: u32,
};

/// One lexical scope: maps a name to the index of its `Local` in `locals`.
const Scope = struct {
    names: std.StringHashMapUnmanaged(u32) = .empty,
};

/// A label in lexical scope: its name and the inner construct node it wraps.
const LabelEntry = struct {
    name: []const u8,
    construct_node: Ast.Index,
};

gpa: std.mem.Allocator,
tree: Ast.Tree,
tokens: []const Token,
source: []const u8,

resolutions: []Resolution,
diags: std.ArrayList(Diagnostic),
owned_msgs: std.ArrayList([]u8),

/// Top-level function table + name → index map.
fns: std.ArrayList(FnSym),
fn_map: std.StringHashMapUnmanaged(u32),

/// Struct type names. A struct-named identifier used as a call CALLEE (i.e.
/// the rejected positional `Point(1,2)`) resolves quietly to `.unresolved` here —
/// Typecheck emits the specific "use named construction" diagnostic, so Resolve
/// must NOT also flag it as undeclared (a double diagnostic).
struct_names: std.StringHashMapUnmanaged(void),

/// Enum type names. A bare enum-named identifier (the `N` in a qualified
/// `N.V` / `N.V(...)`) resolves quietly to `.unresolved` — Typecheck
/// reinterprets it as variant construction, so Resolve must not flag it.
enum_names: std.StringHashMapUnmanaged(void),

/// Per-function lexical state (reset for each function).
scopes: std.ArrayList(Scope),
locals: std.ArrayList(Local),
slot_next: u32,

/// Per-function stack of labels currently in lexical scope (innermost last). A
/// `break @L`/`continue @L` scans it innermost-first by NAME. Names/nodes are
/// borrowed (source slices / frozen indices) — no per-entry free.
label_stack: std.ArrayList(LabelEntry),

/// Resolve names across the whole tree. Caller owns the returned `Result`.
pub fn resolve(gpa: std.mem.Allocator, tree: Ast.Tree, tokens: []const Token, source: []const u8) !Result {
    const resolutions = try gpa.alloc(Resolution, tree.nodes.len);
    @memset(resolutions, .unresolved);

    var r: Resolve = .{
        .gpa = gpa,
        .tree = tree,
        .tokens = tokens,
        .source = source,
        .resolutions = resolutions,
        .diags = .empty,
        .owned_msgs = .empty,
        .fns = .empty,
        .fn_map = .empty,
        .struct_names = .empty,
        .enum_names = .empty,
        .scopes = .empty,
        .locals = .empty,
        .slot_next = 0,
        .label_stack = .empty,
    };
    // Scratch structures are freed here; the Result owns only the three arrays.
    defer {
        r.fns.deinit(gpa);
        r.fn_map.deinit(gpa);
        r.struct_names.deinit(gpa);
        r.enum_names.deinit(gpa);
        for (r.scopes.items) |*s| s.names.deinit(gpa);
        r.scopes.deinit(gpa);
        r.locals.deinit(gpa);
        r.label_stack.deinit(gpa);
    }
    errdefer {
        gpa.free(resolutions);
        r.diags.deinit(gpa);
        for (r.owned_msgs.items) |m| gpa.free(m);
        r.owned_msgs.deinit(gpa);
    }

    try r.run();

    return Result{
        .resolutions = resolutions,
        .diags = try r.diags.toOwnedSlice(gpa),
        .owned_msgs = try r.owned_msgs.toOwnedSlice(gpa),
    };
}

fn run(r: *Resolve) !void {
    if (r.tree.nodes.len == 0) return;
    const root = Ast.root(r.tree.nodes);
    const prog = r.tree.nodes[root];
    if (prog.tag != .program) return; // defensive: nothing to resolve

    const fn_nodes = Ast.rangeSlice(r.tree, prog.lhs);

    // Pass A: build the function table (and detect duplicate function names).
    // Struct names are registered separately (quiet for a struct-named callee).
    for (fn_nodes) |fn_idx| {
        const decl = r.tree.nodes[fn_idx];
        if (decl.tag == .struct_decl) {
            try r.struct_names.put(r.gpa, r.nameText(decl.main_token), {});
            continue;
        }
        if (decl.tag == .enum_decl) {
            try r.enum_names.put(r.gpa, r.nameText(decl.main_token), {});
            continue;
        }
        if (decl.tag != .fn_decl) continue;
        const name = r.nameText(decl.main_token);
        const gop = try r.fn_map.getOrPut(r.gpa, name);
        if (gop.found_existing) {
            try r.emitFmt(r.tokens[decl.main_token].start, "duplicate function '{s}'", .{name});
            continue;
        }
        gop.value_ptr.* = @intCast(r.fns.items.len);
        try r.fns.append(r.gpa, .{ .name_tok = decl.main_token, .decl_node = fn_idx });
    }

    // Seed the synthetic `print` builtin so `print(...)` resolves to a function.
    // Seeded AFTER the user-fn loop and only if absent: a user `fn print` keeps
    // its own (earlier) index, and the builtin's index == user_fn_count, which
    // Typecheck/Codegen rely on. `decl_node = Ast.none` marks it bodyless.
    if (!r.fn_map.contains("print")) {
        try r.fn_map.put(r.gpa, "print", @intCast(r.fns.items.len));
        try r.fns.append(r.gpa, .{ .name_tok = 0, .decl_node = Ast.none });
    }

    // Pass B: resolve each function body in its own lexical state.
    for (fn_nodes) |fn_idx| {
        if (r.tree.nodes[fn_idx].tag != .fn_decl) continue;
        try r.resolveFn(fn_idx);
    }
}

fn resolveFn(r: *Resolve, fn_idx: Ast.Index) error{OutOfMemory}!void {
    // Fresh per-function lexical state.
    for (r.scopes.items) |*s| s.names.deinit(r.gpa);
    r.scopes.clearRetainingCapacity();
    r.locals.clearRetainingCapacity();
    r.slot_next = 0;
    r.label_stack.clearRetainingCapacity();

    const decl = r.tree.nodes[fn_idx];
    const proto = Ast.protoAt(r.tree, decl.lhs);

    try r.pushScope();
    // Parameters live in the function's outermost scope.
    for (proto.params) |param_idx| {
        const param = r.tree.nodes[param_idx];
        _ = try r.declare(param.main_token, "duplicate parameter '{s}'");
    }

    try r.resolveBlock(decl.rhs);

    r.popScope();
}

fn resolveBlock(r: *Resolve, block_idx: Ast.Index) error{OutOfMemory}!void {
    const block = r.tree.nodes[block_idx];
    try r.pushScope();
    for (Ast.rangeSlice(r.tree, block.lhs)) |stmt_idx| {
        try r.resolveStmt(stmt_idx);
    }
    r.popScope();
}

fn resolveStmt(r: *Resolve, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = r.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .var_decl => {
            // Resolve the initializer BEFORE binding the name, so `x := x` sees
            // an outer `x` (Go semantics), not the one being declared.
            try r.resolveExpr(stmt.lhs);
            const slot = try r.declare(stmt.main_token, "redeclaration of '{s}'");
            if (slot) |s| r.resolutions[stmt_idx] = .{ .local = s };
        },
        .assign => {
            const target = r.tree.nodes[stmt.lhs];
            if (target.tag == .field_access) {
                // A field place-store `p.x = v` / `p.a.b = v`: resolve the receiver
                // chain (its root must be a local; resolveExpr flags an undeclared
                // root). The field names are resolved by Typecheck against layouts.
                try r.resolveExpr(stmt.lhs);
            } else {
                // Target is an identifier node; resolve it but flag the cases that
                // make no sense as an assignment target.
                const res = r.lookupName(target.main_token);
                r.resolutions[stmt.lhs] = res;
                switch (res) {
                    .unresolved => try r.emitFmt(
                        r.tokens[target.main_token].start,
                        "assignment to undeclared name '{s}'",
                        .{r.nameText(target.main_token)},
                    ),
                    .func => try r.emitFmt(
                        r.tokens[target.main_token].start,
                        "cannot assign to function '{s}'",
                        .{r.nameText(target.main_token)},
                    ),
                    .local => {},
                    .label => {}, // unreachable on an assign target; defensive
                }
            }
            try r.resolveExpr(stmt.rhs);
        },
        .return_stmt => {
            if (stmt.lhs != Ast.none) try r.resolveExpr(stmt.lhs);
        },
        .expr_stmt => try r.resolveExpr(stmt.lhs),
        // Nested blocks aren't produced as statements in M0, but handle them
        // defensively so the walk is total.
        .block => try r.resolveBlock(stmt_idx),
        .if_stmt => {
            // Condition is resolved in the ENCLOSING scope; each arm opens its own.
            try r.resolveExpr(stmt.lhs);
            const h = Ast.ifHeaderAt(r.tree, stmt.rhs);
            try r.resolveBlock(h.then_block);
            if (h.else_node != Ast.none) {
                // An `else if` is a nested if_stmt: its condition must resolve in
                // THIS enclosing scope, so recurse resolveStmt (not resolveBlock).
                if (r.tree.nodes[h.else_node].tag == .if_stmt)
                    try r.resolveStmt(h.else_node)
                else
                    try r.resolveBlock(h.else_node);
            }
        },
        .while_stmt => {
            try r.resolveExpr(stmt.lhs);
            try r.resolveBlock(stmt.rhs);
        },
        .for_stmt => {
            // Range bounds resolve in the ENCLOSING scope; the loop variable lives
            // in a fresh scope wrapping the body.
            const h = Ast.forHeaderAt(r.tree, stmt.rhs);
            try r.resolveExpr(h.lo);
            try r.resolveExpr(h.hi);
            try r.pushScope();
            const slot = try r.declare(stmt.main_token, "redeclaration of '{s}'");
            if (slot) |s| r.resolutions[stmt_idx] = .{ .local = s };
            try r.resolveBlock(stmt.lhs);
            r.popScope();
        },
        .break_stmt => {
            if (stmt.rhs != Ast.none) try r.resolveLabelTarget(stmt_idx, stmt.rhs, "break");
            if (stmt.lhs != Ast.none) try r.resolveExpr(stmt.lhs);
        },
        .continue_stmt => {
            if (stmt.rhs != Ast.none) try r.resolveLabelTarget(stmt_idx, stmt.rhs, "continue");
        },
        .labeled => try r.resolveLabeled(stmt_idx),
        else => try r.resolveExpr(stmt_idx),
    }
}

fn resolveExpr(r: *Resolve, node_idx: Ast.Index) error{OutOfMemory}!void {
    if (node_idx == Ast.none) return;
    const n = r.tree.nodes[node_idx];
    switch (n.tag) {
        .identifier => {
            const res = r.lookupName(n.main_token);
            r.resolutions[node_idx] = res;
            if (res == .unresolved) {
                // A struct-named identifier resolves quietly (it is either a struct
                // type-name in a position Resolve doesn't bind, or a positional
                // `Point(...)` callee that Typecheck diagnoses specifically).
                if (r.struct_names.contains(r.nameText(n.main_token))) return;
                if (r.enum_names.contains(r.nameText(n.main_token))) return;
                try r.emitFmt(
                    r.tokens[n.main_token].start,
                    "undeclared identifier '{s}'",
                    .{r.nameText(n.main_token)},
                );
            }
        },
        .literal_number, .literal_string, .literal_bool => {},
        .unary => try r.resolveExpr(n.lhs),
        .binary => {
            try r.resolveExpr(n.lhs);
            try r.resolveExpr(n.rhs);
        },
        .call => {
            try r.resolveExpr(n.lhs); // callee
            for (Ast.rangeSlice(r.tree, n.rhs)) |arg| try r.resolveExpr(arg);
        },
        // A struct construction resolves each field-init VALUE (punning's
        // synthesized identifier resolves to the in-scope local); the type-name
        // (n.lhs) is NOT resolved as a value. A field access resolves its receiver.
        .struct_init => {
            for (Ast.rangeSlice(r.tree, n.rhs)) |fi| try r.resolveExpr(r.tree.nodes[fi].lhs);
        },
        .field_access => try r.resolveExpr(n.lhs),
        // M10: variant construction. A unit `.V`/`N.V` binds no value names (the
        // type-name leaf, when present, is the enum type — resolved quietly). A
        // tuple/struct form resolves each payload arg / field-init VALUE (the
        // struct form mirrors struct_init: the field-init's lhs is the value).
        .enum_init_unit => {},
        .enum_init_tuple => for (Ast.rangeSlice(r.tree, n.rhs)) |a| try r.resolveExpr(a),
        .enum_init_struct => for (Ast.rangeSlice(r.tree, n.rhs)) |fi| try r.resolveExpr(r.tree.nodes[fi].lhs),
        .match_expr => try r.resolveMatch(node_idx),
        // A block / if used in VALUE position must still resolve inner names.
        .literal_unit => {},
        .block => try r.resolveBlock(node_idx),
        .labeled => try r.resolveLabeled(node_idx),
        .loop_expr => try r.resolveBlock(n.lhs),
        .if_stmt => {
            try r.resolveExpr(n.lhs); // cond in the enclosing scope
            const h = Ast.ifHeaderAt(r.tree, n.rhs);
            try r.resolveBlock(h.then_block);
            if (h.else_node != Ast.none) {
                if (r.tree.nodes[h.else_node].tag == .if_stmt)
                    try r.resolveExpr(h.else_node) // else-if condition in this scope
                else
                    try r.resolveBlock(h.else_node);
            }
        },
        else => {},
    }
}

/// Resolve a `labeled` wrapper: bring its label into scope (bracketing the body),
/// resolve the inner construct through its normal unlabeled path (so a labeled
/// `for` still writes its loop-var slot onto the inner `for_stmt` node exactly as
/// the unlabeled path does), then pop the label.
fn resolveLabeled(r: *Resolve, idx: Ast.Index) error{OutOfMemory}!void {
    const n = r.tree.nodes[idx];
    const name = r.nameText(n.main_token);
    // Duplicate-in-scope is a (recoverable) error; still push so inner breaks bind
    // to the innermost same-named label and the body resolves.
    for (r.label_stack.items) |e| {
        if (std.mem.eql(u8, e.name, name)) {
            try r.emitFmt(r.tokens[n.main_token].start, "duplicate label '{s}'", .{name});
            break;
        }
    }
    try r.label_stack.append(r.gpa, .{ .name = name, .construct_node = n.lhs });
    const inner = r.tree.nodes[n.lhs];
    switch (inner.tag) {
        .block => try r.resolveBlock(n.lhs),
        // loop/while/for resolve via resolveStmt (loop_expr's body is a block, but
        // while/for resolve their head exprs + loop-var in the enclosing-vs-body
        // scopes — exactly the unlabeled statement path).
        .loop_expr, .while_stmt, .for_stmt => try r.resolveStmt(n.lhs),
        else => {},
    }
    _ = r.label_stack.pop();
}

/// Resolve a `match scrut { pat -> body, ... }`. The scrutinee resolves in the
/// enclosing scope; each arm opens a fresh scope, declares its pattern bindings
/// (writing each `.local` slot onto the `pattern_binding` node, mirroring the
/// for-var pattern), resolves the arm body, then pops.
fn resolveMatch(r: *Resolve, node_idx: Ast.Index) error{OutOfMemory}!void {
    const n = r.tree.nodes[node_idx];
    try r.resolveExpr(n.lhs); // scrutinee
    for (Ast.rangeSlice(r.tree, n.rhs)) |arm_idx| {
        const arm = r.tree.nodes[arm_idx];
        try r.pushScope();
        try r.declarePattern(arm.lhs);
        const h = Ast.armHeaderAt(r.tree, arm.rhs);
        if (h.guard != Ast.none) try r.resolveExpr(h.guard); // guard sees bindings
        try r.resolveExpr(h.body);
        r.popScope();
    }
}

/// Recursively declare a pattern's bindings into the current scope. A
/// `pattern_binding` declares its name (and recurses into a struct-field
/// sub-pattern); `pattern_variant` recurses into each payload child;
/// `pattern_or` declares the first alt and reuses its slots for later alts'
/// same-named bindings (so `A(x) | B(x)` share one slot — codegen relies on it).
fn declarePattern(r: *Resolve, pat_idx: Ast.Index) error{OutOfMemory}!void {
    if (pat_idx == Ast.none) return;
    const pat = r.tree.nodes[pat_idx];
    switch (pat.tag) {
        .pattern_wildcard, .pattern_literal => {},
        .pattern_binding => {
            const slot = try r.declare(pat.main_token, "redeclaration of '{s}'");
            if (slot) |s| r.resolutions[pat_idx] = .{ .local = s };
            if (pat.rhs != Ast.none) try r.declarePattern(pat.rhs);
        },
        .pattern_variant => {
            if (pat.rhs != Ast.none)
                for (Ast.rangeSlice(r.tree, pat.rhs)) |c| try r.declarePattern(c);
        },
        .pattern_or => {
            const alts = Ast.rangeSlice(r.tree, pat.lhs);
            if (alts.len == 0) return;
            try r.declarePattern(alts[0]); // first alt: fresh slots
            for (alts[1..]) |alt| try r.bindOrAltToFirst(alt);
        },
        else => {},
    }
}

/// Resolve a later or-pattern alternative's bindings: reuse the slot of the
/// same-named binding already declared by the first alternative when present,
/// else declare a fresh one (a name-set mismatch the typechecker then rejects).
fn bindOrAltToFirst(r: *Resolve, pat_idx: Ast.Index) error{OutOfMemory}!void {
    if (pat_idx == Ast.none) return;
    const pat = r.tree.nodes[pat_idx];
    switch (pat.tag) {
        .pattern_wildcard, .pattern_literal => {},
        .pattern_binding => {
            const existing = r.lookupName(pat.main_token);
            if (existing == .local) {
                r.resolutions[pat_idx] = existing;
            } else {
                const slot = try r.declare(pat.main_token, "redeclaration of '{s}'");
                if (slot) |s| r.resolutions[pat_idx] = .{ .local = s };
            }
            if (pat.rhs != Ast.none) try r.bindOrAltToFirst(pat.rhs);
        },
        .pattern_variant => {
            if (pat.rhs != Ast.none)
                for (Ast.rangeSlice(r.tree, pat.rhs)) |c| try r.bindOrAltToFirst(c);
        },
        .pattern_or => {
            for (Ast.rangeSlice(r.tree, pat.lhs)) |alt| try r.bindOrAltToFirst(alt);
        },
        else => {},
    }
}

/// Resolve a `break @L`/`continue @L` target: scan the label stack innermost-first
/// by text, writing the target's inner construct node onto the break/continue node
/// as a `.label` resolution. A miss is an undefined/out-of-scope label error.
fn resolveLabelTarget(r: *Resolve, node_idx: Ast.Index, label_tok: u32, comptime verb: []const u8) error{OutOfMemory}!void {
    const name = r.nameText(label_tok);
    var i = r.label_stack.items.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, r.label_stack.items[i].name, name)) {
            r.resolutions[node_idx] = .{ .label = r.label_stack.items[i].construct_node };
            return;
        }
    }
    try r.emitFmt(r.tokens[label_tok].start, verb ++ " to undefined label '{s}'", .{name});
}

// ---- scope / name helpers --------------------------------------------------

fn pushScope(r: *Resolve) !void {
    try r.scopes.append(r.gpa, .{});
}

fn popScope(r: *Resolve) void {
    // Slots are function-wide; closing a scope drops only its name bindings.
    var s = r.scopes.pop().?;
    s.names.deinit(r.gpa);
}

/// Declare a name in the innermost scope. Returns its slot, or null (and emits
/// `dup_fmt`) if the name already exists in the *current* scope. `dup_fmt` must
/// have a single `{s}` for the name.
fn declare(r: *Resolve, name_tok: u32, comptime dup_fmt: []const u8) !?u32 {
    const name = r.nameText(name_tok);
    const scope = &r.scopes.items[r.scopes.items.len - 1];
    const gop = try scope.names.getOrPut(r.gpa, name);
    if (gop.found_existing) {
        try r.emitFmt(r.tokens[name_tok].start, dup_fmt, .{name});
        return null;
    }
    const slot = r.slot_next;
    r.slot_next += 1;
    const local_idx: u32 = @intCast(r.locals.items.len);
    try r.locals.append(r.gpa, .{ .name_tok = name_tok, .slot = slot });
    gop.value_ptr.* = local_idx;
    return slot;
}

/// Look a name up the scope chain (innermost first), then the function table.
fn lookupName(r: *Resolve, name_tok: u32) Resolution {
    const name = r.nameText(name_tok);
    var i = r.scopes.items.len;
    while (i > 0) {
        i -= 1;
        if (r.scopes.items[i].names.get(name)) |local_idx| {
            return .{ .local = r.locals.items[local_idx].slot };
        }
    }
    if (r.fn_map.get(name)) |fn_idx| return .{ .func = fn_idx };
    return .unresolved;
}

fn nameText(r: *const Resolve, tok: u32) []const u8 {
    return r.tokens[tok].text(r.source);
}

/// Format a name-bearing message, own the buffer, and record a diagnostic.
fn emitFmt(r: *Resolve, byte_offset: u32, comptime fmt: []const u8, args: anytype) !void {
    const msg = try std.fmt.allocPrint(r.gpa, fmt, args);
    try r.owned_msgs.append(r.gpa, msg);
    try r.diags.append(r.gpa, .{ .byte_offset = byte_offset, .message = msg });
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;
const Lexer = @import("lex.zig");
const Parser = @import("parse.zig");

const Parsed = struct {
    tokens: []Token,
    tree: Ast.Tree,
    source: []const u8,

    fn deinit(self: *Parsed, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }
};

fn parseSource(gpa: std.mem.Allocator, source: []const u8) !Parsed {
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    var diag: ?Parser.Diagnostic = null;
    const tree = (try Parser.parse(gpa, tokens, source, &diag)) orelse return error.UnexpectedParseFailure;
    return .{ .tokens = tokens, .tree = tree, .source = source };
}

/// Resolve a source and return the diagnostic count (and free everything).
fn resolveDiagCount(source: []const u8) !usize {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, source);
    defer parsed.deinit(gpa);
    var res = try resolve(gpa, parsed.tree, parsed.tokens, parsed.source);
    defer res.deinit(gpa);
    return res.diags.len;
}

test "clean program resolves with zero diagnostics" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "fn add(a: int, b: int) -> int {\n return a + b\n}\n",
    ));
}

test "forward function reference resolves" {
    // main calls add/neg declared below it.
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn main() {
        \\ x := add(1, 2)
        \\ y := neg(x)
        \\ return
        \\}
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn neg(x: int) -> int { return -x }
        \\
    ));
}

test "undeclared identifier yields exactly one diagnostic" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn f() {\n y := x\n return\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolve(gpa, parsed.tree, parsed.tokens, parsed.source);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), res.diags.len);
    try testing.expectEqualStrings("undeclared identifier 'x'", res.diags[0].message);
}

test "duplicate parameter is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f(a: int, a: int) {\n return\n}\n",
    ));
}

test "same-scope := redeclaration is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f() {\n x := 1\n x := 2\n return\n}\n",
    ));
}

test "x := x with no outer is undeclared" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f() {\n x := x\n return\n}\n",
    ));
}

test "x := x with an outer in scope resolves to the outer" {
    // Inner var_decl initializer must see the parameter `x`, not itself.
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn f(x: int) {\n x := x\n return\n}\n");
    defer parsed.deinit(gpa);
    var res = try resolve(gpa, parsed.tree, parsed.tokens, parsed.source);
    defer res.deinit(gpa);
    // The initializer `x` (an identifier expr) must resolve to the parameter's
    // slot 0; the new `:=` local takes slot 1. No diagnostics either way.
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // Find the var_decl node and assert its initializer resolved to slot 0.
    var found = false;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) {
            const init_res = res.resolutions[n.lhs];
            try testing.expect(init_res == .local);
            try testing.expectEqual(@as(u32, 0), init_res.local);
            // The decl itself binds a fresh slot (1).
            const decl_res = res.resolutions[i];
            try testing.expect(decl_res == .local);
            try testing.expectEqual(@as(u32, 1), decl_res.local);
            found = true;
        }
    }
    try testing.expect(found);
}

test "if/while conditions and arm bodies resolve cleanly" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn f(n: int) -> int {
        \\ if n < 0 {
        \\  x := 1
        \\  return x
        \\ } else {
        \\  y := 2
        \\  return y
        \\ }
        \\ while n > 0 { n = n - 1 }
        \\ return n
        \\}
        \\
    ));
}

test "undeclared name in an else-if condition is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f(n: int) {
        \\ if n > 0 {} else if bad < 1 {}
        \\ return
        \\}
        \\
    ));
}

test "assignment to an undeclared name is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn f() {\n x = 1\n return\n}\n",
    ));
}

test "for loop binds its variable and resolves its body cleanly" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn f() -> int {
        \\ s := 0
        \\ for i in 0..5 { s = s + i }
        \\ return s
        \\}
        \\
    ));
}

test "for loop variable is in scope only inside the body" {
    // `i` referenced after the loop is undeclared.
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() -> int {
        \\ for i in 0..5 { x := i }
        \\ return i
        \\}
        \\
    ));
}

test "undeclared range bound is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ for i in 0..n { x := i }
        \\ return
        \\}
        \\
    ));
}

test "loop body and break value resolve cleanly" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        \\fn f() -> int {
        \\ i := 0
        \\ loop {
        \\  if i > 3 { break i }
        \\  i = i + 1
        \\ }
        \\}
        \\
    ));
}

test "labeled loop and break @label resolve cleanly and target the named construct" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa,
        \\fn f() -> int {
        \\ @outer loop {
        \\  @inner loop { break @outer 1 }
        \\ }
        \\}
        \\
    );
    defer parsed.deinit(gpa);
    var res = try resolve(gpa, parsed.tree, parsed.tokens, parsed.source);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // The `@outer` labeled wrapper's construct is the FIRST loop_expr (outermost);
    // emitted last among the two loop_exprs (children precede parents), so it has
    // the larger node index. The break must resolve to THAT node, not the inner.
    var outer_loop: Ast.Index = Ast.none;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .loop_expr) outer_loop = @intCast(i); // last wins → outermost
    }
    var found = false;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .break_stmt) {
            try testing.expect(res.resolutions[i] == .label);
            try testing.expectEqual(outer_loop, res.resolutions[i].label);
            found = true;
        }
    }
    try testing.expect(found);
}

test "break to an undefined label is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() -> int {
        \\ @outer loop { break @nope 1 }
        \\}
        \\
    ));
}

test "continue to an undefined label is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ @outer loop { continue @nope }
        \\}
        \\
    ));
}

test "label out of scope (after the construct) is reported" {
    // `@a` is in scope only inside its loop body; a break after it sees nothing.
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ @a loop { break }
        \\ @b while true { break @a }
        \\ return
        \\}
        \\
    ));
}

test "duplicate label in scope is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        \\fn f() {
        \\ @x loop { @x loop { break } }
        \\ return
        \\}
        \\
    ));
}

test "labeled for binds its loop variable on the inner for_stmt" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa,
        \\fn f() -> int {
        \\ s := 0
        \\ @l for i in 0..5 { s = s + i }
        \\ return s
        \\}
        \\
    );
    defer parsed.deinit(gpa);
    var res = try resolve(gpa, parsed.tree, parsed.tokens, parsed.source);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // The inner for_stmt node carries the loop-var slot resolution.
    var found = false;
    for (parsed.tree.nodes, 0..) |n, i| {
        if (n.tag == .for_stmt) {
            try testing.expect(res.resolutions[i] == .local);
            found = true;
        }
    }
    try testing.expect(found);
}

test "assignment to a function name is reported" {
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "fn g() { return }\nfn f() {\n g = 1\n return\n}\n",
    ));
}

// ---- structs ---------------------------------------------------------------

test "struct construction resolves clean (type name is not flagged)" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: 1 }\n return p.x }\n",
    ));
}

test "field punning resolves the in-scope locals" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int, y: int }\nfn f() -> int { x := 1\n y := 2\n p := P { x, y }\n return p.x }\n",
    ));
}

test "punning an out-of-scope local is reported as undeclared" {
    // `y` is never declared; the punning-synthesized identifier must flag it.
    try testing.expectEqual(@as(usize, 1), try resolveDiagCount(
        "struct P { x: int, y: int }\nfn f() -> int { x := 1\n p := P { x, y }\n return p.x }\n",
    ));
}

test "field place-store resolves its receiver local" {
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int }\nfn f() { p := P { x: 1 }\n p.x = 5\n return }\n",
    ));
}

test "a struct-named callee resolves quietly (no double diagnostic with Typecheck)" {
    // `P(1)` is positional construction (Typecheck rejects it). Resolve must NOT
    // also flag `P` as undeclared — so the resolve diagnostic count is 0.
    try testing.expectEqual(@as(usize, 0), try resolveDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P(1)\n return p.x }\n",
    ));
}

test "forward call to a function resolves the callee" {
    const gpa = testing.allocator;
    var parsed = try parseSource(gpa, "fn f() {\n g()\n return\n}\nfn g() { return }\n");
    defer parsed.deinit(gpa);
    var res = try resolve(gpa, parsed.tree, parsed.tokens, parsed.source);
    defer res.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), res.diags.len);
    // The call's callee identifier resolves to a function.
    var found = false;
    for (parsed.tree.nodes) |n| {
        if (n.tag == .call) {
            const callee_res = res.resolutions[n.lhs];
            try testing.expect(callee_res == .func);
            found = true;
        }
    }
    try testing.expect(found);
}
