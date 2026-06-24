//! Type checking over a parsed `Ast.Tree` plus its `Resolve.Result`.
//!
//! This is the second semantic pass, run only after name resolution succeeded
//! (a name error would otherwise poison every type that depends on it). It walks
//! the program once, infers a `Type` for every expression node, and checks the
//! statement-level rules:
//!
//!   * `name := expr`     — the local takes the initializer's type (() is an
//!                          error: there is nothing to bind).
//!   * `name = expr`      — the value's type must match the variable's type.
//!   * `return expr?`     — the (or () , when bare) type must match the
//!                          function's declared return type.
//!   * `callee(args...)`  — the callee must be a function, and the argument count
//!                          and per-argument types must match its parameters.
//!
//! Expression types: integer/string/bool literals are obvious; an identifier
//! takes its declaration's type (via the resolution); a unary `-`/`!` constrains
//! its operand; binary `+ - * /` are int→int, comparisons/`== !=` yield bool.
//!
//! Errors never cascade: the moment an operand is `invalid` (the poison type) the
//! containing expression silently becomes `invalid` too, so each mistake yields
//! exactly one diagnostic at its origin. Type references (`int`, `bool`) are
//! `identifier` nodes whose text we map straight onto a `Type`. The result is
//! in-memory only (not a cache phase yet).

const std = @import("std");
const Token = @import("ast/Token.zig").Token;
const Ast = @import("ast/Ast.zig");
const Resolve = @import("resolve.zig");
const Resolution = @import("symbols/Resolution.zig").Resolution;
const Sig = @import("symbols/Sig.zig").Sig;
const symbols = @import("symbols/Sym.zig");

const Typecheck = @This();

/// The type kind. `invalid` is the poison/error type: it absorbs further errors
/// so one mistake produces one diagnostic. `@"struct"` carries a `struct_id`
/// indexing the per-program struct table.
pub const Kind = enum(u8) { invalid, unit, int, bool, str, never, @"struct" };

/// A type. A byte-foldable struct (not a tagged union) so it preserves `@memset`,
/// `node_types` triviality, and a stable fingerprint basis. A `@"struct"` kind
/// carries an index into the struct table; all other kinds leave it `no_struct`.
pub const Type = struct {
    kind: Kind,
    struct_id: u32 = no_struct,

    pub const no_struct: u32 = std.math.maxInt(u32);

    pub const invalid: Type = .{ .kind = .invalid };
    pub const unit: Type = .{ .kind = .unit };
    pub const int: Type = .{ .kind = .int };
    pub const @"bool": Type = .{ .kind = .bool };
    pub const str: Type = .{ .kind = .str };
    pub const never: Type = .{ .kind = .never };

    pub fn structT(id: u32) Type {
        return .{ .kind = .@"struct", .struct_id = id };
    }

    pub fn eql(a: Type, b: Type) bool {
        return a.kind == b.kind and (a.kind != .@"struct" or a.struct_id == b.struct_id);
    }

    pub fn isStruct(t: Type) bool {
        return t.kind == .@"struct";
    }
};

/// A resolved struct layout: a self-contained, index-free-ish snapshot threaded
/// into Codegen/Fingerprint. Field types still reference the struct table by id
/// (for nested structs), but offsets/size/align are precomputed here. Owned.
pub const Layout = struct {
    name: []const u8,
    field_names: [][]const u8,
    field_types: []Type,
    offsets: []u32,
    size: u32,
    @"align": u32,
};

/// Source spelling of a type reference → `Type`. Anything else is unknown (a
/// struct name, or an error). The unit type `()` is spelled with parens, not an
/// identifier, so it is NOT here (handled in `typeFromNode` via `literal_unit`).
const type_names = std.StaticStringMap(Type).initComptime(.{
    .{ "int", Type.int },
    .{ "bool", Type.bool },
    .{ "str", Type.str },
});

/// A reported problem. Same shape as `Parser`/`Resolve` diagnostics so the
/// driver/CLI can render either uniformly (byte offset → line:col).
pub const Diagnostic = @import("diagnostics/Diagnostic.zig").Diagnostic;

/// The pass output. Owned by the caller; free with `deinit`.
pub const Result = struct {
    /// One inferred type per AST node (indexed by node index). Only expression
    /// nodes carry a meaningful value; others stay `.invalid`.
    node_types: []Type,
    /// Type diagnostics, in discovery order.
    diags: []Diagnostic,
    /// Heap-allocated diagnostic messages (the data-bearing ones); owned so they
    /// can be freed. Static-literal messages are not in here.
    owned_msgs: [][]u8,
    /// One signature per function index (source order; the synthetic `print` at
    /// index user_fn_count). The codegen fingerprint folds a callee's sig so
    /// a signature change invalidates its callers. `params` are owned.
    sigs: []Sig,
    /// The struct table: one `Layout` per struct id, in declaration order.
    /// Threaded read-only into Codegen/Fingerprint. Owned.
    layouts: []Layout,

    pub fn deinit(self: *Result, gpa: std.mem.Allocator) void {
        gpa.free(self.node_types);
        gpa.free(self.diags);
        for (self.owned_msgs) |m| gpa.free(m);
        gpa.free(self.owned_msgs);
        for (self.sigs) |s| gpa.free(s.params);
        gpa.free(self.sigs);
        for (self.layouts) |l| {
            gpa.free(l.name);
            for (l.field_names) |fn_| gpa.free(fn_);
            gpa.free(l.field_names);
            gpa.free(l.field_types);
            gpa.free(l.offsets);
        }
        gpa.free(self.layouts);
        self.* = undefined;
    }
};

/// A top-level function's signature, decoded once up front so calls can be
/// checked against it (and forward references work).
const FnSym = struct {
    decl_node: Ast.Index,
    params: []Type,
    ret: Type,
};

/// A struct's resolved symbol: its decl node, name, and (after layout) per-field
/// names/types/offsets plus aggregate size/align. `state` guards the layout
/// recursion so a directly- or indirectly-recursive struct is caught once.
const LayoutState = enum { unseen, laying, done };
const StructSym = struct {
    decl_node: Ast.Index,
    name: []const u8,
    field_names: [][]const u8 = &.{},
    field_types: []Type = &.{},
    offsets: []u32 = &.{},
    size: u32 = 0,
    @"align": u32 = 1,
    state: LayoutState = .unseen,
    poisoned: bool = false,
};

/// Natural size/align of a scalar/str type (struct sizes come from the table).
fn scalarSize(kind: Kind) u32 {
    return switch (kind) {
        .int, .bool => 8,
        .str => 16,
        else => 0,
    };
}
fn scalarAlign(kind: Kind) u32 {
    return switch (kind) {
        .int, .bool, .str => 8,
        else => 1,
    };
}
fn roundUp(n: u32, a: u32) u32 {
    if (a == 0) return n;
    return (n + a - 1) / a * a;
}

/// Per-construct context, pushed/popped as bodies are entered. A label-
/// addressable stack: `kind` distinguishes a
/// value-yielding `loop`, a `()`-statement `while`/`for`, and a value-yielding
/// labeled bare block. `is_value` is true for `loop` and `labeled_block`. `label`
/// is the construct's label name (or null when unlabeled) and `construct_node` is
/// the inner construct node a `break @L`/`continue @L` matches against. `join`
/// accumulates the merge of all value-break sites (starting at `never`).
const CtxKind = enum { loop, while_for, labeled_block };
const LoopCtx = struct {
    kind: CtxKind,
    label: ?[]const u8,
    construct_node: Ast.Index,
    is_value: bool,
    join: Type,
    saw_value_break: bool,
    saw_bare_break: bool,
};

gpa: std.mem.Allocator,
tree: Ast.Tree,
tokens: []const Token,
source: []const u8,
resolutions: []const Resolution,

node_types: []Type,
diags: std.ArrayList(Diagnostic),
owned_msgs: std.ArrayList([]u8),

/// Function table, parallel to `Resolve`'s `func` indices: the resolver assigns
/// function indices in source order, and so do we (Pass A below).
fns: std.ArrayList(FnSym),

/// Per-function: a local's type indexed by its slot. Slots are function-wide and
/// assigned densely from 0, so a flat array indexed by slot is exact.
slot_types: std.ArrayList(Type),
cur_ret: Type,
loop_stack: std.ArrayList(LoopCtx),

/// The struct table: one `StructSym` per struct id, plus a name→id map.
structs: std.ArrayList(StructSym),
struct_map: std.StringHashMapUnmanaged(u32),

/// Typecheck a resolved tree. Caller owns the returned `Result`.
pub fn check(
    gpa: std.mem.Allocator,
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const Resolution,
) !Result {
    const node_types = try gpa.alloc(Type, tree.nodes.len);
    @memset(node_types, .invalid);

    var t: Typecheck = .{
        .gpa = gpa,
        .tree = tree,
        .tokens = tokens,
        .source = source,
        .resolutions = resolutions,
        .node_types = node_types,
        .diags = .empty,
        .owned_msgs = .empty,
        .fns = .empty,
        .slot_types = .empty,
        .cur_ret = .unit,
        .loop_stack = .empty,
        .structs = .empty,
        .struct_map = .empty,
    };
    defer {
        for (t.fns.items) |f| gpa.free(f.params);
        t.fns.deinit(gpa);
        t.slot_types.deinit(gpa);
        t.loop_stack.deinit(gpa);
        for (t.structs.items) |s| {
            // field_names entries are BORROWED source slices (from nameText); only
            // the arrays are owned here. The Result snapshot dupes them separately.
            gpa.free(s.field_names);
            gpa.free(s.field_types);
            gpa.free(s.offsets);
        }
        t.structs.deinit(gpa);
        t.struct_map.deinit(gpa);
    }
    errdefer {
        gpa.free(node_types);
        t.diags.deinit(gpa);
        for (t.owned_msgs.items) |m| gpa.free(m);
        t.owned_msgs.deinit(gpa);
    }

    try t.run();

    // Snapshot every fn's signature for the codegen fingerprint. `t.fns` is
    // freed by the `defer` above, so dupe each params slice into owned memory.
    const sigs = try gpa.alloc(Sig, t.fns.items.len);
    errdefer gpa.free(sigs);
    var sigs_built: usize = 0;
    errdefer for (sigs[0..sigs_built]) |s| gpa.free(@constCast(s.params));
    for (t.fns.items, 0..) |f, i| {
        // Identity (kind/name) so a caller's fingerprint folds the bound symbol,
        // not just its sig: a bodyless entry is the synthetic `print` builtin; a
        // bodied one is a user fn named by its decl token. The driver's walkCalls
        // re-derives this from the resolved SymName table, but keeping the sig
        // self-consistent here avoids a misleading half-filled struct.
        const kind: symbols.SymKind = if (f.decl_node == Ast.none) .builtin else .user_fn;
        const name = if (f.decl_node == Ast.none) "print" else t.nameText(t.tree.nodes[f.decl_node].main_token);
        sigs[i] = .{ .kind = kind, .name = name, .params = try gpa.dupe(Type, f.params), .ret = f.ret };
        sigs_built += 1;
    }

    // Snapshot the struct table into owned `Layout`s (the scratch `t.structs` is
    // freed by the `defer` above). Incremental-free errdefer on partial failure.
    const layouts = try gpa.alloc(Layout, t.structs.items.len);
    errdefer gpa.free(layouts);
    var layouts_built: usize = 0;
    errdefer for (layouts[0..layouts_built]) |l| {
        gpa.free(l.name);
        for (l.field_names) |fn_| gpa.free(fn_);
        gpa.free(l.field_names);
        gpa.free(l.field_types);
        gpa.free(l.offsets);
    };
    for (t.structs.items, 0..) |s, i| {
        const fnames = try gpa.alloc([]const u8, s.field_names.len);
        var dn: usize = 0;
        errdefer {
            for (fnames[0..dn]) |x| gpa.free(x);
            gpa.free(fnames);
        }
        for (s.field_names, 0..) |nm, j| {
            fnames[j] = try gpa.dupe(u8, nm);
            dn += 1;
        }
        layouts[i] = .{
            .name = try gpa.dupe(u8, s.name),
            .field_names = @ptrCast(fnames),
            .field_types = try gpa.dupe(Type, s.field_types),
            .offsets = try gpa.dupe(u32, s.offsets),
            .size = s.size,
            .@"align" = s.@"align",
        };
        layouts_built += 1;
    }

    return Result{
        .node_types = node_types,
        .diags = try t.diags.toOwnedSlice(gpa),
        .owned_msgs = try t.owned_msgs.toOwnedSlice(gpa),
        .sigs = sigs,
        .layouts = layouts,
    };
}

fn run(t: *Typecheck) !void {
    if (t.tree.nodes.len == 0) return;
    const prog = t.tree.nodes[Ast.root(t.tree.nodes)];
    if (prog.tag != .program) return; // defensive

    const decl_nodes = Ast.rangeSlice(t.tree, prog.lhs);

    // Pass A0a: register every struct name → id (duplicate names diagnosed). Ids
    // are assigned in declaration order among the struct decls.
    for (decl_nodes) |decl_idx| {
        const decl = t.tree.nodes[decl_idx];
        if (decl.tag != .struct_decl) continue;
        const name = t.nameText(decl.main_token);
        // A struct named after a builtin scalar/str type shadows nothing usable
        // (typeFromNode checks the builtin map FIRST), so reject it rather than
        // register a permanently-unreachable type.
        if (type_names.get(name) != null) {
            try t.emitFmt(t.byteOf(decl.main_token), "struct '{s}' shadows a builtin type", .{name});
            continue;
        }
        if (t.struct_map.get(name) != null) {
            try t.emitFmt(t.byteOf(decl.main_token), "duplicate struct declaration '{s}'", .{name});
            continue;
        }
        const id: u32 = @intCast(t.structs.items.len);
        try t.structs.append(t.gpa, .{ .decl_node = decl_idx, .name = name });
        try t.struct_map.put(t.gpa, name, id);
    }
    // Pass A0b: lay out each struct (visiting-guard catches recursive cycles).
    for (0..t.structs.items.len) |id| {
        try t.layoutStruct(@intCast(id));
    }

    // Pass A: decode every function signature (forward refs resolve fine since
    // calls look the signature up by index, not by walk order). Indices here
    // match `Resolve`'s `func` indices (both number fns in source order).
    for (decl_nodes) |fn_idx| {
        const decl = t.tree.nodes[fn_idx];
        if (decl.tag != .fn_decl) continue;
        const proto = Ast.protoAt(t.tree, decl.lhs);
        const params = try t.gpa.alloc(Type, proto.params.len);
        for (proto.params, 0..) |param_idx, i| {
            const param = t.tree.nodes[param_idx];
            const pty = t.typeFromNode(param.lhs);
            if (pty.kind == .unit) {
                try t.emitFmt(t.byteOf(param.main_token), "parameter '{s}' cannot have type ()", .{t.nameText(param.main_token)});
                params[i] = .invalid; // poison so call-arg checks don't cascade
            } else {
                params[i] = pty;
            }
        }
        const ret: Type = if (proto.ret_type == Ast.none) Type.unit else t.typeFromNode(proto.ret_type);
        try t.fns.append(t.gpa, .{ .decl_node = fn_idx, .params = params, .ret = ret });
    }

    // Synthetic `print(str) -> ()` builtin. Appended AFTER the user-fn loop so
    // its index == user_fn_count, matching Resolve's seeding order (which assigns
    // print the same index). `decl_node = Ast.none` flags it as bodyless so
    // checkFn skips it (Pass B). Codegen emits a hand-written body at this sym.
    {
        const params = try t.gpa.dupe(Type, &.{.str});
        try t.fns.append(t.gpa, .{ .decl_node = Ast.none, .params = params, .ret = .unit });
    }

    // Pass B: check each function body (skip the synthetic, bodyless builtins).
    for (t.fns.items) |f| {
        if (f.decl_node == Ast.none) continue;
        try t.checkFn(f);
    }
}

fn checkFn(t: *Typecheck, f: FnSym) !void {
    const decl = t.tree.nodes[f.decl_node];
    const proto = Ast.protoAt(t.tree, decl.lhs);

    // Rebuild the per-function slot→type table. Parameters get slots 0..N first
    // (the resolver declares them first), then `:=` locals as we encounter them.
    t.slot_types.clearRetainingCapacity();
    for (f.params) |pty| try t.slot_types.append(t.gpa, pty);
    t.cur_ret = f.ret;

    // A function body is a block; a non-unit fn wants its trailing expression to
    // supply the value. A unit fn checks its body in statement context.
    const want_value = (f.ret.kind != .unit and f.ret.kind != .invalid);
    const body_ty = try t.checkBlock(decl.rhs, want_value);

    // A non-unit function must produce a value on every path: either all paths
    // structurally return, OR the body's trailing expression has the declared
    // type (no explicit `return` needed). An else-less `if` or a `while`
    // never guarantees a return, and a trailing non-expression item is unit.
    if (want_value and !blockReturns(t, decl.rhs)) {
        if (body_ty.kind == .invalid) {
            // poison already reported elsewhere; no extra diagnostic
        } else if (Type.eql(body_ty, f.ret) or body_ty.kind == .never) {
            // trailing expression supplies the value — OK (or it is a `never`
            // expression, e.g. a break-less `loop`, that can never fall through).
        } else {
            try t.emitFmt(t.byteOf(decl.main_token), "function '{s}' must return {s} but may fall off the end", .{ t.nameText(decl.main_token), t.typeName(f.ret) });
        }
    }
    _ = proto;
}

/// Structural definite-return: a block returns on every path iff its last
/// statement does (earlier statements can't satisfy the requirement, since
/// anything after a guaranteed return would be unreachable).
fn blockReturns(t: *const Typecheck, block_idx: Ast.Index) bool {
    const stmts = Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return false;
    return stmtReturns(t, stmts[stmts.len - 1]);
}

/// Whether a single statement guarantees a return on every path through it.
fn stmtReturns(t: *const Typecheck, stmt_idx: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        .return_stmt => true,
        // A statement-position block/if is parsed wrapped in an `expr_stmt`; unwrap
        // it so a trailing diverging bare block (`{ return 5 }`) or parenthesized
        // value-if satisfies definite-return and a divergent arm merges correctly.
        .expr_stmt => stmtReturns(t, stmt.lhs),
        .block => blockReturns(t, stmt_idx),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            // An else-less `if` can be skipped, so it never guarantees a return.
            if (h.else_node == Ast.none) break :blk false;
            const then_ok = blockReturns(t, h.then_block);
            const else_ok = if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtReturns(t, h.else_node)
            else
                blockReturns(t, h.else_node);
            break :blk then_ok and else_ok;
        },
        // Conservative: a `while` may never execute, so it can't guarantee a
        // return (`while true { return }` is rejected — acceptable for now).
        .while_stmt => false,
        // A `loop` diverges (returns/never-falls-through) iff it has NO `break`
        // targeting it: the only ways out are `return` or an outer construct.
        .loop_expr => loopDiverges(t, stmt_idx),
        // A labeled wrapper is transparent for definite-return: it returns iff its
        // inner construct does (a labeled bare block via its trailing stmt).
        .labeled => stmtReturns(t, t.tree.nodes[stmt_idx].lhs),
        .for_stmt, .break_stmt, .continue_stmt => false,
        else => false,
    };
}

/// A `loop` diverges (control never falls past it) iff no `break` targets it.
/// `target` is the loop's own node — a break "targets THIS loop" iff it is a
/// BARE break encountered at this loop's nesting level OR a labeled break whose
/// resolved `.label` target == `target`. For the bare case the scan descends
/// `if`/`block`/`expr_stmt` but NOT a nested loop (a bare break there binds to
/// the inner loop); for the labeled case it MUST descend nested loops/labeled
/// wrappers to find a `break @target` buried inside.
fn loopDiverges(t: *const Typecheck, loop_idx: Ast.Index) bool {
    return !blockHasBreak(t, t.tree.nodes[loop_idx].lhs, loop_idx);
}

fn blockHasBreak(t: *const Typecheck, block_idx: Ast.Index, target: Ast.Index) bool {
    for (Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs)) |s| {
        if (stmtHasBreak(t, s, target)) return true;
    }
    return false;
}

/// Whether `stmt` contains a `break` that targets the loop node `target`.
fn stmtHasBreak(t: *const Typecheck, stmt_idx: Ast.Index, target: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        // A bare break (no label) binds to the innermost loop — counts only when
        // `target` IS the innermost loop, i.e. the bare break is found before any
        // nested loop swallows it (the nested-loop arms below stop the descent for
        // bare breaks). A labeled break counts iff its resolved target matches.
        .break_stmt => if (t.resolutions[stmt_idx] == .label)
            t.resolutions[stmt_idx].label == target
        else
            true,
        .expr_stmt => stmtHasBreak(t, stmt.lhs, target),
        .block => blockHasBreak(t, stmt_idx, target),
        // A labeled wrapper is transparent: descend its inner construct (a
        // `break @target` may live inside a nested labeled loop).
        .labeled => stmtHasBreak(t, stmt.lhs, target),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            if (blockHasBreak(t, h.then_block, target)) break :blk true;
            if (h.else_node == Ast.none) break :blk false;
            break :blk if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtHasBreak(t, h.else_node, target)
            else
                blockHasBreak(t, h.else_node, target);
        },
        // A nested loop/for/while swallows BARE breaks, but a `break @target`
        // buried inside it still targets `target` — so descend its body and only
        // count labeled breaks that name `target`.
        .loop_expr => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        .while_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].rhs, target),
        .for_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        else => false,
    };
}

/// Scan a nested loop's body for a labeled `break @target` (bare breaks here bind
/// to the nested loop, so they do NOT count for `target`).
fn nestedHasLabeledBreak(t: *const Typecheck, block_idx: Ast.Index, target: Ast.Index) bool {
    for (Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs)) |s| {
        if (stmtHasLabeledBreak(t, s, target)) return true;
    }
    return false;
}

fn stmtHasLabeledBreak(t: *const Typecheck, stmt_idx: Ast.Index, target: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        .break_stmt => t.resolutions[stmt_idx] == .label and t.resolutions[stmt_idx].label == target,
        .expr_stmt => stmtHasLabeledBreak(t, stmt.lhs, target),
        .block => nestedHasLabeledBreak(t, stmt_idx, target),
        .labeled => stmtHasLabeledBreak(t, stmt.lhs, target),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            if (nestedHasLabeledBreak(t, h.then_block, target)) break :blk true;
            if (h.else_node == Ast.none) break :blk false;
            break :blk if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtHasLabeledBreak(t, h.else_node, target)
            else
                nestedHasLabeledBreak(t, h.else_node, target);
        },
        .loop_expr => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        .while_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].rhs, target),
        .for_stmt => nestedHasLabeledBreak(t, t.tree.nodes[stmt_idx].lhs, target),
        else => false,
    };
}

/// Does this block's final statement DIVERGE — i.e. control never falls past it,
/// so when used as a value-merge arm the arm is `never`? Distinct from
/// `blockReturns` ("guarantees a function return"): a `break`/`continue` diverts
/// control out of the innermost loop (genuinely divergent for the merge) but does
/// NOT guarantee a function return. The merge consumer needs divergence; the
/// definite-return check needs return-guarantee. Do not conflate them.
fn blockDiverges(t: *const Typecheck, block_idx: Ast.Index) bool {
    const stmts = Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) return false;
    return stmtDiverges(t, stmts[stmts.len - 1]);
}

fn stmtDiverges(t: *const Typecheck, stmt_idx: Ast.Index) bool {
    const stmt = t.tree.nodes[stmt_idx];
    return switch (stmt.tag) {
        .return_stmt, .break_stmt, .continue_stmt => true,
        .expr_stmt => stmtDiverges(t, stmt.lhs),
        .block => blockDiverges(t, stmt_idx),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            if (h.else_node == Ast.none) break :blk false;
            const then_ok = blockDiverges(t, h.then_block);
            const else_ok = if (t.tree.nodes[h.else_node].tag == .if_stmt)
                stmtDiverges(t, h.else_node)
            else
                blockDiverges(t, h.else_node);
            break :blk then_ok and else_ok;
        },
        .while_stmt => false,
        .loop_expr => loopDiverges(t, stmt_idx),
        .labeled => labeledDiverges(t, stmt_idx),
        .for_stmt => false,
        else => false,
    };
}

/// A `labeled` wrapper diverges (control never falls past it) when its inner
/// construct does. For a labeled loop this is `loopDiverges` (break-less). For a
/// labeled bare block it is: the block's trailing stmt diverges AND no `break`
/// targets the block (a targeting break would exit normally, past the block).
fn labeledDiverges(t: *const Typecheck, idx: Ast.Index) bool {
    const inner_idx = t.tree.nodes[idx].lhs;
    const inner = t.tree.nodes[inner_idx];
    return switch (inner.tag) {
        .loop_expr => loopDiverges(t, inner_idx),
        .while_stmt, .for_stmt => false,
        .block => blockDiverges(t, inner_idx) and !blockHasBreak(t, inner_idx, inner_idx),
        else => false,
    };
}

/// Type a block. Every non-final statement is checked for effect. The block's
/// VALUE is the final item's type IFF that item is an expression carrier and a
/// value is wanted; otherwise the block is `()`. When `!want_value`, a trailing
/// if/block is checked in STATEMENT context (an else-less trailing `if` stays a
/// `()` statement, never "value-if requires else"). Records the type on the node.
fn checkBlock(t: *Typecheck, block_idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
    const stmts = Ast.rangeSlice(t.tree, t.tree.nodes[block_idx].lhs);
    if (stmts.len == 0) {
        t.node_types[block_idx] = .unit;
        return .unit;
    }
    for (stmts[0 .. stmts.len - 1]) |s| try t.checkStmt(s); // effect only
    const last = stmts[stmts.len - 1];
    const last_n = t.tree.nodes[last];
    var bt: Type = .unit;
    if (last_n.tag == .expr_stmt) {
        bt = try t.typeOf(last_n.lhs);
        t.node_types[last] = bt; // the expr_stmt carries the value type
    } else if (want_value and (last_n.tag == .if_stmt or last_n.tag == .block)) {
        bt = try t.typeOf(last); // value context: validates + types + memoizes
    } else {
        try t.checkStmt(last); // statement context (incl. trailing else-less if)
    }
    t.node_types[block_idx] = bt;
    return bt;
}

fn checkStmt(t: *Typecheck, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const stmt = t.tree.nodes[stmt_idx];
    switch (stmt.tag) {
        .var_decl => {
            const ty = try t.typeOf(stmt.lhs);
            t.node_types[stmt_idx] = ty;
            // Record the new local's type at its slot. The resolver bound the
            // var_decl node to a `.local` slot; append/extend slot_types to fit.
            if (t.resolutions[stmt_idx] == .local) {
                const slot = t.resolutions[stmt_idx].local;
                try t.setSlot(slot, ty);
            }
            if (ty.kind == .unit) {
                try t.emitFmt(t.byteOf(stmt.main_token), "cannot bind () to '{s}'", .{t.nameText(stmt.main_token)});
            }
        },
        .assign => {
            const rhs = try t.typeOf(stmt.rhs);
            const target = t.tree.nodes[stmt.lhs];
            // The place is either an identifier (a local slot) or a field path.
            const lhs: Type = switch (target.tag) {
                .identifier => if (t.resolutions[stmt.lhs] == .local)
                    t.slotType(t.resolutions[stmt.lhs].local)
                else
                    .invalid,
                .field_access => try t.typeOf(stmt.lhs),
                else => .invalid,
            };
            t.node_types[stmt.lhs] = lhs;
            if (lhs.kind != .invalid and rhs.kind != .invalid and !Type.eql(lhs, rhs)) {
                try t.emitFmt(t.byteOf(target.main_token), "cannot assign {s} to variable of type {s}", .{ t.typeName(rhs), t.typeName(lhs) });
            }
        },
        .return_stmt => {
            const ty: Type = if (stmt.lhs == Ast.none) Type.unit else try t.typeOf(stmt.lhs);
            // Skip when either side is poison so an unknown return type (already
            // reported) doesn't trigger a spurious second diagnostic here. A
            // `never`-typed operand (e.g. `return x` where x binds a break-less
            // loop) is also fine: control never actually reaches the return, so
            // `never` unifies with any declared type — mirrors the trailing-expr
            // body check above.
            if (ty.kind != .invalid and ty.kind != .never and t.cur_ret.kind != .invalid and !Type.eql(ty, t.cur_ret)) {
                try t.emitFmt(t.byteOf(stmt.main_token), "return type {s} does not match declared {s}", .{ t.typeName(ty), t.typeName(t.cur_ret) });
            }
        },
        .expr_stmt => _ = try t.typeOf(stmt.lhs),
        .block => _ = try t.checkBlock(stmt_idx, false),
        .if_stmt => {
            const ct = try t.typeOf(stmt.lhs);
            if (ct.kind != .invalid and ct.kind != .bool)
                try t.emit(t.byteOf(t.tree.nodes[stmt.lhs].main_token), "if condition must be bool");
            const h = Ast.ifHeaderAt(t.tree, stmt.rhs);
            _ = try t.checkBlock(h.then_block, false);
            if (h.else_node != Ast.none) {
                if (t.tree.nodes[h.else_node].tag == .if_stmt)
                    try t.checkStmt(h.else_node)
                else
                    _ = try t.checkBlock(h.else_node, false);
            }
        },
        .while_stmt => try t.checkWhile(stmt_idx, null),
        .for_stmt => try t.checkFor(stmt_idx, null),
        .labeled => _ = try t.checkLabeled(stmt_idx, false),
        .break_stmt => {
            const ctx = t.targetCtx(stmt_idx) orelse {
                // No matching context: a bare break with an empty stack ("outside a
                // loop"); a labeled break is reported by resolve as undefined.
                if (t.resolutions[stmt_idx] != .label)
                    try t.emit(t.byteOf(stmt.main_token), "break outside of a loop");
                if (stmt.lhs != Ast.none) _ = try t.typeOf(stmt.lhs);
                return;
            };
            if (stmt.lhs == Ast.none) {
                ctx.saw_bare_break = true;
                if (ctx.is_value) ctx.join = try t.merge(stmt.main_token, ctx.join, Type.unit);
            } else {
                const vt = try t.typeOf(stmt.lhs);
                if (!ctx.is_value) {
                    if (vt.kind != .invalid and vt.kind != .unit)
                        try t.emit(t.byteOf(stmt.main_token), "cannot break with a value out of a while/for loop");
                } else {
                    ctx.saw_value_break = true;
                    ctx.join = try t.merge(stmt.main_token, ctx.join, vt);
                }
            }
        },
        .continue_stmt => {
            const ctx = t.targetCtx(stmt_idx) orelse {
                if (t.resolutions[stmt_idx] != .label)
                    try t.emit(t.byteOf(stmt.main_token), "continue outside of a loop");
                return;
            };
            // `continue` is meaningful only on a loop; a labeled bare block is not.
            if (ctx.kind == .labeled_block)
                try t.emit(t.byteOf(stmt.main_token), "cannot continue a labeled block (not a loop)");
        },
        else => _ = try t.typeOf(stmt_idx),
    }
}

/// Select the context a `break`/`continue` targets. A labeled break/continue
/// (resolved to `.label`) scans the stack top-down for the matching construct
/// node (the NAMED target, not the innermost). A bare break/continue scans
/// top-down for the innermost LOOP (skipping labeled bare blocks).
fn targetCtx(t: *Typecheck, stmt_idx: Ast.Index) ?*LoopCtx {
    const items = t.loop_stack.items;
    if (t.resolutions[stmt_idx] == .label) {
        const target = t.resolutions[stmt_idx].label;
        var i = items.len;
        while (i > 0) {
            i -= 1;
            if (items[i].construct_node == target) return &items[i];
        }
        return null;
    }
    var i = items.len;
    while (i > 0) {
        i -= 1;
        if (items[i].kind != .labeled_block) return &items[i];
    }
    return null;
}

/// Check a `while` statement; `label` is its label name (or null). A `()` loop.
fn checkWhile(t: *Typecheck, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = t.tree.nodes[stmt_idx];
    const ct = try t.typeOf(stmt.lhs);
    if (ct.kind != .invalid and ct.kind != .bool)
        try t.emit(t.byteOf(t.tree.nodes[stmt.lhs].main_token), "while condition must be bool");
    try t.loop_stack.append(t.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
    _ = try t.checkBlock(stmt.rhs, false);
    _ = t.loop_stack.pop();
}

/// Check a `for` statement; `label` is its label name (or null). A `()` loop.
fn checkFor(t: *Typecheck, stmt_idx: Ast.Index, label: ?[]const u8) error{OutOfMemory}!void {
    const stmt = t.tree.nodes[stmt_idx];
    const h = Ast.forHeaderAt(t.tree, stmt.rhs);
    const lo = try t.typeOf(h.lo);
    const hi = try t.typeOf(h.hi);
    if (lo.kind != .invalid and lo.kind != .int)
        try t.emit(t.byteOf(t.tree.nodes[h.lo].main_token), "for range bounds must be int");
    if (hi.kind != .invalid and hi.kind != .int)
        try t.emit(t.byteOf(t.tree.nodes[h.hi].main_token), "for range bounds must be int");
    if (t.resolutions[stmt_idx] == .local) try t.setSlot(t.resolutions[stmt_idx].local, Type.int);
    try t.loop_stack.append(t.gpa, .{ .kind = .while_for, .label = label, .construct_node = stmt_idx, .is_value = false, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
    _ = try t.checkBlock(stmt.lhs, false);
    _ = t.loop_stack.pop();
}

/// Check a `labeled` wrapper. The inner construct is typed through its label-
/// aware variant; the wrapper's type is the inner's type (the loop value, `()`
/// for while/for, or the labeled-block value). `want_value` flows to the block
/// case so a statement-context labeled bare block stays `()`-discarded.
fn checkLabeled(t: *Typecheck, idx: Ast.Index, want_value: bool) error{OutOfMemory}!Type {
    const n = t.tree.nodes[idx];
    const label = t.nameText(n.main_token);
    const inner = t.tree.nodes[n.lhs];
    const ty: Type = switch (inner.tag) {
        .block => try t.typeOfLabeledBlock(n.lhs, label, want_value),
        .loop_expr => try t.typeOfLoop(n.lhs, inner, label),
        .while_stmt => blk: {
            try t.checkWhile(n.lhs, label);
            break :blk Type.unit;
        },
        .for_stmt => blk: {
            try t.checkFor(n.lhs, label);
            break :blk Type.unit;
        },
        else => Type.invalid,
    };
    t.node_types[idx] = ty;
    return ty;
}

/// Type a labeled BARE BLOCK used as a value: its value is the merge of its
/// trailing-expression value and every `break @label <expr>` targeting it
/// (`never` if it can only diverge). Lowers like a value-loop minus the back-edge.
fn typeOfLabeledBlock(t: *Typecheck, block_idx: Ast.Index, label: []const u8, want_value: bool) error{OutOfMemory}!Type {
    try t.loop_stack.append(t.gpa, .{ .kind = .labeled_block, .label = label, .construct_node = block_idx, .is_value = true, .join = .never, .saw_value_break = false, .saw_bare_break = false });
    const fall = try t.checkBlock(block_idx, want_value);
    const ctx = t.loop_stack.pop().?;
    // The trailing-expr value is unreachable iff the block's last statement
    // diverges; in that case the block's value comes entirely from its breaks.
    const ft: Type = if (blockDiverges(t, block_idx)) Type.never else fall;
    return t.merge(t.tree.nodes[block_idx].main_token, ft, ctx.join);
}

/// Infer (and memoize) the type of an expression node.
fn typeOf(t: *Typecheck, node_idx: Ast.Index) error{OutOfMemory}!Type {
    if (node_idx == Ast.none) return .invalid;
    const n = t.tree.nodes[node_idx];
    const ty: Type = switch (n.tag) {
        .literal_number => Type.int,
        .literal_bool => Type.@"bool",
        .literal_string => Type.str,
        .identifier => switch (t.resolutions[node_idx]) {
            .local => |slot| t.slotType(slot),
            .func => blk: {
                try t.emitFmt(t.byteOf(n.main_token), "function '{s}' is not a value", .{t.nameText(n.main_token)});
                break :blk Type.invalid;
            },
            .unresolved => blk: {
                // Resolve quietly skips a struct-named identifier (it expects this
                // to be a struct type-name in a position it doesn't bind, or a
                // positional `Point(...)` callee it diagnoses elsewhere). A BARE
                // struct name used as a value (`q := P`, `P.x`) reaches here with
                // no diagnostic — report it so it never escapes to codegen.
                if (t.struct_map.get(t.nameText(n.main_token)) != null)
                    try t.emitFmt(t.byteOf(n.main_token), "type '{s}' is not a value", .{t.nameText(n.main_token)});
                break :blk Type.invalid;
            },
            .label => Type.invalid, // never on an identifier node (break/continue only)
        },
        .unary => blk: {
            const operand = try t.typeOf(n.lhs);
            if (operand.kind == .invalid) break :blk Type.invalid;
            const op = t.tokens[n.main_token].tag;
            switch (op) {
                .minus => {
                    if (operand.kind == .int) break :blk Type.int;
                    try t.emit(t.byteOf(n.main_token), "operand of '-' must be int");
                },
                .bang => {
                    if (operand.kind == .bool) break :blk Type.@"bool";
                    try t.emit(t.byteOf(n.main_token), "operand of '!' must be bool");
                },
                else => {},
            }
            break :blk Type.invalid;
        },
        .binary => blk: {
            const lt = try t.typeOf(n.lhs);
            const rt = try t.typeOf(n.rhs);
            if (lt.kind == .invalid or rt.kind == .invalid) break :blk Type.invalid;
            const op = t.tokens[n.main_token].tag;
            const op_text = t.tokens[n.main_token].text(t.source);
            switch (op) {
                .plus, .minus, .star, .slash => {
                    if (lt.kind == .int and rt.kind == .int) break :blk Type.int;
                    try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                },
                .lt, .lt_eq, .gt, .gt_eq => {
                    if (lt.kind == .int and rt.kind == .int) break :blk Type.@"bool";
                    try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must be int", .{op_text});
                },
                .eq_eq, .bang_eq => {
                    if (Type.eql(lt, rt) and (lt.kind == .int or lt.kind == .bool)) break :blk Type.@"bool";
                    if (Type.eql(lt, rt) and lt.kind == .str) {
                        // Same type, but str comparison isn't supported — say so,
                        // rather than the misleading "must have the same type".
                        try t.emitFmt(t.byteOf(n.main_token), "'{s}' on str is unsupported", .{op_text});
                    } else {
                        try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must have the same type", .{op_text});
                    }
                },
                .amp_amp, .pipe_pipe => {
                    if (lt.kind == .bool and rt.kind == .bool) break :blk Type.@"bool";
                    try t.emitFmt(t.byteOf(n.main_token), "operands of '{s}' must be bool", .{op_text});
                },
                else => {},
            }
            break :blk Type.invalid;
        },
        .call => try t.typeOfCall(node_idx, n),
        .struct_init => try t.typeOfStructInit(node_idx, n),
        .field_access => try t.typeOfFieldAccess(node_idx, n),
        .literal_unit => Type.unit,
        .block => try t.checkBlock(node_idx, true),
        .if_stmt => try t.typeOfIf(node_idx, n),
        .loop_expr => return t.typeOfLoop(node_idx, n, null), // sets node_types itself
        .labeled => return t.checkLabeled(node_idx, true), // sets node_types itself
        else => Type.invalid,
    };
    t.node_types[node_idx] = ty;
    return ty;
}

/// Type a `Name { field: value, ... }` construction. The type name must be a
/// known struct; every declared field must be supplied exactly once; each
/// value's type must match the declared field type. Result is the struct type.
fn typeOfStructInit(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    _ = node_idx;
    const name = t.nameText(t.tree.nodes[n.lhs].main_token);
    const id = t.struct_map.get(name) orelse {
        for (Ast.rangeSlice(t.tree, n.rhs)) |fi| _ = try t.typeOf(t.tree.nodes[fi].lhs);
        try t.emitFmt(t.byteOf(t.tree.nodes[n.lhs].main_token), "unknown struct type '{s}'", .{name});
        return .invalid;
    };
    const sym = t.structs.items[id];
    const inits = Ast.rangeSlice(t.tree, n.rhs);

    // Track which declared fields are supplied (for missing/duplicate checks).
    var seen = try t.gpa.alloc(bool, sym.field_names.len);
    defer t.gpa.free(seen);
    @memset(seen, false);

    for (inits) |fi_idx| {
        const fi = t.tree.nodes[fi_idx];
        const fname = t.nameText(fi.main_token);
        const vt = try t.typeOf(fi.lhs);
        // Find the declared field by name.
        var found: ?usize = null;
        for (sym.field_names, 0..) |dn, j| {
            if (std.mem.eql(u8, dn, fname)) {
                found = j;
                break;
            }
        }
        if (found) |j| {
            if (seen[j]) {
                try t.emitFmt(t.byteOf(fi.main_token), "duplicate field '{s}' in '{s}'", .{ fname, name });
            }
            seen[j] = true;
            const fty = sym.field_types[j];
            if (vt.kind != .invalid and fty.kind != .invalid and !Type.eql(vt, fty)) {
                try t.emitFmt(t.byteOf(fi.main_token), "field '{s}': expected {s}, got {s}", .{ fname, t.typeName(fty), t.typeName(vt) });
            }
        } else {
            try t.emitFmt(t.byteOf(fi.main_token), "unknown field '{s}' in '{s}'", .{ fname, name });
        }
    }
    for (sym.field_names, 0..) |dn, j| {
        if (!seen[j]) try t.emitFmt(t.byteOf(n.main_token), "missing field '{s}' in '{s}'", .{ dn, name });
    }
    return Type.structT(id);
}

/// Type a `recv.field` access. The receiver must be a struct; the field must
/// exist. Yields the field type. Poison receivers stay silent (already reported).
fn typeOfFieldAccess(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    _ = node_idx;
    const base = try t.typeOf(n.lhs);
    if (base.kind == .invalid) return .invalid;
    if (!base.isStruct()) {
        try t.emitFmt(t.byteOf(n.main_token), "cannot access field '{s}' of non-struct type {s}", .{ t.nameText(n.main_token), t.typeName(base) });
        return .invalid;
    }
    const sym = t.structs.items[base.struct_id];
    const fname = t.nameText(n.main_token);
    for (sym.field_names, 0..) |dn, j| {
        if (std.mem.eql(u8, dn, fname)) return sym.field_types[j];
    }
    try t.emitFmt(t.byteOf(n.main_token), "no field '{s}' in struct '{s}'", .{ fname, sym.name });
    return .invalid;
}

/// Type an `if`/`else` used as a VALUE expression. Both arms must agree:
/// the merge is the one phi/join. An else-less value-if is a type error
/// ("value-if requires else"). A fully-diverging arm (returns on all paths) is
/// compatible with the other arm's concrete type (its value side is unreachable).
fn typeOfIf(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    _ = node_idx;
    const ct = try t.typeOf(n.lhs);
    if (ct.kind != .invalid and ct.kind != .bool)
        try t.emit(t.byteOf(t.tree.nodes[n.lhs].main_token), "if condition must be bool");
    const h = Ast.ifHeaderAt(t.tree, n.rhs);
    if (h.else_node == Ast.none) {
        try t.emit(t.byteOf(n.main_token), "value-if requires else");
        _ = try t.checkBlock(h.then_block, false); // validate the arm anyway
        return .invalid;
    }
    const then_ty0 = try t.checkBlock(h.then_block, true);
    const then_ty: Type = if (blockDiverges(t, h.then_block)) Type.never else then_ty0;
    var else_ty: Type = undefined;
    if (t.tree.nodes[h.else_node].tag == .if_stmt) {
        const e0 = try t.typeOfIf(h.else_node, t.tree.nodes[h.else_node]); // else-if ladder
        else_ty = if (stmtDiverges(t, h.else_node)) Type.never else e0;
    } else {
        const e0 = try t.checkBlock(h.else_node, true);
        else_ty = if (blockDiverges(t, h.else_node)) Type.never else e0;
    }
    return t.merge(n.main_token, then_ty, else_ty);
}

/// Type a `loop` used as a value expression. Its body is checked in statement
/// context; each value-break fills the loop's `join`. A loop with NO break at
/// all is `never` (it can only exit via `return` or an outer construct).
fn typeOfLoop(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node, label: ?[]const u8) error{OutOfMemory}!Type {
    try t.loop_stack.append(t.gpa, .{ .kind = .loop, .label = label, .construct_node = node_idx, .is_value = true, .join = Type.never, .saw_value_break = false, .saw_bare_break = false });
    _ = try t.checkBlock(n.lhs, false); // body in statement ctx; breaks fill join
    const ctx = t.loop_stack.pop().?;
    const ty: Type = if (!ctx.saw_value_break and !ctx.saw_bare_break) Type.never else ctx.join;
    t.node_types[node_idx] = ty;
    return ty;
}

/// Merge two value types into a single join. `never` is the bottom type: it
/// unifies with anything (a `never` expression never produces a value), so
/// `never + T = T` and `never + never = never`. No implicit coercion otherwise:
/// a mismatch is a type error.
fn merge(t: *Typecheck, at: u32, a: Type, b: Type) error{OutOfMemory}!Type {
    if (a.kind == .invalid or b.kind == .invalid) return .invalid; // poison absorbs
    if (a.kind == .never) return b; // covers never+never → never
    if (b.kind == .never) return a;
    if (Type.eql(a, b)) return a; // agreement → one phi type
    try t.emitFmt(t.byteOf(at), "branches yield different types ({s} vs {s})", .{ t.typeName(a), t.typeName(b) });
    return .invalid; // mismatch, no coercion
}

fn typeOfCall(t: *Typecheck, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Type {
    _ = node_idx;
    const callee_res = t.resolutions[n.lhs];
    if (callee_res != .func) {
        // Type the args anyway so their own errors surface, then poison.
        for (Ast.rangeSlice(t.tree, n.rhs)) |arg| _ = try t.typeOf(arg);
        if (callee_res == .local) {
            try t.emit(t.byteOf(n.main_token), "called value is not a function");
        } else if (t.tree.nodes[n.lhs].tag == .identifier) {
            // A struct-named callee `Point(1,2)` is positional construction, which
            // we reject — point at named construction instead.
            const cname = t.nameText(t.tree.nodes[n.lhs].main_token);
            if (t.struct_map.get(cname) != null)
                try t.emitFmt(t.byteOf(n.main_token), "use named construction '{s} {{ ... }}', not '{s}(...)'", .{ cname, cname });
        }
        // `.unresolved` was already reported by resolve.
        return .invalid;
    }
    const f = t.fns.items[callee_res.func];
    const args = Ast.rangeSlice(t.tree, n.rhs);
    if (args.len != f.params.len) {
        for (args) |arg| _ = try t.typeOf(arg);
        try t.emitFmt(t.byteOf(n.main_token), "expected {d} argument(s), got {d}", .{ f.params.len, args.len });
        return f.ret;
    }
    for (args, f.params, 0..) |arg, pty, i| {
        const at = try t.typeOf(arg);
        // Skip when either side is poison (e.g. a parameter whose type failed to
        // resolve) so we don't cascade a spurious "expected invalid" message.
        if (at.kind != .invalid and pty.kind != .invalid and !Type.eql(at, pty)) {
            try t.emitFmt(t.byteOf(t.tree.nodes[arg].main_token), "argument {d}: expected {s}, got {s}", .{ i + 1, t.typeName(pty), t.typeName(at) });
        }
    }
    return f.ret;
}

/// Map a type-reference node (an `identifier`, or a `literal_unit` for `()`) to
/// a `Type`.
fn typeFromNode(t: *Typecheck, type_node: Ast.Index) Type {
    if (type_node == Ast.none) return Type.unit;
    if (t.tree.nodes[type_node].tag == .literal_unit) return Type.unit; // explicit `-> ()` / `p: ()`
    const tok = t.tree.nodes[type_node].main_token;
    const name = t.nameText(tok);
    if (type_names.get(name)) |b| return b;
    if (t.struct_map.get(name)) |id| return Type.structT(id);
    t.emitFmt(t.byteOf(tok), "unknown type '{s}'", .{name}) catch {};
    return .invalid;
}

/// Lay out struct `id`: field offsets in declaration order with natural
/// alignment, size = aligned total, align = max field align. A `laying` field
/// of struct type means a cycle (direct or indirect) → infinite size, poisoned.
fn layoutStruct(t: *Typecheck, id: u32) error{OutOfMemory}!void {
    if (t.structs.items[id].state == .done) return;
    t.structs.items[id].state = .laying;

    const decl = t.tree.nodes[t.structs.items[id].decl_node];
    const field_nodes = Ast.rangeSlice(t.tree, decl.lhs);
    const n = field_nodes.len;

    // An empty struct lays out to size 0 — a zero-size aggregate is an ABI/codegen
    // hazard (no eightbytes, a degenerate sret). Reject it with a clean diagnostic.
    var empty_poison = false;
    if (n == 0) {
        try t.emitFmt(t.byteOf(decl.main_token), "empty struct '{s}' is not allowed", .{t.structs.items[id].name});
        empty_poison = true;
    }

    const names = try t.gpa.alloc([]const u8, n);
    errdefer t.gpa.free(names);
    const types = try t.gpa.alloc(Type, n);
    errdefer t.gpa.free(types);
    const offsets = try t.gpa.alloc(u32, n);
    errdefer t.gpa.free(offsets);

    var running: u32 = 0;
    var max_align: u32 = 1;
    var poisoned = false;
    for (field_nodes, 0..) |field_idx, i| {
        const field = t.tree.nodes[field_idx];
        names[i] = t.nameText(field.main_token);
        const fty = t.typeFromNode(field.lhs);
        types[i] = fty;
        var fsize: u32 = 0;
        var falign: u32 = 1;
        if (fty.isStruct()) {
            if (t.structs.items[fty.struct_id].state == .laying) {
                try t.emitFmt(t.byteOf(decl.main_token), "recursive struct '{s}' has infinite size", .{t.structs.items[id].name});
                poisoned = true;
            } else {
                try t.layoutStruct(fty.struct_id);
                fsize = t.structs.items[fty.struct_id].size;
                falign = t.structs.items[fty.struct_id].@"align";
            }
        } else if (fty.kind == .invalid or fty.kind == .unit) {
            // unknown/() field type already diagnosed (or `()` rejected below).
            if (fty.kind == .unit) {
                try t.emitFmt(t.byteOf(field.main_token), "field '{s}' cannot have type ()", .{names[i]});
                poisoned = true;
            }
        } else {
            fsize = scalarSize(fty.kind);
            falign = scalarAlign(fty.kind);
        }
        const off = roundUp(running, falign);
        offsets[i] = off;
        running = off + fsize;
        if (falign > max_align) max_align = falign;
    }

    t.structs.items[id].field_names = names;
    t.structs.items[id].field_types = types;
    t.structs.items[id].offsets = offsets;
    t.structs.items[id].@"align" = max_align;
    t.structs.items[id].size = if (poisoned or empty_poison) 0 else roundUp(running, max_align);
    t.structs.items[id].poisoned = poisoned or empty_poison;
    t.structs.items[id].state = .done;
}

/// Human-readable name of a type (a struct's declared name, else its kind tag).
fn typeName(t: *const Typecheck, ty: Type) []const u8 {
    if (ty.kind == .@"struct" and ty.struct_id < t.structs.items.len)
        return t.structs.items[ty.struct_id].name;
    return @tagName(ty.kind);
}

/// The type at a local slot, or `invalid` if out of range (shouldn't happen).
fn slotType(t: *const Typecheck, slot: u32) Type {
    return if (slot < t.slot_types.items.len) t.slot_types.items[slot] else .invalid;
}

/// Record the type of a local slot, growing the table to fit (slots are dense
/// and monotonically increasing within a function).
fn setSlot(t: *Typecheck, slot: u32, ty: Type) !void {
    while (t.slot_types.items.len <= slot) try t.slot_types.append(t.gpa, .invalid);
    t.slot_types.items[slot] = ty;
}

fn nameText(t: *const Typecheck, tok: u32) []const u8 {
    return t.tokens[tok].text(t.source);
}

fn byteOf(t: *const Typecheck, tok: u32) u32 {
    return t.tokens[tok].start;
}

/// Record a static-literal diagnostic.
fn emit(t: *Typecheck, byte_offset: u32, message: []const u8) !void {
    try t.diags.append(t.gpa, .{ .byte_offset = byte_offset, .message = message });
}

/// Format a data-bearing message, own the buffer, and record a diagnostic.
fn emitFmt(t: *Typecheck, byte_offset: u32, comptime fmt: []const u8, args: anytype) !void {
    const msg = try std.fmt.allocPrint(t.gpa, fmt, args);
    try t.owned_msgs.append(t.gpa, msg);
    try t.diags.append(t.gpa, .{ .byte_offset = byte_offset, .message = msg });
}

const testing = std.testing;
const Lexer = @import("lex.zig");
const Parser = @import("parse.zig");

const Checked = struct {
    tokens: []Token,
    tree: Ast.Tree,
    resolve: Resolve.Result,
    result: Result,
    source: []const u8,

    fn deinit(self: *Checked, gpa: std.mem.Allocator) void {
        self.result.deinit(gpa);
        self.resolve.deinit(gpa);
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }
};

fn checkSource(source: []const u8) !Checked {
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
    const result = try check(gpa, tree, tokens, source, res.resolutions);
    return .{ .tokens = tokens, .tree = tree, .resolve = res, .result = result, .source = source };
}

/// Typecheck a source and return the diagnostic count (and free everything).
fn checkDiagCount(source: []const u8) !usize {
    const gpa = testing.allocator;
    var c = try checkSource(source);
    defer c.deinit(gpa);
    return c.result.diags.len;
}

test "clean program typechecks with zero diagnostics" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn neg(x: int) -> int { return -x }
        \\fn main() {
        \\ x := add(1, 2)
        \\ y := neg(x)
        \\ ok := x == 3
        \\ z := x > y
        \\ return
        \\}
        \\
    ));
}

test "call argument count mismatch" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn add(a: int, b: int) -> int { return a + b }\nfn f() {\n x := add(1)\n return\n}\n",
    ));
}

test "call argument type mismatch (bool to int param)" {
    const gpa = testing.allocator;
    var c = try checkSource(
        "fn add(a: int, b: int) -> int { return a + b }\nfn f() {\n x := add(1, true)\n return\n}\n",
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("argument 2: expected int, got bool", c.result.diags[0].message);
}

test "return type mismatch" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n return true\n}\n",
    ));
}

test "bare return in a non-unit function" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n return\n}\n",
    ));
}

test ":= from a () call is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn g() { return }\nfn f() {\n x := g()\n return\n}\n",
    ));
}

test "assignment type mismatch" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n x := 1\n x = true\n return\n}\n",
    ));
}

test "arithmetic on bool is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n b := true\n y := b + 1\n return\n}\n",
    ));
}

test "comparison yields bool" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n x := 1 < 2\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The var_decl's bound type is bool.
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) try testing.expectEqual(Type.bool, c.result.node_types[i]);
    }
}

test "equality type mismatch is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n b := true == 1\n return\n}\n",
    ));
}

test "one bad identifier yields exactly one typecheck diagnostic (poison)" {
    // `bad` is undeclared (a resolve error); typecheck must NOT add more diags
    // for the binary/var_decl that consume its poison type.
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n x := bad + 1 + 2\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "unknown type name in a parameter" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f(a: nope) {\n return\n}\n",
    ));
}

test "unknown return type does not cascade (poison)" {
    // `nope` unknown → exactly one diagnostic; `return 1` must not add a spurious
    // 'does not match declared invalid'.
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> nope {\n return 1\n}\n",
    ));
}

test "unknown parameter type does not cascade to call arguments" {
    // `nope` unknown → one diagnostic; calling g(1) must not add 'expected invalid'.
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn g(a: nope) -> int {\n return 0\n}\nfn h() {\n x := g(1)\n return\n}\n",
    ));
}

test "() parameter type is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn k(a: ()) -> int {\n return 0\n}\n",
    ));
}

test "non-unit function that falls off the end is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n x := 1\n}\n",
    ));
}

test "non-unit function ending in return is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n return 1\n}\n",
    ));
}

test "if/else where both arms return is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n if n < 0 { return 0 } else { return 1 }\n}\n",
    ));
}

test "else-less if does not guarantee a return" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f(n: int) -> int {\n if n < 0 { return 0 }\n}\n",
    ));
}

test "while loop does not guarantee a return" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f(n: int) -> int {\n while n > 0 { return 1 }\n}\n",
    ));
}

test "else-if ladder where all arms return is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n if n < 0 { return 0 } else if n > 0 { return 1 } else { return 2 }\n}\n",
    ));
}

test "trailing bare block that returns satisfies definite-return" {
    // The `{ return 5 }` is parsed wrapped in an expr_stmt; stmtReturns must unwrap
    // it so the body is seen to return on every path (no fall-off-the-end error).
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n { return 5 }\n}\n",
    ));
}

test "parenthesized diverging if satisfies definite-return" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n (if n > 0 { return 1 } else { return 2 })\n}\n",
    ));
}

test "value-if with an expr_stmt-wrapped divergent arm merges to the live arm's type" {
    // then-arm is a bare block that diverges; the divergent side is unreachable, so
    // the merge must take the else-arm's int — no 'unit vs int' mismatch.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n x := if n > 0 { { return 1 } } else { 2 }\n return x\n}\n",
    ));
}

test "non-bool if condition is rejected with one diagnostic" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n if 1 { return }\n return\n}\n",
    ));
}

test "non-bool while condition is rejected with one diagnostic" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n while 1 { return }\n return\n}\n",
    ));
}

test "&& on int operands is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n b := 1 && 2\n return\n}\n",
    ));
}

test "&& and || on bool operands yield bool" {
    const gpa = testing.allocator;
    var c = try checkSource("fn f() {\n b := (1 < 2) && (3 > 4)\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) try testing.expectEqual(Type.bool, c.result.node_types[i]);
    }
}

test "string literal types as str" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n s := \"hi\"\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The var_decl's bound type is str.
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .var_decl) try testing.expectEqual(Type.str, c.result.node_types[i]);
    }
}

test "print of a string literal typechecks clean" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn main() {\n print(\"hi\")\n return\n}\n",
    ));
}

test "print of an int is an argument-type error" {
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n print(42)\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
    try testing.expectEqualStrings("argument 1: expected str, got int", c.result.diags[0].message);
}

test "print with no arguments is an arity error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn main() {\n print()\n return\n}\n",
    ));
}

test "a bare string literal in int context is a type error" {
    // `x := "x"` makes x a str; `x + 1` then fails (operands of '+' must be int).
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n x := \"x\"\n y := x + 1\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 1), c.result.diags.len);
}

test "value-if without else is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n c := 1\n x := if c > 0 { 1 }\n return x\n}\n",
    ));
}

test "value-if with mismatched arm types is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n c := 1\n x := if c > 0 { 1 } else { \"s\" }\n return x\n}\n",
    ));
}

test "value-if with agreeing arms is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n c := 1\n x := if c > 0 { 1 } else { 2 }\n return x\n}\n",
    ));
}

test "value-if where one arm diverges merges to the other arm type" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f(n: int) -> int {\n x := if n < 0 { return 0 } else { 1 }\n return x\n}\n",
    ));
}

test "trailing expression satisfies a non-unit function" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n 41 + 1\n}\n",
    ));
}

test "trailing expression of the wrong type is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n \"s\"\n}\n",
    ));
}

test "bare block expression takes its trailing type" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n x := { a := 1\n a + 1 }\n return x\n}\n",
    ));
}

test "unit-returning fn with no -> is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn noop() {\n }\nfn main() -> int {\n noop()\n return 5\n}\n",
    ));
}

test "explicit -> () return type is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn noop() -> () {\n }\nfn main() -> int {\n noop()\n return 0\n}\n",
    ));
}

test "loop value-break agreement is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n i := 0\n loop {\n if i >= 5 { break i * 10 }\n i = i + 1\n }\n}\n",
    ));
}

test "loop value-break disagreement is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() -> int {\n i := 0\n loop {\n if i >= 5 { break i } else { break i > 0 }\n i = i + 1\n }\n}\n",
    ));
}

test "break-less loop is never and satisfies an int fn return" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn ready(n: int) -> bool { return n >= 3 }\nfn f() -> int {\n i := 0\n loop {\n if ready(i) { return 7 }\n i = i + 1\n }\n}\n",
    ));
}

test "value-if with a break-less-loop arm merges to the other arm (never unifies)" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn ready() -> bool { return true }\nfn f() -> int {\n c := 1\n x := if c > 0 { loop { if ready() { return 0 } } } else { 5 }\n return x\n}\n",
    ));
}

test "break with a value in a while loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n while true { break 5 }\n return\n}\n",
    ));
}

test "break with a value in a for loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n for i in 0..5 { break i }\n return\n}\n",
    ));
}

test "bare break in a while loop is accepted" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() {\n while true { break }\n return\n}\n",
    ));
}

test "break outside a loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n break\n return\n}\n",
    ));
}

test "continue outside a loop is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n continue\n return\n}\n",
    ));
}

test "for range bound that is not int is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "fn f() {\n for i in true..5 { x := i }\n return\n}\n",
    ));
}

test "for body using its int loop variable typechecks clean" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        "fn f() -> int {\n s := 0\n for i in 0..5 { s = s + i }\n return s\n}\n",
    ));
}

test "print resolves and typechecks at index user_fn_count" {
    // The synthetic `print` must occupy the index right after the user fns in
    // BOTH Resolve and Typecheck so Codegen can lower print(...) as bl print_sym.
    const gpa = testing.allocator;
    var c = try checkSource("fn main() {\n print(\"hi\")\n return\n}\n");
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // Find the call's callee identifier; it must resolve to func == 1 (main is
    // the only user fn, index 0; print is seeded next at index 1).
    var found = false;
    for (c.tree.nodes) |n| {
        if (n.tag == .call) {
            const callee_res = c.resolve.resolutions[n.lhs];
            try testing.expect(callee_res == .func);
            try testing.expectEqual(@as(u32, 1), callee_res.func);
            found = true;
        }
    }
    try testing.expect(found);
}

test "labeled bare block value typechecks: trailing + breaks must agree" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f(a: bool, b: bool) -> int {
        \\ x := @calc {
        \\  if a { break @calc 1 }
        \\  if b { break @calc 2 }
        \\  3
        \\ }
        \\ return x
        \\}
        \\
    ));
}

test "labeled bare block with mismatched break types is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f(a: bool) -> int {
        \\ x := @calc {
        \\  if a { break @calc 1 }
        \\  true
        \\ }
        \\ return x
        \\}
        \\
    ));
}

test "break a value out of an outer loop typechecks" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f() -> int {
        \\ @outer loop {
        \\  for j in 0..10 { if j == 4 { break @outer j * 10 } }
        \\ }
        \\}
        \\
    ));
}

test "continue on a labeled bare block is rejected (not a loop)" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f() {
        \\ @blk { continue @blk }
        \\ return
        \\}
        \\
    ));
}

test "break with a value out of a labeled while is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f() {
        \\ @w while true { break @w 5 }
        \\ return
        \\}
        \\
    ));
}

test "break with a value out of a labeled for is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        \\fn f() {
        \\ @ff for i in 0..3 { break @ff 5 }
        \\ return
        \\}
        \\
    ));
}

test "a break-less labeled loop used as a value is never (merges with caller)" {
    // The outer loop has only `break @outer` from inside, so the inner loop is
    // break-less (never) yet the program typechecks via the outer's value.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f() -> int {
        \\ @outer loop {
        \\  loop { break @outer 7 }
        \\ }
        \\}
        \\
    ));
}

test "labeled loop that only breaks to an outer does NOT count as diverging-free" {
    // `@L loop { for j in 0..3 { break @L } }`: the for's bare break binds to the
    // for; the outer loop is exited by `break @L`, so it must NOT be `never`.
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\fn f() {
        \\ @L loop { for j in 0..3 { break @L } }
        \\ return
        \\}
        \\
    ));
}

test "clean struct construct/read/store/free-fn typechecks with zero diagnostics" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct Point { x: int, y: int }
        \\fn area(p: Point) -> int { p.x * p.y }
        \\fn main() -> int {
        \\ p := Point { x: 6, y: 7 }
        \\ p.x = 6
        \\ q := p
        \\ return area(q)
        \\}
        \\
    ));
}

test "field punning typechecks clean" {
    try testing.expectEqual(@as(usize, 0), try checkDiagCount(
        \\struct Point { x: int, y: int }
        \\fn f() -> int {
        \\ x := 1
        \\ y := 2
        \\ p := Point { x, y }
        \\ return p.x + p.y
        \\}
        \\
    ));
}

test "nested field access typechecks clean and records the field type" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct Inner { v: int }
        \\struct Outer { i: Inner, w: int }
        \\fn f() -> int {
        \\ o := Outer { i: Inner { v: 1 }, w: 2 }
        \\ return o.i.v + o.w
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
}

test "missing field in construction is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int, y: int }\nfn f() -> int { p := P { x: 1 }\n return p.x }\n",
    ));
}

test "unknown field in construction is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: 1, z: 2 }\n return p.x }\n",
    ));
}

test "unknown field in access is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: 1 }\n return p.q }\n",
    ));
}

test "field type mismatch in construction is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P { x: true }\n return p.x }\n",
    ));
}

test "positional construction Point(1) is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() -> int { p := P(1)\n return p.x }\n",
    ));
}

test "directly-recursive struct is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct N { n: N }\nfn main() -> int { return 0 }\n",
    ));
}

test "indirectly-recursive struct cycle is rejected" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct A { b: B }\nstruct B { a: A }\nfn main() -> int { return 0 }\n",
    ));
}

test "assign-to-field type mismatch is an error" {
    try testing.expectEqual(@as(usize, 1), try checkDiagCount(
        "struct P { x: int }\nfn f() { p := P { x: 1 }\n p.x = true\n return }\n",
    ));
}

test "node_types carry the struct type with the right id, and a 3-int layout is 0/8/16 size 24" {
    const gpa = testing.allocator;
    var c = try checkSource(
        \\struct V3 { a: int, b: int, c: int }
        \\fn f() -> int {
        \\ v := V3 { a: 1, b: 2, c: 3 }
        \\ return v.a
        \\}
        \\
    );
    defer c.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), c.result.diags.len);
    // The struct_init node carries a @"struct" type with struct_id 0.
    var found = false;
    for (c.tree.nodes, 0..) |n, i| {
        if (n.tag == .struct_init) {
            try testing.expectEqual(Kind.@"struct", c.result.node_types[i].kind);
            try testing.expectEqual(@as(u32, 0), c.result.node_types[i].struct_id);
            found = true;
        }
    }
    try testing.expect(found);
    // Layout: offsets 0/8/16, size 24, align 8.
    const l = c.result.layouts[0];
    try testing.expectEqual(@as(usize, 3), l.offsets.len);
    try testing.expectEqual(@as(u32, 0), l.offsets[0]);
    try testing.expectEqual(@as(u32, 8), l.offsets[1]);
    try testing.expectEqual(@as(u32, 16), l.offsets[2]);
    try testing.expectEqual(@as(u32, 24), l.size);
    try testing.expectEqual(@as(u32, 8), l.@"align");
}
