//! Pure, read-only control-flow analysis over a resolved AST: definite-return,
//! break-targeting, and divergence walks lifted out of `BodyChecker` as free
//! functions. Read-only inputs travel by value in a `Ctx` (`Ctx.fromChecker(bc)`).

const std = @import("std");
const Ast = @import("ast/Ast.zig");
const Token = @import("ast/Token.zig").Token;
const Resolution = @import("symbols/Resolution.zig").Resolution;
const LayoutEngine = @import("layout/Engine.zig");
const Type = LayoutEngine.Type;
const VariantSym = LayoutEngine.VariantSym;
const EnumSym = LayoutEngine.EnumSym;

/// Bound on the stack-allocated variant-coverage bitmap the exhaustiveness walks use.
/// It is a soundness-safe cap, not a hard limit: every walk that reads it falls back to
/// "not exhaustive" (`return false`) when an enum has more variants, so a larger enum is
/// merely treated conservatively, never miscompiled. `PatternChecker`'s App-aware mirror
/// (`orCoversTyApp`) reads this SAME constant so the two coverage paths cannot drift.
pub const max_cover_variants = 64;

/// The read-only slice of checker state the control-flow walks read. Copied by
/// value into each entry point; the walks thread it down unchanged.
pub const Ctx = struct {
    tree: Ast.Tree,
    resolutions: []const Resolution,
    /// Per-node types; only `matchDiverges` reads it (the scrutinee's type).
    node_types: []const Type,
    /// The program's enum symbols; read for variant coverage in `matchDiverges`.
    enums: []const EnumSym,
    tokens: []const Token,
    source: []const u8,

    /// Text of the token at `tok` (variant/name identifiers).
    fn nameText(ctx: Ctx, tok: u32) []const u8 {
        return ctx.tokens[tok].text(ctx.source);
    }

    /// Build a `Ctx` from anything with the matching fields (a `*const
    /// BodyChecker`), reaching through `model.enums` for the enum table.
    pub fn fromChecker(bc: anytype) Ctx {
        return .{
            .tree = bc.tree,
            .resolutions = bc.resolutions,
            .node_types = bc.node_types,
            .enums = bc.model.enums,
            .tokens = bc.tokens,
            .source = bc.source,
        };
    }
};

pub fn blockReturns(ctx: Ctx, block_idx: Ast.Index) bool {
    const stmts = Ast.rangeSlice(ctx.tree, ctx.tree.nodes[block_idx.int()].lhs.int());
    if (stmts.len == 0) return false;
    return stmtReturns(ctx, stmts[stmts.len - 1]);
}

pub fn stmtReturns(ctx: Ctx, stmt_idx: Ast.Index) bool {
    const stmt = ctx.tree.nodes[stmt_idx.int()];
    return switch (stmt.tag) {
        .return_stmt => true,
        // A statement-position block/if is parsed wrapped in an `expr_stmt`; unwrap
        // it so a trailing diverging bare block (`{ return 5 }`) or parenthesized
        // value-if satisfies definite-return and a divergent arm merges correctly.
        .expr_stmt => stmtReturns(ctx, stmt.lhs),
        .block => blockReturns(ctx, stmt_idx),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(ctx.tree, stmt.rhs.int());
            // An else-less `if` can be skipped, so it never guarantees a return.
            if (h.else_node == Ast.none) break :blk false;
            const then_ok = blockReturns(ctx, h.then_block);
            const else_ok = if (ctx.tree.nodes[h.else_node.int()].tag == .if_stmt)
                stmtReturns(ctx, h.else_node)
            else
                blockReturns(ctx, h.else_node);
            break :blk then_ok and else_ok;
        },
        // Conservative: a `while` may never execute, so it can't guarantee a
        // return (`while true { return }` is rejected — acceptable for now).
        .while_stmt => false,
        // A `loop` diverges (returns/never-falls-through) iff it has NO `break`
        // targeting it: the only ways out are `return` or an outer construct.
        .loop_expr => loopDiverges(ctx, stmt_idx),
        // A labeled wrapper is transparent for definite-return: it returns iff its
        // inner construct does (a labeled bare block via its trailing stmt).
        .labeled => stmtReturns(ctx, ctx.tree.nodes[stmt_idx.int()].lhs),
        .for_stmt, .break_stmt, .continue_stmt => false,
        else => false,
    };
}

pub fn loopDiverges(ctx: Ctx, loop_idx: Ast.Index) bool {
    return !blockHasBreak(ctx, ctx.tree.nodes[loop_idx.int()].lhs, loop_idx);
}

pub fn blockHasBreak(ctx: Ctx, block_idx: Ast.Index, target: Ast.Index) bool {
    for (Ast.rangeSlice(ctx.tree, ctx.tree.nodes[block_idx.int()].lhs.int())) |s| {
        if (stmtHasBreak(ctx, s, target)) return true;
    }
    return false;
}

pub fn stmtHasBreak(ctx: Ctx, stmt_idx: Ast.Index, target: Ast.Index) bool {
    const stmt = ctx.tree.nodes[stmt_idx.int()];
    return switch (stmt.tag) {
        // A bare break (no label) binds to the innermost loop — counts only when
        // `target` IS the innermost loop, i.e. the bare break is found before any
        // nested loop swallows it (the nested-loop arms below stop the descent for
        // bare breaks). A labeled break counts iff its resolved target matches.
        .break_stmt => if (ctx.resolutions[stmt_idx.int()] == .label)
            ctx.resolutions[stmt_idx.int()].label == target
        else
            true,
        .expr_stmt => stmtHasBreak(ctx, stmt.lhs, target),
        .block => blockHasBreak(ctx, stmt_idx, target),
        // A labeled wrapper is transparent: descend its inner construct (a
        // `break @target` may live inside a nested labeled loop).
        .labeled => stmtHasBreak(ctx, stmt.lhs, target),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(ctx.tree, stmt.rhs.int());
            if (blockHasBreak(ctx, h.then_block, target)) break :blk true;
            if (h.else_node == Ast.none) break :blk false;
            break :blk if (ctx.tree.nodes[h.else_node.int()].tag == .if_stmt)
                stmtHasBreak(ctx, h.else_node, target)
            else
                blockHasBreak(ctx, h.else_node, target);
        },
        // A nested loop/for/while swallows BARE breaks, but a `break @target`
        // buried inside it still targets `target` — so descend its body and only
        // count labeled breaks that name `target`.
        .loop_expr => nestedHasLabeledBreak(ctx, ctx.tree.nodes[stmt_idx.int()].lhs, target),
        .while_stmt => nestedHasLabeledBreak(ctx, ctx.tree.nodes[stmt_idx.int()].rhs, target),
        .for_stmt => nestedHasLabeledBreak(ctx, ctx.tree.nodes[stmt_idx.int()].lhs, target),
        else => false,
    };
}

pub fn nestedHasLabeledBreak(ctx: Ctx, block_idx: Ast.Index, target: Ast.Index) bool {
    for (Ast.rangeSlice(ctx.tree, ctx.tree.nodes[block_idx.int()].lhs.int())) |s| {
        if (stmtHasLabeledBreak(ctx, s, target)) return true;
    }
    return false;
}

pub fn stmtHasLabeledBreak(ctx: Ctx, stmt_idx: Ast.Index, target: Ast.Index) bool {
    const stmt = ctx.tree.nodes[stmt_idx.int()];
    return switch (stmt.tag) {
        .break_stmt => ctx.resolutions[stmt_idx.int()] == .label and ctx.resolutions[stmt_idx.int()].label == target,
        .expr_stmt => stmtHasLabeledBreak(ctx, stmt.lhs, target),
        .block => nestedHasLabeledBreak(ctx, stmt_idx, target),
        .labeled => stmtHasLabeledBreak(ctx, stmt.lhs, target),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(ctx.tree, stmt.rhs.int());
            if (nestedHasLabeledBreak(ctx, h.then_block, target)) break :blk true;
            if (h.else_node == Ast.none) break :blk false;
            break :blk if (ctx.tree.nodes[h.else_node.int()].tag == .if_stmt)
                stmtHasLabeledBreak(ctx, h.else_node, target)
            else
                nestedHasLabeledBreak(ctx, h.else_node, target);
        },
        .loop_expr => nestedHasLabeledBreak(ctx, ctx.tree.nodes[stmt_idx.int()].lhs, target),
        .while_stmt => nestedHasLabeledBreak(ctx, ctx.tree.nodes[stmt_idx.int()].rhs, target),
        .for_stmt => nestedHasLabeledBreak(ctx, ctx.tree.nodes[stmt_idx.int()].lhs, target),
        else => false,
    };
}

pub fn blockDiverges(ctx: Ctx, block_idx: Ast.Index) bool {
    const stmts = Ast.rangeSlice(ctx.tree, ctx.tree.nodes[block_idx.int()].lhs.int());
    if (stmts.len == 0) return false;
    return stmtDiverges(ctx, stmts[stmts.len - 1]);
}

pub fn stmtDiverges(ctx: Ctx, stmt_idx: Ast.Index) bool {
    const stmt = ctx.tree.nodes[stmt_idx.int()];
    return switch (stmt.tag) {
        .return_stmt, .break_stmt, .continue_stmt => true,
        .expr_stmt => stmtDiverges(ctx, stmt.lhs),
        .block => blockDiverges(ctx, stmt_idx),
        .if_stmt => blk: {
            const h = Ast.ifHeaderAt(ctx.tree, stmt.rhs.int());
            if (h.else_node == Ast.none) break :blk false;
            const then_ok = blockDiverges(ctx, h.then_block);
            const else_ok = if (ctx.tree.nodes[h.else_node.int()].tag == .if_stmt)
                stmtDiverges(ctx, h.else_node)
            else
                blockDiverges(ctx, h.else_node);
            break :blk then_ok and else_ok;
        },
        .while_stmt => false,
        .loop_expr => loopDiverges(ctx, stmt_idx),
        .labeled => labeledDiverges(ctx, stmt_idx),
        .for_stmt => false,
        // A `match` diverges iff it is exhaustive AND every arm body diverges.
        .match_expr => matchDiverges(ctx, stmt_idx),
        else => false,
    };
}

pub fn matchDiverges(ctx: Ctx, node_idx: Ast.Index) bool {
    const n = ctx.tree.nodes[node_idx.int()];
    const st = ctx.node_types[n.lhs.int()];
    // Conservative for int/bool scrutinees: `return false` (loses only a
    // definite-return optimization, never miscompiles). Only enums get the
    // variant-coverage analysis here.
    if (!st.isEnum()) return false;
    const arms = Ast.rangeSlice(ctx.tree, n.rhs.int());
    if (arms.len == 0) return false;
    var has_wildcard = false;
    const e = ctx.enums[st.enum_id];
    var seen = [_]bool{false} ** max_cover_variants;
    for (arms) |arm_idx| {
        const arm = ctx.tree.nodes[arm_idx.int()];
        const h = Ast.armHeaderAt(ctx.tree, arm.rhs.int());
        if (!stmtDiverges(ctx, h.body)) return false;
        if (h.guard != Ast.none) continue; // a guard can fail → no coverage
        const pat = ctx.tree.nodes[arm.lhs.int()];
        if (pat.tag == .pattern_wildcard) {
            has_wildcard = true;
        } else if (pat.tag == .pattern_variant) {
            const vname = ctx.nameText(pat.main_token);
            for (e.variants, 0..) |v, i| {
                if (i < seen.len and std.mem.eql(u8, v.name, vname) and variantPayloadIrrefutable(ctx, arm.lhs, v)) seen[i] = true;
            }
        }
    }
    if (has_wildcard) return true;
    if (e.variants.len > seen.len) return false;
    for (e.variants, 0..) |_, i| if (!seen[i]) return false;
    return true;
}

pub fn labeledDiverges(ctx: Ctx, idx: Ast.Index) bool {
    const inner_idx = ctx.tree.nodes[idx.int()].lhs;
    const inner = ctx.tree.nodes[inner_idx.int()];
    return switch (inner.tag) {
        .loop_expr => loopDiverges(ctx, inner_idx),
        .while_stmt, .for_stmt => false,
        .block => blockDiverges(ctx, inner_idx) and !blockHasBreak(ctx, inner_idx, inner_idx),
        else => false,
    };
}

pub fn armDiverges(ctx: Ctx, node_idx: Ast.Index) bool {
    return stmtDiverges(ctx, node_idx);
}

// These are pure walks over the AST + enum table too: they answer "does this
// pattern always match?", which `matchDiverges` needs for variant coverage and
// the pattern checker needs for exhaustiveness. Kept here so the coverage logic
// lives beside the divergence logic that consumes it.

pub fn irrefutable(ctx: Ctx, pat_idx: Ast.Index, ty: Type) bool {
    if (ty.kind == .invalid) return true; // poison already reported; don't cascade a spurious miss
    const pat = ctx.tree.nodes[pat_idx.int()];
    return switch (pat.tag) {
        .pattern_wildcard => true,
        .pattern_binding => pat.rhs == Ast.none or irrefutable(ctx, pat.rhs, ty),
        .pattern_literal => false,
        .pattern_variant => blk: {
            // Total only when the enum has exactly ONE variant (the tag test cannot
            // fail) AND that variant's payload is fully covered. Against a
            // multi-variant enum a single `.V` is refutable.
            if (ty.kind != .@"enum") break :blk false;
            const e = ctx.enums[ty.enum_id];
            if (e.variants.len != 1) break :blk false;
            break :blk variantPayloadIrrefutable(ctx, pat_idx, e.variants[0]);
        },
        .pattern_or => orCoversType(ctx, pat_idx, ty),
        else => false,
    };
}

pub fn variantPayloadIrrefutable(ctx: Ctx, pat_idx: Ast.Index, variant: VariantSym) bool {
    const pat = ctx.tree.nodes[pat_idx.int()];
    const binders = if (pat.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(ctx.tree, pat.rhs.int());
    switch (variant.form) {
        .unit => return binders.len == 0,
        .tuple => {
            if (binders.len != variant.field_types.len) return false;
            for (binders, variant.field_types) |b, fty| if (!irrefutable(ctx, b, fty)) return false;
            return true;
        },
        .@"struct" => {
            for (binders) |b_idx| {
                const b = ctx.tree.nodes[b_idx.int()];
                const src = if (b.lhs != Ast.none) ctx.nameText(ctx.tree.nodes[b.lhs.int()].main_token) else ctx.nameText(b.main_token);
                var fty: Type = .invalid;
                for (variant.field_names, 0..) |dn, j| if (std.mem.eql(u8, dn, src)) {
                    fty = variant.field_types[j];
                    break;
                };
                if (!irrefutable(ctx, b_idx, fty)) return false;
            }
            return true;
        },
    }
}

pub fn orCoversType(ctx: Ctx, or_idx: Ast.Index, ty: Type) bool {
    const alts = Ast.rangeSlice(ctx.tree, ctx.tree.nodes[or_idx.int()].lhs.int());
    for (alts) |a| if (irrefutable(ctx, a, ty)) return true;
    if (ty.kind == .@"enum") {
        const e = ctx.enums[ty.enum_id];
        var seen = [_]bool{false} ** max_cover_variants;
        if (e.variants.len > seen.len) return false;
        for (alts) |a| {
            const ap = ctx.tree.nodes[a.int()];
            if (ap.tag != .pattern_variant) continue;
            const vname = ctx.nameText(ap.main_token);
            for (e.variants, 0..) |v, i| {
                if (std.mem.eql(u8, v.name, vname) and variantPayloadIrrefutable(ctx, a, v)) seen[i] = true;
            }
        }
        for (e.variants, 0..) |_, i| if (!seen[i]) return false;
        return true;
    }
    return false;
}

// These build small ASTs by hand (the same node-array + extra idiom as Ast.zig's
// own tests) rather than parsing, so each test pins the exact node graph a walk
// sees. A tiny `Builder` appends nodes/extra and hands back indices.

const testing = std.testing;

const Builder = struct {
    nodes: std.ArrayList(Ast.Node) = .empty,
    extra: std.ArrayList(u32) = .empty,
    resolutions: std.ArrayList(Resolution) = .empty,
    node_types: std.ArrayList(Type) = .empty,
    gpa: std.mem.Allocator,

    fn init(gpa: std.mem.Allocator) Builder {
        return .{ .gpa = gpa };
    }

    fn deinit(b: *Builder) void {
        b.nodes.deinit(b.gpa);
        b.extra.deinit(b.gpa);
        b.resolutions.deinit(b.gpa);
        b.node_types.deinit(b.gpa);
    }

    /// Append a node; also grow the parallel `resolutions`/`node_types` arrays so
    /// every node index is addressable. Returns the new node's index.
    fn add(b: *Builder, node: Ast.Node) !Ast.Index {
        const idx = Ast.Index.from(@intCast(b.nodes.items.len));
        try b.nodes.append(b.gpa, node);
        try b.resolutions.append(b.gpa, .unresolved);
        try b.node_types.append(b.gpa, .invalid);
        return idx;
    }

    /// Append a `{start, len}` range header over `items`, returning the header
    /// cell as an `Ast.Index` (what a `Node`'s lhs/rhs stores). The `Index` run is
    /// written into the `[]u32` `extra` verbatim (layout-identical).
    fn range(b: *Builder, items: []const Ast.Index) !Ast.Index {
        const start: u32 = @intCast(b.extra.items.len);
        const cells: []const u32 = @ptrCast(items);
        try b.extra.appendSlice(b.gpa, cells);
        const header: u32 = @intCast(b.extra.items.len);
        try b.extra.append(b.gpa, start);
        try b.extra.append(b.gpa, @intCast(items.len));
        return Ast.Index.from(header);
    }

    /// Append a 2-cell header `{a, b}` (if/for/arm), returning its index.
    fn pair(b: *Builder, a: Ast.Index, c: Ast.Index) !Ast.Index {
        const header: u32 = @intCast(b.extra.items.len);
        try b.extra.append(b.gpa, a.int());
        try b.extra.append(b.gpa, c.int());
        return Ast.Index.from(header);
    }

    fn block(b: *Builder, stmts: []const Ast.Index) !Ast.Index {
        const r = try b.range(stmts);
        return b.add(.{ .tag = .block, .main_token = 0, .lhs = r, .rhs = Ast.none });
    }

    fn ret(b: *Builder) !Ast.Index {
        return b.add(.{ .tag = .return_stmt, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    }

    /// A bare `break` (no label): `resolutions[idx]` left `.unresolved`.
    fn breakBare(b: *Builder) !Ast.Index {
        return b.add(.{ .tag = .break_stmt, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    }

    /// A labeled `break @target`: resolves to `target`'s inner construct node.
    fn breakTo(b: *Builder, target: Ast.Index) !Ast.Index {
        const idx = try b.add(.{ .tag = .break_stmt, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
        b.resolutions.items[idx.int()] = .{ .label = target };
        return idx;
    }

    fn loopOf(b: *Builder, body: Ast.Index) !Ast.Index {
        return b.add(.{ .tag = .loop_expr, .main_token = 0, .lhs = body, .rhs = Ast.none });
    }

    fn ctx(b: *Builder) Ctx {
        return .{
            .tree = .{ .nodes = b.nodes.items, .extra = b.extra.items },
            .resolutions = b.resolutions.items,
            .node_types = b.node_types.items,
            .enums = &.{},
            .tokens = &.{},
            .source = "",
        };
    }
};

test "loop with no break diverges; loop with a bare break does not" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    // loop { return } — no break, so the only exit is `return` → diverges.
    const div_body = try b.block(&.{try b.ret()});
    const div_loop = try b.loopOf(div_body);

    // loop { break } — a bare break escapes → does not diverge.
    const brk = try b.breakBare();
    const brk_body = try b.block(&.{brk});
    const brk_loop = try b.loopOf(brk_body);

    const c = b.ctx();
    try testing.expect(loopDiverges(c, div_loop));
    try testing.expect(!loopDiverges(c, brk_loop));
    // stmtDiverges routes a loop_expr through loopDiverges.
    try testing.expect(stmtDiverges(c, div_loop));
    try testing.expect(!stmtDiverges(c, brk_loop));
    // A break-less loop is a definite return; a breaking one falls through.
    try testing.expect(stmtReturns(c, div_loop));
    try testing.expect(!stmtReturns(c, brk_loop));
}

test "a bare break in a nested loop does not escape the outer loop" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    // outer: loop { inner: loop { break } }
    // The bare break binds to `inner`, so `outer` sees no escaping break → outer
    // still diverges even though a break exists textually inside it.
    const brk = try b.breakBare();
    const inner_body = try b.block(&.{brk});
    const inner = try b.loopOf(inner_body);
    const outer_body = try b.block(&.{inner});
    const outer = try b.loopOf(outer_body);

    const c = b.ctx();
    try testing.expect(!loopDiverges(c, inner)); // inner is broken out of
    try testing.expect(loopDiverges(c, outer)); // outer is not
}

test "a labeled break escapes the outer loop it targets" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    // outer: loop { inner: loop { break @outer } }
    // The labeled break targets `outer`, so it DOES escape past `inner`: outer no
    // longer diverges, but inner (which the break passes through) does. Nodes are
    // added children-first, so build the break bare, then retarget it to `outer`
    // once that index exists.
    const brk = try b.breakBare();
    const inner_body = try b.block(&.{brk});
    const inner = try b.loopOf(inner_body);
    const outer_body = try b.block(&.{inner});
    const outer = try b.loopOf(outer_body);
    // Now retarget the break to `outer` (labeled break @outer).
    b.resolutions.items[brk.int()] = .{ .label = outer };

    const c = b.ctx();
    try testing.expect(loopDiverges(c, inner)); // bare-break gone; break is labeled to outer
    try testing.expect(!loopDiverges(c, outer)); // labeled break escapes outer
    // The break inside inner is a labeled break naming outer.
    try testing.expect(stmtHasLabeledBreak(c, inner, outer));
    try testing.expect(!stmtHasBreak(c, inner, inner)); // it does not target inner
    try testing.expect(stmtHasBreak(c, outer, outer)); // it targets outer
}

test "if diverges only when both arms diverge; else-less never does" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    const then_ret = try b.block(&.{try b.ret()});
    const else_ret = try b.block(&.{try b.ret()});
    const else_fall = try b.block(&.{}); // empty → falls through
    const cond = try b.add(.{ .tag = .literal_bool, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });

    // if c { return } else { return } → diverges / returns
    const both = try b.add(.{ .tag = .if_stmt, .main_token = 0, .lhs = cond, .rhs = try b.pair(then_ret, else_ret) });
    // if c { return } else { } → one arm falls through
    const one = try b.add(.{ .tag = .if_stmt, .main_token = 0, .lhs = cond, .rhs = try b.pair(then_ret, else_fall) });
    // if c { return } (no else) → skippable
    const none_else = try b.add(.{ .tag = .if_stmt, .main_token = 0, .lhs = cond, .rhs = try b.pair(then_ret, Ast.none) });

    const c = b.ctx();
    try testing.expect(stmtDiverges(c, both));
    try testing.expect(stmtReturns(c, both));
    try testing.expect(!stmtDiverges(c, one));
    try testing.expect(!stmtReturns(c, one));
    try testing.expect(!stmtDiverges(c, none_else));
    try testing.expect(!stmtReturns(c, none_else));
}

test "nested blocks: divergence and definite-return see through wrapping" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    // { { { return } } } — trailing nested blocks are transparent.
    const inner = try b.block(&.{try b.ret()});
    const mid = try b.block(&.{inner});
    const outer = try b.block(&.{mid});

    // { <expr_stmt wrapping return-less...> } — an empty inner block falls through.
    const empty_inner = try b.block(&.{});
    const wrap_empty = try b.block(&.{empty_inner});

    const c = b.ctx();
    try testing.expect(blockReturns(c, outer));
    try testing.expect(blockDiverges(c, outer));
    try testing.expect(!blockReturns(c, wrap_empty));
    try testing.expect(!blockDiverges(c, wrap_empty));
    // A non-trailing statement does not decide a block's return: put a return
    // BEFORE an empty block and confirm the block does NOT count as returning.
    const early_ret = try b.ret();
    const trailing_empty = try b.block(&.{});
    const early = try b.block(&.{ early_ret, trailing_empty });
    const c2 = b.ctx();
    try testing.expect(!blockReturns(c2, early));
}

test "match diverges iff exhaustive and every arm diverges" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    // A two-variant enum `E { A, B }` (both unit). Build the enum symbol table.
    const variants = [_]VariantSym{
        .{ .name = "A", .form = .unit },
        .{ .name = "B", .form = .unit },
    };
    const enums = [_]EnumSym{
        .{ .decl_node = Ast.Index.from(0), .name = "E", .variants = @constCast(&variants) },
    };
    const scrut_ty = Type.enumT(0);

    // We need `nameText(pat.main_token)` to return the variant name, so back the
    // patterns with a tiny token/source pair. token[0]="A", token[1]="B".
    const source = "AB";
    const tokens = [_]Token{
        .{ .tag = .identifier, .start = 0, .end = 1 },
        .{ .tag = .identifier, .start = 1, .end = 2 },
    };

    // Helper: a diverging arm body is `{ return }`; a falling body is `{}`.
    // A `.V` unit variant pattern: pattern_variant, main_token = variant tok.
    const patA = try b.add(.{ .tag = .pattern_variant, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const patB = try b.add(.{ .tag = .pattern_variant, .main_token = 1, .lhs = Ast.none, .rhs = Ast.none });

    const bodyA = try b.block(&.{try b.ret()});
    const bodyB = try b.block(&.{try b.ret()});
    const armA = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = patA, .rhs = try b.pair(Ast.none, bodyA) });
    const armB = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = patB, .rhs = try b.pair(Ast.none, bodyB) });

    // The scrutinee node; its node_type must be the enum type (matchDiverges reads
    // node_types[n.lhs]). Use a plain identifier as the scrutinee.
    const scrut = try b.add(.{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });

    // Exhaustive match: both A and B covered, both bodies diverge → diverges.
    const arms_full = try b.range(&.{ armA, armB });
    const match_full = try b.add(.{ .tag = .match_expr, .main_token = 0, .lhs = scrut, .rhs = arms_full });

    // Non-exhaustive: only A covered → does NOT diverge.
    const arms_partial = try b.range(&.{armA});
    const match_partial = try b.add(.{ .tag = .match_expr, .main_token = 0, .lhs = scrut, .rhs = arms_partial });

    // Exhaustive coverage but one arm falls through: B's body is `{}` → not diverge.
    const bodyB_fall = try b.block(&.{});
    const armB_fall = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = patB, .rhs = try b.pair(Ast.none, bodyB_fall) });
    const arms_fall = try b.range(&.{ armA, armB_fall });
    const match_fall = try b.add(.{ .tag = .match_expr, .main_token = 0, .lhs = scrut, .rhs = arms_fall });

    // Set the scrutinee's type in node_types.
    b.node_types.items[scrut.int()] =scrut_ty;

    const c: Ctx = .{
        .tree = .{ .nodes = b.nodes.items, .extra = b.extra.items },
        .resolutions = b.resolutions.items,
        .node_types = b.node_types.items,
        .enums = &enums,
        .tokens = &tokens,
        .source = source,
    };

    try testing.expect(matchDiverges(c, match_full));
    try testing.expect(!matchDiverges(c, match_partial));
    try testing.expect(!matchDiverges(c, match_fall));
    // stmtDiverges routes a match_expr through matchDiverges.
    try testing.expect(stmtDiverges(c, match_full));
    try testing.expect(!stmtDiverges(c, match_partial));
}

test "guarded arm never contributes coverage" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    const variants = [_]VariantSym{
        .{ .name = "A", .form = .unit },
        .{ .name = "B", .form = .unit },
    };
    const enums = [_]EnumSym{
        .{ .decl_node = Ast.Index.from(0), .name = "E", .variants = @constCast(&variants) },
    };
    const source = "AB";
    const tokens = [_]Token{
        .{ .tag = .identifier, .start = 0, .end = 1 },
        .{ .tag = .identifier, .start = 1, .end = 2 },
    };

    const patA = try b.add(.{ .tag = .pattern_variant, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const patB = try b.add(.{ .tag = .pattern_variant, .main_token = 1, .lhs = Ast.none, .rhs = Ast.none });
    const guard = try b.add(.{ .tag = .literal_bool, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });

    const bodyA = try b.block(&.{try b.ret()});
    const bodyB = try b.block(&.{try b.ret()});
    // Arm A is guarded → covers nothing even though it diverges.
    const armA = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = patA, .rhs = try b.pair(guard, bodyA) });
    const armB = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = patB, .rhs = try b.pair(Ast.none, bodyB) });

    const scrut = try b.add(.{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const arms = try b.range(&.{ armA, armB });
    const match = try b.add(.{ .tag = .match_expr, .main_token = 0, .lhs = scrut, .rhs = arms });
    b.node_types.items[scrut.int()] =Type.enumT(0);

    const c: Ctx = .{
        .tree = .{ .nodes = b.nodes.items, .extra = b.extra.items },
        .resolutions = b.resolutions.items,
        .node_types = b.node_types.items,
        .enums = &enums,
        .tokens = &tokens,
        .source = source,
    };
    // A covered only via a guarded arm → not exhaustive → does not diverge.
    try testing.expect(!matchDiverges(c, match));
}

test "wildcard arm makes a match exhaustive" {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    const variants = [_]VariantSym{
        .{ .name = "A", .form = .unit },
        .{ .name = "B", .form = .unit },
    };
    const enums = [_]EnumSym{
        .{ .decl_node = Ast.Index.from(0), .name = "E", .variants = @constCast(&variants) },
    };
    const source = "A";
    const tokens = [_]Token{.{ .tag = .identifier, .start = 0, .end = 1 }};

    const patA = try b.add(.{ .tag = .pattern_variant, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const patW = try b.add(.{ .tag = .pattern_wildcard, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const bodyA = try b.block(&.{try b.ret()});
    const bodyW = try b.block(&.{try b.ret()});
    const armA = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = patA, .rhs = try b.pair(Ast.none, bodyA) });
    const armW = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = patW, .rhs = try b.pair(Ast.none, bodyW) });
    const scrut = try b.add(.{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const arms = try b.range(&.{ armA, armW });
    const match = try b.add(.{ .tag = .match_expr, .main_token = 0, .lhs = scrut, .rhs = arms });
    b.node_types.items[scrut.int()] =Type.enumT(0);

    const c: Ctx = .{
        .tree = .{ .nodes = b.nodes.items, .extra = b.extra.items },
        .resolutions = b.resolutions.items,
        .node_types = b.node_types.items,
        .enums = &enums,
        .tokens = &tokens,
        .source = source,
    };
    // `_` covers the missing B; both bodies diverge → diverges.
    try testing.expect(matchDiverges(c, match));
}

/// Build a match over an `N`-variant enum where every variant has its own
/// diverging arm (`.Vk => { return }`), and report whether `matchDiverges` sees it
/// as exhaustive. The scrutinee is textually total, so the ONLY thing that can make
/// this false is the `max_cover_variants` bitmap cap. Names are two-letter, unique
/// per variant, backed by a source buffer the pattern tokens slice into.
fn allVariantsDivergeExhaustive(comptime N: usize) !bool {
    var b = Builder.init(testing.allocator);
    defer b.deinit();

    var src: [N * 2]u8 = undefined;
    var tokens: [N]Token = undefined;
    var variants: [N]VariantSym = undefined;
    for (0..N) |k| {
        src[k * 2] = @intCast('A' + k / 26);
        src[k * 2 + 1] = @intCast('A' + k % 26);
        tokens[k] = .{ .tag = .identifier, .start = @intCast(k * 2), .end = @intCast(k * 2 + 2) };
        variants[k] = .{ .name = src[k * 2 ..][0..2], .form = .unit };
    }
    const enums = [_]EnumSym{
        .{ .decl_node = Ast.Index.from(0), .name = "E", .variants = &variants },
    };

    var arm_list: [N]Ast.Index = undefined;
    for (0..N) |k| {
        const pat = try b.add(.{ .tag = .pattern_variant, .main_token = @intCast(k), .lhs = Ast.none, .rhs = Ast.none });
        const body = try b.block(&.{try b.ret()});
        arm_list[k] = try b.add(.{ .tag = .match_arm, .main_token = 0, .lhs = pat, .rhs = try b.pair(Ast.none, body) });
    }
    const scrut = try b.add(.{ .tag = .identifier, .main_token = 0, .lhs = Ast.none, .rhs = Ast.none });
    const arms = try b.range(&arm_list);
    const match = try b.add(.{ .tag = .match_expr, .main_token = 0, .lhs = scrut, .rhs = arms });
    b.node_types.items[scrut.int()] = Type.enumT(0);

    const c: Ctx = .{
        .tree = .{ .nodes = b.nodes.items, .extra = b.extra.items },
        .resolutions = b.resolutions.items,
        .node_types = b.node_types.items,
        .enums = &enums,
        .tokens = &tokens,
        .source = &src,
    };
    return matchDiverges(c, match);
}

test "variant coverage is exact at the cap and conservative past it" {
    // At exactly `max_cover_variants`, a textually-total all-diverging match is seen
    // as exhaustive (every bitmap slot fits). One variant past the cap, the walk falls
    // back to "not exhaustive" (`e.variants.len > seen.len`) rather than reading out of
    // bounds — a soundness-safe under-approximation, never a miscompile.
    try testing.expect(try allVariantsDivergeExhaustive(max_cover_variants));
    try testing.expect(!try allVariantsDivergeExhaustive(max_cover_variants + 1));
}
