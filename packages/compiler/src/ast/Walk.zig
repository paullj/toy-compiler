//! The ONE place AST visit order is defined.
//!
//! `walk` drives a single, exhaustive `switch` over `Ast.Tag` in canonical LHS->RHS order
//! and emits a typed `Event` stream to a comptime visitor. The fingerprint's visitors
//! (`query/AstWalk.zig`), the escape and spec passes, lower and the LSP all ride this one
//! stream, so they cannot drift. A new `Ast.Tag` is a non-exhaustive-switch COMPILE ERROR
//! here, not silent drift across hand-mirrored walks. The two correctness invariants the
//! walk must never violate:
//!
//!   * NO ABSOLUTE INDICES, NO SOURCE OFFSETS — the fold sees node TAGS and leaf
//!     token TEXT, never an index/offset that shifts when a sibling is edited.
//!   * ORDER-SENSITIVE, NEVER XOR — children fold in wiring order with arity
//!     sentinels, so `a - b` and `b - a` cannot collide.

const std = @import("std");
const Token = @import("Token.zig").Token;
const TokenTag = @import("Token.zig").Tag;
const Ast = @import("Ast.zig");

/// The read-only inputs a walk needs to spell a leaf. `tree`/`tokens`/`source`
/// are the same trio every consumer already threads; `leaf`/`tokenText` fold the
/// inline `tokens[n.main_token].text(source)` idiom the old walks repeated.
pub const Source = struct {
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,

    pub fn leaf(self: Source, idx: Ast.Index) []const u8 {
        const n = self.tree.nodes[idx.int()];
        return self.tokens[n.main_token].text(self.source);
    }

    pub fn tokenText(self: Source, tok: u32) []const u8 {
        return self.tokens[tok].text(self.source);
    }
};

/// Every semantic step an order-sensitive consumer observes, in canonical
/// LHS->RHS emission order. `enter` brackets each node so consumers that read
/// `frozen[idx]` key off `enter.idx`; the fold primitives (`leaf`/`raw_leaf`/
/// `count`/`flag`) map 1:1 to the bytes `Fingerprint` folds. `touch`/`callee`/
/// `type_ref`/`operator`/`try_operator` are position-specific events the hash
/// IGNORES (its byte stream is unchanged, so every existing fingerprint is preserved
/// and warm caches never churn); only `TouchedVisitor`/`CallVisitor` react to them.
pub const Event = union(enum) {
    /// Entry of the node at `idx` (tag `tag`), emitted before any child. The tag
    /// byte the hash folds is the visitor's job (see `HashVisitor.on`).
    enter: struct { idx: Ast.Index, tag: Ast.Node.Tag },
    /// Length-prefixed leaf text.
    leaf: []const u8,
    /// Token-text leaf folded WITHOUT recursing a node (a `break`/`continue`
    /// `@label` target, whose label names a token, not a child node; a param's `mut`).
    raw_leaf: []const u8,
    /// Arity sentinel.
    count: u32,
    /// Optionality/shape sentinel.
    flag: bool,
    /// A node whose `node_types[idx]` is a touched type. Emitted at every node the
    /// touched walk collects (i.e. NOT inside a pattern subtree or an enum-init
    /// type-name, which the touched walk never descended into).
    touch: struct { idx: Ast.Index },
    /// A call site: `idx` is the callee identifier node (`call.lhs`), whose
    /// `resolutions[idx]` carries the bound symbol. Emitted AFTER the callee
    /// subtree and BEFORE the args, matching where the old callee walk recorded
    /// the sig (so nested calls in the callee record first). `call` is the enclosing
    /// `.call` node itself — the reader needs it to reach the value-arg nodes when a
    /// bare inferred generic callee must fold its resolved INSTANCE sig.
    callee: struct { idx: Ast.Index, call: Ast.Index },
    /// A `fn_decl` param/return type-ref position. `ordinal` indexes the proto's
    /// params; `is_ret` picks the return. Emitted immediately before recursing the
    /// type-ref node, matching where the touched walk folds the OWNING sig's
    /// param/return type (carrying the cross-module-correct global id).
    type_ref: struct { idx: Ast.Index, ordinal: u32, is_ret: bool },
    /// A desugaring binary operator on a `.binary` node `idx`: `==`/`!=` (Eq/Ord), a
    /// comparison `<`/`>`/`<=`/`>=` (Ord), or arithmetic `+`/`-`/`*`/`/` (Add/Sub/Mul/Div).
    /// Emitted AFTER both operands, like `.callee` fires after a call's callee subtree, and
    /// only for a token that has a candidate method (see `opMethod`). `CallVisitor` folds the
    /// resolved witness for a struct/enum operand so `p == q`/`a < b`/`v1 + v2` tracks the SAME
    /// witness identity the desugared call's reloc targets (editing an `impl … has …` body
    /// invalidates operator callers).
    operator: struct { idx: Ast.Index },
    /// A postfix `?` on a `.try_expr` node `idx`. Emitted AFTER the operand, like the three
    /// operator events. `CallVisitor` reacts ONLY for a WIDENING Result `?` (operand error type
    /// differs from the enclosing return's), folding the resolved `From` witness so `inner()?`
    /// tracks the SAME witness identity the desugared widen's reloc targets (adding/removing an
    /// `impl RetErr has From[OpErr]` invalidates the enclosing fn's codegen unit).
    try_operator: struct { idx: Ast.Index },
};

/// The source extent `[start, end)` of the subtree at `idx`: from its first token to its
/// last, where the tokens are the main tokens of every node the canonical walk enters,
/// widened to balance the brackets inside it — a call's `)` or a block's `}` closes a node
/// but is no node's main token, and a subtree starting inside a parenthesis (`(a + b) * c`)
/// must take the `(` too; parentheses make no node of their own.
pub fn nodeSpan(src: Source, idx: Ast.Index) Span {
    var ext: Extent = .{ .tree = src.tree, .tokens = src.tokens };
    walk(src, idx, &ext) catch unreachable;
    // The canonical walk does not descend into a protocol (its method signatures carry
    // no fingerprinted body), so its extent takes them here.
    const root = src.tree.nodes[idx.int()];
    if (root.tag == .protocol_decl) for (Ast.rangeSlice(src.tree, root.lhs.int())) |m| walk(src, m, &ext) catch unreachable;
    const toks = src.tokens;
    var first = ext.first;
    var last = ext.last;
    // A declaration's main token is its name, and it is spelled from its keywords. Every
    // real child follows the name; one that seems to precede it is synthesized (a protocol
    // method's `self` type points at the protocol's name), so the name bounds the start.
    const is_decl = switch (root.tag) {
        .fn_decl, .extern_fn_decl, .struct_decl, .tuple_struct_decl, .enum_decl, .protocol_decl, .type_alias_decl, .local_fn => true,
        else => false,
    };
    if (is_decl) first = root.main_token;
    var open: u32 = 0; // openers still unclosed at the end
    var unopened: u32 = 0; // closers with no opener inside the range
    for (toks[first .. last + 1]) |t| switch (bracketDelta(t.tag)) {
        1 => open += 1,
        -1 => if (open > 0) {
            open -= 1;
        } else {
            unopened += 1;
        },
        else => {},
    };
    if (is_decl) while (first > 0) switch (toks[first - 1].tag) {
        .kw_pub, .kw_fn, .kw_extern, .kw_struct, .kw_enum, .kw_protocol, .kw_type => first -= 1,
        else => break,
    };
    // Newlines are tokens; a multi-line construct's closer sits on a later line.
    while (unopened > 0) {
        var i = first;
        while (i > 0 and toks[i - 1].tag == .newline) i -= 1;
        if (i == 0 or bracketDelta(toks[i - 1].tag) != 1) break;
        first = i - 1;
        unopened -= 1;
    }
    while (open > 0) {
        var i = last + 1;
        while (i < toks.len and toks[i].tag == .newline) i += 1;
        if (i >= toks.len or bracketDelta(toks[i].tag) != -1) break;
        last = i;
        open -= 1;
    }
    return .{ .start = toks[first].start, .end = toks[last].end };
}

pub const Span = @import("../diagnostics/model.zig").Span;

const Extent = struct {
    tree: Ast.Tree,
    tokens: []const Token,
    first: u32 = std.math.maxInt(u32),
    last: u32 = 0,

    pub fn on(self: *Extent, ev: Event) void {
        switch (ev) {
            .enter => |e| {
                const n = self.tree.nodes[e.idx.int()];
                // A bodyless signature's synthesized empty block sits on the token after the
                // signature, which on a one-line `protocol P { fn m(self) }` is the protocol's `}`.
                if (n.tag == .block and self.tokens[n.main_token].tag != .l_brace) return;
                // An inferred variant (`.a(1)`, a `.a` pattern) is spelled from its `.`,
                // which is no node's main token.
                const lead: u32 = switch (n.tag) {
                    .enum_init_unit, .enum_init_tuple, .enum_init_struct, .pattern_variant => if (n.lhs == Ast.none) 1 else 0,
                    else => 0,
                };
                self.first = @min(self.first, n.main_token - lead);
                self.last = @max(self.last, n.main_token);
                // A piped call's last node token can sit well before its `)`: a filled
                // `_` has no node, so `x |> f(a, _)` would end at `a`.
                if (n.tag == .call and self.tokens[n.main_token].tag == .pipe_gt)
                    if (Ast.pipeArgsClose(self.tokens, n.main_token)) |close| {
                        self.last = @max(self.last, close);
                    };
            },
            else => {},
        }
    }
};

fn bracketDelta(tag: TokenTag) i32 {
    return switch (tag) {
        .l_paren, .l_bracket, .l_brace => 1,
        .r_paren, .r_bracket, .r_brace => -1,
        else => 0,
    };
}

/// The error set of `visitor.on`, or the empty set when the visitor has no `on`.
fn VisitorError(comptime V: type) type {
    const T = @typeInfo(V).pointer.child;
    if (!@hasDecl(T, "on")) return error{};
    const info = @typeInfo(@TypeOf(T.on)).@"fn";
    const Ret = info.return_type.?;
    return switch (@typeInfo(Ret)) {
        .error_union => |eu| eu.error_set,
        else => error{},
    };
}

/// Drive ONE traversal of the subtree at `idx`, emitting events in canonical
/// order. `visitor.on(ev)` runs IFF that method exists (comptime); otherwise the
/// event is dropped at zero cost.
pub fn walk(src: Source, idx: Ast.Index, visitor: anytype) VisitorError(@TypeOf(visitor))!void {
    return walkInner(src, idx, true, visitor);
}

/// Emit `ev` to the visitor iff it declares an `on` method (comptime no-op else).
/// `on` may be fallible or infallible; the `try` is taken only for the fallible
/// case, so an infallible visitor (e.g. `HashVisitor`) folds at zero error cost.
inline fn emit(visitor: anytype, ev: Event) !void {
    const T = @typeInfo(@TypeOf(visitor)).pointer.child;
    if (comptime @hasDecl(T, "on")) {
        const Ret = @typeInfo(@TypeOf(T.on)).@"fn".return_type.?;
        if (comptime @typeInfo(Ret) == .error_union) {
            try visitor.on(ev);
        } else {
            visitor.on(ev);
        }
    }
}

/// A lambda's or a local fn's proto and body: the param count, whether each param and
/// the ret are spelled (an untyped lambda param, an inferred ret), then the body.
fn walkClosure(src: Source, n: Ast.Node, collect: bool, visitor: anytype) VisitorError(@TypeOf(visitor))!void {
    const tree = src.tree;
    const proto = Ast.protoAt(tree, n.lhs.int());
    try emit(visitor, .{ .count = @intCast(proto.params.len) });
    for (proto.params) |p| {
        try emit(visitor, .{ .flag = tree.nodes[p.int()].lhs != Ast.none }); // fn(x) != fn(x: int)
        try walkInner(src, p, collect, visitor);
    }
    try emit(visitor, .{ .flag = proto.ret_type != Ast.none }); // inferred != -> R
    if (proto.ret_type != Ast.none) try walkInner(src, proto.ret_type, collect, visitor);
    try walkInner(src, n.rhs, collect, visitor);
}

/// `collect` is the touched-collection context: it is the canonical order's full
/// recursion EXCEPT it turns OFF inside the two subtrees the touched walk never
/// descended into — a `match_arm` pattern and an enum-init type-name — so `.touch`
/// events fire at EXACTLY the nodes the old `walkTouchedSig` collected. Every
/// other event (`enter`/`leaf`/`count`/`flag`/`callee`/`type_ref`) fires
/// regardless, so the hash byte stream and the callee order are unaffected.
fn walkInner(src: Source, idx: Ast.Index, collect: bool, visitor: anytype) VisitorError(@TypeOf(visitor))!void {
    if (idx == Ast.none) return;
    const tree = src.tree;
    const n = tree.nodes[idx.int()];
    try emit(visitor, .{ .enter = .{ .idx = idx, .tag = n.tag } });
    if (collect) try emit(visitor, .{ .touch = .{ .idx = idx } });
    const leaf = src.leaf(idx);
    switch (n.tag) {
        .literal_number, .literal_float, .literal_string, .literal_bool, .literal_char, .identifier, .empty_list => try emit(visitor, .{ .leaf = leaf }),

        .unary => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor);
        },
        .binary => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor); // lhs THEN rhs: a-b != b-a
            try walkInner(src, n.rhs, collect, visitor);
            // After the operands (mirroring `.callee`'s post-subtree placement), signal a
            // desugaring operator so `CallVisitor` can fold the resolved witness. Gated on the
            // token having a candidate method, so a non-desugaring binary emits nothing.
            if (opMethods(src.tokens[n.main_token].tag).len > 0) try emit(visitor, .{ .operator = .{ .idx = idx } });
        },
        .call => {
            try walkInner(src, n.lhs, collect, visitor);
            try emit(visitor, .{ .callee = .{ .idx = n.lhs, .call = idx } }); // record sig AFTER callee, BEFORE args
            const args = Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(args.len) }); // f() != f(0)
            for (args) |a| try walkInner(src, a, collect, visitor);
        },
        .var_decl => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor);
        },
        .assign => {
            try walkInner(src, n.lhs, collect, visitor);
            try walkInner(src, n.rhs, collect, visitor);
        },
        .return_stmt => {
            try emit(visitor, .{ .flag = n.lhs != Ast.none }); // bare return != return v
            if (n.lhs != Ast.none) try walkInner(src, n.lhs, collect, visitor);
        },
        .expr_stmt => try walkInner(src, n.lhs, collect, visitor),
        .block => {
            const stmts = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(stmts.len) });
            for (stmts) |s| try walkInner(src, s, collect, visitor);
        },
        .param => {
            try emit(visitor, .{ .leaf = leaf });
            // `lowerFn` types a `mut` param's slot from this token, so it is part of the fn's
            // own key (#25).
            if (Ast.isMutParam(tree, src.tokens, idx)) try emit(visitor, .{ .raw_leaf = src.tokenText(n.main_token - 1) });
            try walkInner(src, n.lhs, collect, visitor);
        },
        .fn_decl => {
            try emit(visitor, .{ .leaf = leaf });
            const proto = Ast.protoAt(tree, n.lhs.int());
            // Fold the ordered generic-param NAMES so editing a generic signature
            // (add / remove / reorder `[T,U]`) flips the template fingerprint (and thus
            // every instance built from it). Gated on non-empty, so a non-generic fn
            // folds nothing new.
            if (proto.generic_params.len > 0) {
                try emit(visitor, .{ .count = @intCast(proto.generic_params.len) });
                for (proto.generic_params) |gp| {
                    try emit(visitor, .{ .leaf = src.leaf(gp) });
                    // Fold a `[T has P]` bound protocol-ref so `[T]`->`[T has P]` and a
                    // bound-name change flip the template body-fp (hence every instance).
                    // Unbounded params (lhs == none) emit nothing new; the count sentinel
                    // above already prevents a `[T has P]` (count 1) from aliasing a
                    // `[T,U]` (count 2).
                    const bound = tree.nodes[gp.int()].lhs;
                    if (bound != Ast.none) try walkInner(src, bound, collect, visitor);
                }
            }
            try emit(visitor, .{ .count = @intCast(proto.params.len) });
            for (proto.params, 0..) |p, i| {
                // Fold the OWNING sig's param type (carries the right GLOBAL id,
                // incl. a cross-module qualified `mod.Type`) at the type-ref
                // position the old touched walk did, BEFORE recursing the param.
                const pty_node = tree.nodes[p.int()].lhs;
                if (pty_node != Ast.none) try emit(visitor, .{ .type_ref = .{ .idx = pty_node, .ordinal = @intCast(i), .is_ret = false } });
                try walkInner(src, p, collect, visitor);
            }
            try emit(visitor, .{ .flag = proto.ret_type != Ast.none });
            if (proto.ret_type != Ast.none) {
                try emit(visitor, .{ .type_ref = .{ .idx = proto.ret_type, .ordinal = 0, .is_ret = true } });
                try walkInner(src, proto.ret_type, collect, visitor);
            }
            try walkInner(src, n.rhs, collect, visitor);
        },
        .while_stmt => {
            try walkInner(src, n.lhs, collect, visitor);
            try walkInner(src, n.rhs, collect, visitor);
        },
        .if_stmt => {
            try walkInner(src, n.lhs, collect, visitor);
            const head = Ast.ifHeaderAt(tree, n.rhs.int());
            try walkInner(src, head.then_block, collect, visitor);
            try emit(visitor, .{ .flag = head.else_node != Ast.none });
            if (head.else_node != Ast.none) try walkInner(src, head.else_node, collect, visitor);
        },
        // Never a fingerprint root / never reached inside a fn-body walk. An
        // `impl_decl`/`impl_has_decl` is a top-level decl (its methods are walked as
        // ordinary `fn_decl` fingerprint roots) and a `protocol_decl`'s bodyless sigs
        // never enter the fn table, so all three fold nothing here.
        .program, .import_decl, .impl_decl, .protocol_decl, .impl_has_decl, .type_alias_decl => {},
        // Zero-sized leaf: the tag byte IS its content. Reached as a value literal
        // and as a `()` type-ref (under param/fn_decl ret).
        .literal_unit => {},
        .loop_expr => try walkInner(src, n.lhs, collect, visitor),
        .for_stmt => {
            try emit(visitor, .{ .leaf = leaf });
            const head = Ast.forHeaderAt(tree, n.rhs.int());
            try walkInner(src, head.lo, collect, visitor);
            try walkInner(src, head.hi, collect, visitor);
            try walkInner(src, n.lhs, collect, visitor);
        },
        .for_in_stmt => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.rhs, collect, visitor);
            try walkInner(src, n.lhs, collect, visitor);
        },
        .for_in2_stmt => {
            try emit(visitor, .{ .leaf = leaf }); // first binding token
            const hh = Ast.forIn2HeaderAt(tree, n.lhs.int());
            try emit(visitor, .{ .raw_leaf = src.tokenText(tree.nodes[hh.val_leaf.int()].main_token) }); // second binding token
            try walkInner(src, n.rhs, collect, visitor); // iterable
            try walkInner(src, hh.body, collect, visitor); // body
        },
        .break_stmt => {
            // The label target identity is load-bearing; fold its TEXT (not a
            // token index — index-free). `rhs` names a TOKEN, not a node.
            const label = Ast.labelTok(n);
            try emit(visitor, .{ .flag = label != .none }); // bare != @label
            if (label.unwrap()) |t| try emit(visitor, .{ .raw_leaf = src.tokenText(t.int()) });
            try emit(visitor, .{ .flag = n.lhs != Ast.none }); // bare break != break v
            if (n.lhs != Ast.none) try walkInner(src, n.lhs, collect, visitor);
        },
        .continue_stmt => {
            const label = Ast.labelTok(n);
            try emit(visitor, .{ .flag = label != .none }); // bare != @label
            if (label.unwrap()) |t| try emit(visitor, .{ .raw_leaf = src.tokenText(t.int()) });
        },
        .labeled => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor);
        },
        .struct_decl => {
            try emit(visitor, .{ .leaf = leaf });
            const fields = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(fields.len) });
            for (fields) |f| try walkInner(src, f, collect, visitor);
        },
        .struct_init => {
            try walkInner(src, n.lhs, collect, visitor);
            const inits = Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(inits.len) });
            for (inits) |fi| try walkInner(src, fi, collect, visitor);
        },
        .field_init => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor);
        },
        .field_access => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor);
        },
        .tuple_struct_decl => {
            try emit(visitor, .{ .leaf = leaf });
            const types = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(types.len) });
            for (types) |ty| try walkInner(src, ty, collect, visitor);
        },
        .tuple_field => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor);
        },
        .enum_decl => {
            try emit(visitor, .{ .leaf = leaf });
            const variants = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(variants.len) });
            for (variants) |v| try walkInner(src, v, collect, visitor);
        },
        .enum_variant_unit => try emit(visitor, .{ .leaf = leaf }),
        .enum_variant_tuple => {
            try emit(visitor, .{ .leaf = leaf });
            const types = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(types.len) });
            for (types) |ty| try walkInner(src, ty, collect, visitor);
        },
        .enum_variant_struct => {
            try emit(visitor, .{ .leaf = leaf });
            const fields = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(fields.len) });
            for (fields) |f| try walkInner(src, f, collect, visitor);
        },
        // Variant construction: a sentinel distinguishes inferred `.V` from
        // qualified `N.V`, then the variant name, then the payload. The touched
        // walk never descended the type-name (`n.lhs`), so recurse it with
        // `collect=false` to suppress its `.touch` while still folding its hash.
        .enum_init_unit => {
            try emit(visitor, .{ .flag = n.lhs == Ast.none }); // inferred vs qualified
            if (n.lhs != Ast.none) try walkInner(src, n.lhs, false, visitor);
            try emit(visitor, .{ .leaf = leaf });
        },
        .enum_init_tuple => {
            try emit(visitor, .{ .flag = n.lhs == Ast.none });
            if (n.lhs != Ast.none) try walkInner(src, n.lhs, false, visitor);
            try emit(visitor, .{ .leaf = leaf });
            const args = Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(args.len) });
            for (args) |a| try walkInner(src, a, collect, visitor);
        },
        .enum_init_struct => {
            try emit(visitor, .{ .flag = n.lhs == Ast.none });
            if (n.lhs != Ast.none) try walkInner(src, n.lhs, false, visitor);
            try emit(visitor, .{ .leaf = leaf });
            const inits = Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(inits.len) });
            for (inits) |fi| try walkInner(src, fi, collect, visitor);
        },
        .match_expr => {
            try walkInner(src, n.lhs, collect, visitor);
            const arms = Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(arms.len) });
            for (arms) |arm| try walkInner(src, arm, collect, visitor);
        },
        .match_arm => {
            // The touched walk never descended the pattern (`n.lhs`); recurse it
            // with `collect=false` so its `.touch` is suppressed while the hash
            // still folds the pattern spelling.
            try walkInner(src, n.lhs, false, visitor);
            const ah = Ast.armHeaderAt(tree, n.rhs.int());
            try emit(visitor, .{ .flag = ah.guard != Ast.none }); // guard sentinel
            if (ah.guard != Ast.none) try walkInner(src, ah.guard, collect, visitor);
            try walkInner(src, ah.body, collect, visitor);
        },
        .pattern_variant => {
            try emit(visitor, .{ .flag = n.lhs == Ast.none }); // inferred vs qualified
            if (n.lhs != Ast.none) try walkInner(src, n.lhs, collect, visitor);
            try emit(visitor, .{ .leaf = leaf });
            const binders = if (n.rhs == Ast.none) &[_]Ast.Index{} else Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(binders.len) });
            for (binders) |b| try walkInner(src, b, collect, visitor);
        },
        .pattern_wildcard => {}, // the tag byte IS its content
        // A poison leaf: inert, no children. The `.enter` tag byte (emitted above)
        // is its whole contribution to the fingerprint, like `pattern_wildcard`.
        .error_node => {},
        // Generics. Unreachable for non-generic source and gated before
        // codegen for generic source (T0013), so these fold nothing load-bearing —
        // but the walk must stay exhaustive and deterministic.
        .generic_param => try emit(visitor, .{ .leaf = leaf }),
        .type_app => {
            try emit(visitor, .{ .leaf = leaf });
            try walkInner(src, n.lhs, collect, visitor);
            const args = Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(args.len) });
            for (args) |a| try walkInner(src, a, collect, visitor);
        },
        .fn_type => {
            const params = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(params.len) }); // fn(int) != fn(int, int)
            for (params) |p| try walkInner(src, p, collect, visitor);
            try emit(visitor, .{ .flag = n.rhs != Ast.none }); // fn(int) != fn(int) -> int
            if (n.rhs != Ast.none) try walkInner(src, n.rhs, collect, visitor);
        },
        // No `type_ref`: that event folds the OWNING fn's sig by ordinal, and a lambda's
        // or a local fn's params are not the owner's. Their type-refs fold by spelling.
        .lambda => try walkClosure(src, n, collect, visitor),
        .local_fn => {
            try emit(visitor, .{ .leaf = leaf });
            try walkClosure(src, n, collect, visitor);
        },
        .pattern_binding => {
            try emit(visitor, .{ .leaf = leaf });
            try emit(visitor, .{ .flag = n.lhs != Ast.none }); // rename vs pun
            if (n.lhs != Ast.none) try walkInner(src, n.lhs, collect, visitor);
            try emit(visitor, .{ .flag = n.rhs != Ast.none }); // has sub-pattern
            if (n.rhs != Ast.none) try walkInner(src, n.rhs, collect, visitor);
        },
        // Postfix `?`: the `.enter` tag byte (emitted above) makes `x?` fp-distinct
        // from `x`; recursing the operand folds its spelling AND discovers its
        // Option/Result `App` instance for monomorphization. The `?` mints no instance of
        // its own — the residual is built from the already-reified enclosing return enum.
        //
        // The `.try_operator` after the operand lets `CallVisitor` fold the resolved `From`
        // witness for a WIDENING Result `?` (operand error type differs from the enclosing
        // return's). CallVisitor reads the enclosing fn's return type from the threaded `fn_sig`
        // (`walkCalls` threads it, mirroring `walkTouchedSig`), so adding/removing an
        // `impl RetErr has From[OpErr]` flips the enclosing fn's codegen key.
        .try_expr => {
            try walkInner(src, n.lhs, collect, visitor);
            try emit(visitor, .{ .try_operator = .{ .idx = idx } });
        },
        // A bodyless C-ABI declaration: fold its name + param/return type-refs (an
        // edit to the extern signature must bust the cache) but never a body or
        // generics. Mirrors `fn_decl`'s proto walk minus the generic/body arms.
        .extern_fn_decl => {
            try emit(visitor, .{ .leaf = leaf });
            const proto = Ast.protoAt(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(proto.params.len) });
            for (proto.params, 0..) |p, i| {
                const pty_node = tree.nodes[p.int()].lhs;
                if (pty_node != Ast.none) try emit(visitor, .{ .type_ref = .{ .idx = pty_node, .ordinal = @intCast(i), .is_ret = false } });
                try walkInner(src, p, collect, visitor);
            }
            try emit(visitor, .{ .flag = proto.ret_type != Ast.none });
            if (proto.ret_type != Ast.none) {
                try emit(visitor, .{ .type_ref = .{ .idx = proto.ret_type, .ordinal = 0, .is_ret = true } });
                try walkInner(src, proto.ret_type, collect, visitor);
            }
        },
        // Transparent over its inner block: `unsafe { .. }` folds and walks exactly
        // as the block does (the `.enter` tag byte keeps it fp-distinct).
        .unsafe_block => try walkInner(src, n.lhs, collect, visitor),
        .pattern_literal => try emit(visitor, .{ .leaf = leaf }),
        .pattern_or => {
            const alts = Ast.rangeSlice(tree, n.lhs.int());
            try emit(visitor, .{ .count = @intCast(alts.len) });
            for (alts) |a| try walkInner(src, a, collect, visitor);
        },
        // A non-empty list literal: fold the element arity (so `[1]` != `[1,2]`) then
        // walk each element (mirroring `.call`'s args tail, minus the callee).
        .list_literal => {
            const elems = Ast.rangeSlice(tree, n.rhs.int());
            try emit(visitor, .{ .count = @intCast(elems.len) });
            for (elems) |el| try walkInner(src, el, collect, visitor);
        },
        // A value index: the `.enter` tag byte keeps it fp-distinct; walk receiver
        // then index (lhs before rhs, like `.binary` minus the operator event).
        .index => {
            try walkInner(src, n.lhs, collect, visitor);
            try walkInner(src, n.rhs, collect, visitor);
        },
    }
}

/// Candidate desugaring method names for a binary operator token, in resolution order
/// (first `.one` witness wins). Empty for a non-desugaring token — which is also the gate
/// deciding whether `walkInner` emits a `.operator` event at all. `==`/`!=` try `eq` then
/// fall back to `cmp` (an Ord-refinement `==` has no `eq`, but its `Ord::cmp` is the reloc
/// target); comparisons map to `cmp`; arithmetic to its per-token method.
pub fn opMethods(tag: TokenTag) []const []const u8 {
    return switch (tag) {
        .eq_eq, .bang_eq => &.{ "eq", "cmp" },
        .lt, .lt_eq, .gt, .gt_eq => &.{"cmp"},
        .plus => &.{"add"},
        .minus => &.{"sub"},
        .star => &.{"mul"},
        .slash => &.{"div"},
        else => &.{},
    };
}
