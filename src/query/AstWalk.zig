//! The ONE place AST visit order is defined.
//!
//! `walk` drives a single, exhaustive `switch` over `Ast.Tag` in canonical
//! LHS->RHS order and emits a typed `Event` stream to a comptime visitor. Three
//! consumers ride this same stream and so CANNOT drift:
//!
//!   * `HashVisitor` folds the structural body fingerprint (tag bytes + leaf text
//!     + arity/optionality sentinels) — byte-identical to what `Fingerprint`
//!     folded when it owned its own walk.
//!   * `CallVisitor` records each callee's `Sig` at a `.callee` event (reading
//!     `frozen.resolutions` at the callee identifier), in body-walk order.
//!   * `TouchedVisitor` records each touched type's layout at a `.touch` event
//!     (reading `frozen.node_types`) plus the owning fn's param/return types at a
//!     `.type_ref` event.
//!
//! A new `Ast.Tag` is a non-exhaustive-switch COMPILE ERROR here, not silent
//! drift across hand-mirrored walks. The two correctness invariants the walk must
//! never violate are unchanged from when `Fingerprint` owned the walk:
//!
//!   * NO ABSOLUTE INDICES, NO SOURCE OFFSETS — the fold sees node TAGS and leaf
//!     token TEXT, never an index/offset that shifts when a sibling is edited.
//!   * ORDER-SENSITIVE, NEVER XOR — children fold in wiring order with arity
//!     sentinels, so `a - b` and `b - a` cannot collide.

const std = @import("std");
const Token = @import("../ast/Token.zig").Token;
const TokenTag = @import("../ast/Token.zig").Tag;
const Ast = @import("../ast/Ast.zig");
const Typecheck = @import("../types.zig");
const Mono = @import("../symbols/Mono.zig");

pub const Sig = @import("../symbols/Sig.zig").Sig;
const Fingerprint = @import("Fingerprint.zig");
pub const TouchedType = Fingerprint.TouchedType;

/// Fixed sentinel params for the builtin scalar `eq` fold. FILE-SCOPE (not a stack
/// temporary) because `Fingerprint.fingerprint` reads each folded `Sig.params` AFTER the
/// call walk returns — a `&.{recv,recv}` local would dangle. `eq` is homogeneous, so the
/// params mirror the receiver Kind; choosing per-kind (int vs bool) is strictly more
/// forward-stable than a single kind-agnostic array at zero cost, and — once chosen —
/// must stay stable or warm caches invalidate.
const eq_params_int = [_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int };
const eq_params_bool = [_]Typecheck.Type{ Typecheck.Type.bool, Typecheck.Type.bool };
// The builtin scalar `hash` sentinel params: 1-ary (`hash(self) -> int`), so a single
// receiver-typed element — distinct from the 2-ary `eq` sentinel above by both count + name.
const hash_params_int = [_]Typecheck.Type{Typecheck.Type.int};
const hash_params_bool = [_]Typecheck.Type{Typecheck.Type.bool};
// The `print` builtin scalar-dispatch sentinel params: `print(x)` on an `int` folds a
// fixed `__display_int` sentinel; on a `bool`, a fixed `display_bool` sentinel — so a
// `print(x)` site's fingerprint distinguishes the arg KIND (int vs bool vs str vs a
// struct/enum's resolved Display witness) for incremental soundness, mirroring the builtin
// scalar `eq`/`hash` sentinels. A struct/enum print folds its resolved witness (see `.callee`).
const display_params_int = [_]Typecheck.Type{Typecheck.Type.int};
const display_params_bool = [_]Typecheck.Type{Typecheck.Type.bool};

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
    /// `@label` target, whose label names a token, not a child node).
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
fn opMethods(tag: TokenTag) []const []const u8 {
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

pub fn updateU32(h: *std.hash.Wyhash, v: u32) void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    h.update(&buf);
}

/// Length-prefixed leaf text, so `"ab"+"c"` cannot collide with `"a"+"bc"`.
pub fn updateLeaf(h: *std.hash.Wyhash, text: []const u8) void {
    updateU32(h, @intCast(text.len));
    h.update(text);
}

/// Folds the structural body fingerprint. The byte stream is identical to the
/// walk `Fingerprint` owned before this module: `enter` -> the tag byte;
/// `leaf`/`raw_leaf` -> length-prefixed text; `count` -> u32 LE; `flag` ->
/// `@intFromBool` as one byte. `touch`/`callee`/`type_ref` are IGNORED (the
/// type-ref node is already folded via its own `enter` recursion, so the hash is
/// byte-identical to today).
pub const HashVisitor = struct {
    h: *std.hash.Wyhash,

    pub fn on(self: *HashVisitor, ev: Event) void {
        switch (ev) {
            .enter => |e| self.h.update(&[_]u8{@intFromEnum(e.tag)}),
            .leaf, .raw_leaf => |t| updateLeaf(self.h, t),
            .count => |c| updateU32(self.h, c),
            .flag => |f| self.h.update(&[_]u8{@intFromBool(f)}),
            .touch, .callee, .type_ref, .operator, .try_operator => {},
        }
    }
};

/// Records each callee's identity-and-signature in body-walk order, so the
/// fingerprint's (b) component lines up positionally with this stream. Generic
/// over the frozen view (was `anytype` duck typing — a missing field is now a
/// compile error, not a runtime panic).
pub fn CallVisitor(comptime Frozen: type) type {
    return struct {
        const Self = @This();
        gpa: std.mem.Allocator,
        frozen: *const Frozen,
        out: *std.ArrayList(Sig),
        /// The OWNING fn's typecheck Sig, threaded by `walkCalls` (mirroring `TouchedVisitor`).
        /// Its `.ret` is the enclosing fn's reified return enum — the source of the target
        /// error type a widening `?` converts INTO. Defaulted `null` so the ~7 test
        /// CallVisitor constructions that don't fold a `?` widen stay unchanged; a `.try_operator`
        /// with no threaded sig folds nothing (`orelse return`).
        fn_sig: ?Sig = null,

        pub fn on(self: *Self, ev: Event) error{OutOfMemory}!void {
            switch (ev) {
                .callee => |c| {
                    // A generic call `id[int](..)`: the callee node is a
                    // `type_app` whose base identifier carries the template's `.func`.
                    // Resolve (template gid + the concrete type-args the checker wrote
                    // into node_types) to the reified instance's mangled name +
                    // substituted sig, so a caller folds the SAME instance identity the
                    // reloc targets. `Mono.find` is a deterministic scan.
                    const cn = self.frozen.tree.nodes[c.idx.int()];
                    if (cn.tag == .type_app) {
                        // An explicit-protocol-args method call `v.into[int]()`: a
                        // `type_app` over a `field_access` NOT bound to a `.func` (a
                        // qualified generic fn binds its field_access to `.func`). Fold the
                        // resolved witness (mirroring the bare method fold below), so the
                        // fingerprint tracks the SAME witness lower emits. Anything else is a
                        // generic FUNCTION call.
                        if (self.frozen.tree.nodes[cn.lhs.int()].tag == .field_access and
                            cn.lhs.int() < self.frozen.resolutions.len and self.frozen.resolutions[cn.lhs.int()] != .func)
                        {
                            try self.foldMethodCalleeExplicit(cn);
                        } else {
                            try self.foldViaInstanceRef(c.call);
                        }
                        return;
                    }
                    const res = self.frozen.resolutions[c.idx.int()];
                    // `res.func` indexes BOTH `names` (the resolved SymName{kind,name},
                    // what the .func reloc target carries) and `sigs` (params/ret).
                    // Fold the full identity so a builtin<->user_fn shadow switch flips
                    // the caller's hash.
                    if (res == .func and res.func < self.frozen.sigs.len and res.func < self.frozen.names.len) {
                        const sig = self.frozen.sigs[res.func];
                        // A bare inferred generic call `id(7)`: a PLAIN-IDENTIFIER
                        // callee resolving to a generic template (its sig params carry
                        // type_vars). Fold the resolved INSTANCE sig — not the template —
                        // so the caller folds the SAME identity the reloc targets (mirroring
                        // the type_app path). The plain-identifier gate matches the other
                        // three sites; a bare qualified generic call is rejected at Pass C.
                        if (self.frozen.tree.nodes[c.idx.int()].tag == .identifier and sig.hasTypeVar()) {
                            try self.foldViaInstanceRef(c.call);
                            return;
                        }
                        const nm = self.frozen.names[res.func];
                        // The `print` builtin: its reloc target depends on the single
                        // arg's TYPE (int->__display_int, bool->inline, struct/enum->the
                        // resolved Display witness), not on the fixed `print` Sig. Fold by the
                        // arg kind so a caller recompiles when the resolved witness changes
                        // (e.g. the arg struct gains an explicit `impl Display`) — the same
                        // incremental-soundness discipline as the `.operator` witness fold.
                        // `str`/`unit` keep the raw `print` write path, so folding the `print`
                        // Sig for them leaves those fingerprints byte-identical (warm cache).
                        if (nm.kind == .builtin and std.mem.eql(u8, nm.name, "print") and
                            try self.foldPrintCallee(c.call)) return;
                        try self.out.append(self.gpa, .{ .kind = nm.kind, .name = nm.name, .params = sig.params, .ret = sig.ret });
                        return;
                    }
                    // A method call `recv.m(..)` on a VALUE receiver: the callee is
                    // a `field_access` NOT bound to a `.func`, and the receiver types to
                    // a concrete struct/enum. Fold the resolved method's Sig so editing
                    // the method's signature/return flips every caller's fingerprint
                    // (stale-cache soundness — mirrors the plain-func fold above).
                    if (cn.tag == .field_access and res != .func and cn.lhs.int() < self.frozen.node_types.len) {
                        const recv = self.frozen.node_types[cn.lhs.int()];
                        if (recv.kind == .@"struct" or recv.kind == .@"enum" or recv.isScalar())
                        {
                            const member = self.frozen.tokens[cn.main_token].text(self.frozen.source);
                            // The SAME multi-conformance resolver the checker + lower use, so
                            // the folded witness matches the reloc target. `.one` is
                            // byte-identical to the earlier single-conformance `findMethod` for
                            // every error-free program (an ambiguous bare call halts the compile
                            // before codegen,
                            // so the fp is never taken); `.ambiguous`/`.none` fold nothing here.
                            switch (Typecheck.resolveConformanceMethod(self.frozen.methods, recv, member, null, null)) {
                                .one => |m| try self.foldWitness(m),
                                .none, .ambiguous => {
                                    // A builtin scalar `eq`/`hash`: it has NO real
                                    // fn_id/instance (the recognizer lowers to an inline machine
                                    // op, no symbol), so fold a FIXED sentinel Sig instead.
                                    // Deterministic + stable across builds; the member NAME +
                                    // per-arity params separate `eq` ([Self,Self]->bool) from
                                    // `hash` ([Self]->int).
                                    if (Typecheck.builtinScalarMethod(recv, member)) |bm| {
                                        if (std.mem.eql(u8, member, "hash")) {
                                            const params: []const Typecheck.Type = if (recv.kind == .bool) &hash_params_bool else &hash_params_int;
                                            try self.out.append(self.gpa, .{ .kind = .builtin, .name = "hash", .params = params, .ret = bm.ret });
                                        } else {
                                            const params: []const Typecheck.Type = if (recv.kind == .bool) &eq_params_bool else &eq_params_int;
                                            try self.out.append(self.gpa, .{ .kind = .builtin, .name = "eq", .params = params, .ret = bm.ret });
                                        }
                                    }
                                },
                            }
                        }
                    }
                },
                .operator => |e| {
                    // A desugaring operator on a struct/enum operand desugars to that type's
                    // witness — `==`/`!=` to `Eq::eq` (or, for an Ord-refinement operand with no
                    // `eq`, its `Ord::cmp`), a comparison to `Ord::cmp`, arithmetic to its
                    // Add/Sub/Mul/Div method. Fold the FIRST resolving witness so the caller
                    // tracks the SAME identity the desugared call's reloc targets — the SAME
                    // `.one` resolver the `recv.m(..)` method fold above uses. A scalar operand
                    // (int/bool/str/unit) inlines to a machine op with NO symbol, so there is
                    // nothing to fold. A pre-typecheck view (lhs out of range or an invalid recv)
                    // folds nothing.
                    const bn = self.frozen.tree.nodes[e.idx.int()];
                    if (bn.lhs.int() >= self.frozen.node_types.len) return;
                    const recv = self.frozen.node_types[bn.lhs.int()];
                    if (recv.kind != .@"struct" and recv.kind != .@"enum") return;
                    for (opMethods(self.frozen.tokens[bn.main_token].tag)) |method| {
                        const pid = Typecheck.witnessProtocolId(self.frozen.prelude_ids, method);
                        switch (Typecheck.resolveConformanceMethod(self.frozen.methods, recv, method, pid, null)) {
                            .one => |m| return self.foldWitness(m),
                            .none, .ambiguous => {},
                        }
                    }
                },
                .try_operator => |e| {
                    // A WIDENING Result `?` re-emits `err(e)` as `RetErr.from(e)`; fold that
                    // `From` witness so the enclosing fn's codegen key depends on the resolved
                    // conformance — via the SAME resolver+args lower's buildResidual uses
                    // (lower.zig:3063), tracking the SAME witness the widen's reloc targets. Only
                    // fires for a Result operand whose error type DIFFERS from the enclosing return's:
                    // the identity case (op_err==ret_err), an Option `?`, and every non-`?` fn fold
                    // NOTHING. Every index is guarded — a pre-typecheck / FakeFrozen view (no threaded
                    // sig or empty tables) folds nothing, mirroring the eq/ord/arith guards above.
                    const fs = self.fn_sig orelse return;
                    if (fs.ret.kind != .@"enum" or fs.ret.enum_id >= self.frozen.enum_layouts.len) return;
                    const tn = self.frozen.tree.nodes[e.idx.int()];
                    if (tn.lhs.int() >= self.frozen.node_types.len) return;
                    const op = self.frozen.node_types[tn.lhs.int()];
                    if (op.kind != .@"enum" or op.enum_id >= self.frozen.enum_layouts.len) return;
                    const ol = self.frozen.enum_layouts[op.enum_id];
                    const rl = self.frozen.enum_layouts[fs.ret.enum_id];
                    if (ol.native_family != .result or rl.native_family != .result) return;
                    if (ol.variants.len < 2 or rl.variants.len < 2) return;
                    if (ol.variants[1].field_types.len < 1 or rl.variants[1].field_types.len < 1) return;
                    const op_err = Typecheck.errPayload(ol);
                    const ret_err = Typecheck.errPayload(rl);
                    if (Typecheck.Type.eql(op_err, ret_err)) return;
                    switch (Typecheck.resolveConformanceMethod(self.frozen.methods, ret_err, "from", self.frozen.prelude_ids.from, &.{op_err})) {
                        .one => |m| try self.foldWitness(m),
                        .none, .ambiguous => {},
                    }
                },
                else => {},
            }
        }

        /// Fold a resolved method witness's identity+sig. A GENERIC-type instance
        /// method carries the mono `instance` index -> fold the INSTANCE identity
        /// (the real reloc target), not the never-lowered template's `names/sigs[fn_id]`.
        fn foldWitness(self: *Self, m: Typecheck.Method) error{OutOfMemory}!void {
            // A SOURCE-LESS derived witness: fold the synthetic unit's identity (its
            // mangled name + `[T,T]->bool` sig) so the caller's fingerprint tracks the
            // derive — a nested-field override flips the unit's key AND, via this fold, the
            // caller's. Checked FIRST (a derive Method has `fn_id == 0`). `dv.params` is a
            // borrowed slice into the stable `derives` table, so storing it is safe.
            if (m.derive) |di| {
                if (di < self.frozen.derives.len) {
                    const dv = &self.frozen.derives[di];
                    // Fold the recipe's TRUE ret (`bool` for an Eq derive, `Ordering` for an
                    // Ord derive) rather than a hardcoded bool, so an Ord-derive witness folds
                    // its real sig. The mangled name already disambiguates, so this is
                    // correctness-neutral — done for cleanliness.
                    try self.out.append(self.gpa, .{ .kind = .user_fn, .name = dv.name.?, .params = dv.params, .ret = dv.ret });
                }
                return;
            }
            if (m.instance) |ii| {
                if (ii < self.frozen.instances.len) {
                    const inst = self.frozen.instances[ii];
                    try self.out.append(self.gpa, .{ .kind = .user_fn, .name = inst.name.?, .params = inst.params, .ret = inst.ret });
                }
            } else if (m.fn_id < self.frozen.names.len and m.fn_id < self.frozen.sigs.len) {
                const nm = self.frozen.names[m.fn_id];
                const sig = self.frozen.sigs[m.fn_id];
                try self.out.append(self.gpa, .{ .kind = nm.kind, .name = nm.name, .params = sig.params, .ret = sig.ret });
            }
        }

        /// Fold the `print` builtin callee by its single arg's type, returning TRUE when
        /// it folded an arg-kind-specific identity so the caller skips the plain `print` fold.
        /// `int` -> a fixed `__display_int` sentinel; `bool` -> a fixed `display_bool` sentinel;
        /// a struct/enum -> its resolved `Display` witness (via `foldWitness`, so gaining an
        /// explicit `impl Display` flips the caller). Returns FALSE for `str`/`unit` (or a
        /// pre-typecheck / non-conforming view) so the caller folds the plain `print` Sig,
        /// leaving `print("..")` fingerprints byte-identical (warm cache preserved).
        fn foldPrintCallee(self: *Self, call_idx: Ast.Index) error{OutOfMemory}!bool {
            const call_node = self.frozen.tree.nodes[call_idx.int()];
            const pargs = Ast.rangeSlice(self.frozen.tree, call_node.rhs.int());
            if (pargs.len != 1 or pargs[0].int() >= self.frozen.node_types.len) return false;
            const at = self.frozen.node_types[pargs[0].int()];
            switch (at.kind) {
                .int => {
                    try self.out.append(self.gpa, .{ .kind = .builtin, .name = "__display_int", .params = &display_params_int, .ret = Typecheck.Type.unit });
                    return true;
                },
                .bool => {
                    try self.out.append(self.gpa, .{ .kind = .builtin, .name = "display_bool", .params = &display_params_bool, .ret = Typecheck.Type.unit });
                    return true;
                },
                .@"struct", .@"enum" => switch (Typecheck.resolveConformanceMethod(self.frozen.methods, at, "display", self.frozen.prelude_ids.display, null)) {
                    .one => |m| {
                        try self.foldWitness(m);
                        return true;
                    },
                    .none, .ambiguous => return false, // plain `print` fold
                },
                else => return false, // str/unit: the raw write path — plain `print` fold
            }
        }

        /// Fold an explicit-protocol-args method callee `v.into[int]()`: resolve the
        /// witness by the type-arg node_types (the SAME rule the checker + lower use), so
        /// the fingerprint tracks the SAME witness the reloc targets.
        fn foldMethodCalleeExplicit(self: *Self, cn: Ast.Node) error{OutOfMemory}!void {
            const fa = self.frozen.tree.nodes[cn.lhs.int()];
            if (fa.lhs.int() >= self.frozen.node_types.len) return;
            const recv = self.frozen.node_types[fa.lhs.int()];
            switch (recv.kind) {
                .@"struct", .@"enum", .int, .bool, .str, .unit => {},
                else => return,
            }
            const member = self.frozen.tokens[fa.main_token].text(self.frozen.source);
            const targ_nodes = Ast.rangeSlice(self.frozen.tree, cn.rhs.int());
            var buf: [8]Typecheck.Type = undefined;
            if (targ_nodes.len > buf.len) return;
            for (targ_nodes, 0..) |tn, i| {
                if (tn.int() >= self.frozen.node_types.len) return; // pre-typecheck view
                buf[i] = self.frozen.node_types[tn.int()];
            }
            switch (Typecheck.resolveConformanceMethod(self.frozen.methods, recv, member, null, buf[0..targ_nodes.len])) {
                .one => |m| try self.foldWitness(m),
                .none, .ambiguous => {},
            }
        }

        /// Recover the generic call's `(gid, type-args)` via the SHARED `callInstanceRef`
        /// (explicit turbofish or bare inference), find the reified instance, and fold its
        /// INSTANCE sig. A miss (pre-typecheck view, arity/conflict/unbound, or an unminted
        /// instance) folds nothing, so exactly one-or-zero Sig per call is preserved and the
        /// fold tracks the SAME `Instance` mono discovery enqueued.
        fn foldViaInstanceRef(self: *Self, call_idx: Ast.Index) error{OutOfMemory}!void {
            const call_node = self.frozen.tree.nodes[call_idx.int()];
            const ref = (try Mono.callInstanceRef(self.gpa, self.frozen, call_node)) orelse return;
            defer self.gpa.free(ref.args);
            const ii = Mono.find(self.frozen.instances, ref.gid, ref.args) orelse return;
            const inst = self.frozen.instances[ii];
            try self.out.append(self.gpa, .{ .kind = .user_fn, .name = inst.name.?, .params = inst.params, .ret = inst.ret });
        }
    };
}

/// Records each touched type's layout in body-walk order: at a `.touch` event the
/// node's `node_types[idx]` (guarded `idx < len`); at a `.type_ref` event the
/// OWNING fn's sig param/return type (falling back to a bare-name re-scan when no
/// sig is threaded). Owns the layout slices it appends (free via
/// `freeTouched`). Generic over the frozen view.
pub fn TouchedVisitor(comptime Frozen: type) type {
    return struct {
        const Self = @This();
        gpa: std.mem.Allocator,
        frozen: *const Frozen,
        fn_sig: ?Sig,
        out: *std.ArrayList(TouchedType),

        pub fn on(self: *Self, ev: Event) error{OutOfMemory}!void {
            switch (ev) {
                .touch => |t| {
                    if (t.idx.int() < self.frozen.node_types.len) try appendTouched(self.gpa, self.frozen, self.frozen.node_types[t.idx.int()], self.out);
                },
                .type_ref => |r| {
                    // Prefer the typecheck-resolved sig type (carries the right
                    // GLOBAL id, incl. a cross-module qualified `mod.Type`); fall
                    // back to the bare-name re-resolution when no sig is threaded.
                    const ty = if (self.fn_sig) |s| (if (r.is_ret)
                        s.ret
                    else if (r.ordinal < s.params.len)
                        s.params[r.ordinal]
                    else
                        typeRefToType(self.frozen, r.idx)) else typeRefToType(self.frozen, r.idx);
                    try appendTouched(self.gpa, self.frozen, ty, self.out);
                },
                else => {},
            }
        }
    };
}

/// Resolve a DECLARED type-ref node (a param/return type annotation) to a Type by
/// name, using the struct/enum tables. Builtins map to their scalar/str kinds; a
/// struct/enum name resolves to its id; anything else (incl. `()`) -> unit.
/// Mirrors `lower`'s type-ref resolution so the fingerprint folds the SAME layout
/// codegen will use, independent of node_types (which Typecheck never sets here).
pub fn typeRefToType(frozen: anytype, type_node: Ast.Index) Typecheck.Type {
    const n = frozen.tree.nodes[type_node.int()];
    if (n.tag == .literal_unit) return Typecheck.Type.unit;
    const name = frozen.tokens[n.main_token].text(frozen.source);
    if (std.mem.eql(u8, name, "int")) return Typecheck.Type.int;
    if (std.mem.eql(u8, name, "bool")) return Typecheck.Type.@"bool";
    if (std.mem.eql(u8, name, "str")) return Typecheck.Type.str;
    for (frozen.layouts, 0..) |l, id| {
        if (std.mem.eql(u8, l.name, name)) return Typecheck.Type.structT(@intCast(id));
    }
    for (frozen.enum_layouts, 0..) |e, id| {
        if (std.mem.eql(u8, e.name, name)) return Typecheck.Type.enumT(@intCast(id));
    }
    return Typecheck.Type.unit;
}

/// Append a `TouchedType` for `ty`, building the struct/enum layout descriptor.
pub fn appendTouched(gpa: std.mem.Allocator, frozen: anytype, ty: Typecheck.Type, out: *std.ArrayList(TouchedType)) !void {
    if (ty.kind == .@"struct") {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        try structLayoutBytes(gpa, frozen, ty.struct_id, &buf);
        try out.append(gpa, .{ .kind = .@"struct", .layout = try buf.toOwnedSlice(gpa) });
        return;
    }
    if (ty.kind == .@"enum") {
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        try enumLayoutBytes(gpa, frozen, ty.enum_id, &buf);
        try out.append(gpa, .{ .kind = .@"enum", .layout = try buf.toOwnedSlice(gpa) });
        return;
    }
    // Only an `.int` carries a meaningful descriptor; every other kind folds the stable
    // default byte (single-sourced via `carriesIntDesc`, so this can't drift from the
    // other identity serializers' int-only rule).
    const int_desc: u8 = if (Typecheck.Type.carriesIntDesc(ty.kind))
        @bitCast(ty.int_desc)
    else
        @bitCast(Typecheck.IntDesc{});
    try out.append(gpa, .{ .kind = ty.kind, .int_desc = int_desc });
}

/// Collect the signatures of every function `idx` calls, in body walk order (the
/// order `Fingerprint.fingerprint`'s (b) component expects) by riding the one AST
/// walk here — so there is no second hand-mirrored walk to drift from the fold.
///
/// `fn_sig` threads in the OWNING fn's signature so the `?`-widen fold can read the
/// enclosing fn's reified return-Result error type — the target a widening `?` converts
/// INTO via `From`. It is the fn's typecheck Sig when known; null elsewhere (a `null` sig
/// folds no `?` witness, so those fingerprints stay byte-identical).
pub fn walkCalls(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, fn_sig: ?Sig, out: *std.ArrayList(Sig)) !void {
    const Frozen = @TypeOf(frozen.*);
    var v = CallVisitor(Frozen){ .gpa = gpa, .frozen = frozen, .fn_sig = fn_sig, .out = out };
    try walk(.{ .tree = frozen.tree, .tokens = frozen.tokens, .source = frozen.source }, idx, &v);
}

/// Collect the types `idx` touches (its node types under the subtree, in walk order),
/// each as a `TouchedType` carrying — for an aggregate — an index-free layout descriptor
/// so a field-layout edit flips every using fn's hash. Feeds the fingerprint's (c)
/// component. Caller frees each `layout` slice (see `freeTouched`).
///
/// `fn_sig` threads in the OWNING fn's signature so the fn_decl case folds the
/// ABI-correct param/return types. Its `params`/`ret` carry the GLOBAL struct/enum ids the
/// typechecker resolved — including a CROSS-MODULE qualified type-ref `b: rect.Rect` which
/// a bare-name scan would otherwise mis-resolve to the FIRST same-named type in the
/// program-wide layout table. Folding the sig types makes a pub-type LAYOUT edit reach
/// EXACTLY the importers that name it.
pub fn walkTouchedSig(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, fn_sig: ?Sig, out: *std.ArrayList(TouchedType)) error{OutOfMemory}!void {
    const Frozen = @TypeOf(frozen.*);
    var v = TouchedVisitor(Frozen){ .gpa = gpa, .frozen = frozen, .fn_sig = fn_sig, .out = out };
    try walk(.{ .tree = frozen.tree, .tokens = frozen.tokens, .source = frozen.source }, idx, &v);
}

/// Free the layout slices owned by a `walkTouchedSig` result.
pub fn freeTouched(gpa: std.mem.Allocator, items: []const TouchedType) void {
    for (items) |t| if (t.layout.len > 0) gpa.free(t.layout);
}

/// Build the ordered `TouchedType` list for a monomorphized instance's concrete
/// type-args, each carrying its full index-free layout descriptor via the same
/// `appendTouched` the (c) touched fold uses. Fed into `fingerprint`'s (d) component so
/// `id[int]` and `id[Point]` diverge and a struct-layout edit to a type-arg invalidates
/// exactly the dependent instance. Empty for a non-generic fn. Caller frees via `freeTouched`.
pub fn walkTypeArgs(gpa: std.mem.Allocator, frozen: anytype, args: []const @import("../layout/Type.zig").Type, out: *std.ArrayList(TouchedType)) error{OutOfMemory}!void {
    for (args) |a| try appendTouched(gpa, frozen, a, out);
}

/// Build the ordered `Fingerprint.ResolvedConformance` list for a monomorphized
/// instance's resolved `[T has P]` bounds, fed into `fingerprint`'s (e) component. Each
/// entry's `conform` is the conforming type's index-free layout descriptor built by the
/// SAME `appendTouched` the type-arg (d) fold uses; `protocol_name` and `witness_syms` are
/// borrowed from the `Mono.ResolvedConformance` (both outlive codegen). Empty for a
/// non-bounded instance. Caller frees via `freeConformances`.
pub fn walkConformances(gpa: std.mem.Allocator, frozen: anytype, conformances: []const Mono.ResolvedConformance, out: *std.ArrayList(Fingerprint.ResolvedConformance)) error{OutOfMemory}!void {
    for (conformances) |rc| {
        var tmp: std.ArrayList(TouchedType) = .empty;
        defer tmp.deinit(gpa);
        try appendTouched(gpa, frozen, rc.conform_ty, &tmp);
        // The conformance's protocol type-args, each as an index-free layout descriptor
        // built by the SAME `appendTouched` -> the (e) fold distinguishes `Into[int]` from
        // `Into[bool]`. OWNED (freed via `freeConformances`).
        var pargs: std.ArrayList(TouchedType) = .empty;
        errdefer {
            freeTouched(gpa, pargs.items);
            pargs.deinit(gpa);
        }
        for (rc.protocol_args) |pa| try appendTouched(gpa, frozen, pa, &pargs);
        // Reserve the out slot BEFORE detaching pargs so the append can't fail once
        // ownership of pargs_owned + tmp.items[0].layout has left the errdefer's cover.
        try out.ensureUnusedCapacity(gpa, 1);
        const pargs_owned = try pargs.toOwnedSlice(gpa);
        out.appendAssumeCapacity(.{ .protocol_name = rc.protocol_name, .conform = tmp.items[0], .witness_syms = rc.witness_syms, .protocol_args = pargs_owned });
    }
}

/// Free the conform-layout + protocol-arg-layout slices owned by a `walkConformances`
/// result (the outer list is caller-owned; `protocol_name`/`witness_syms` are borrowed).
pub fn freeConformances(gpa: std.mem.Allocator, items: []const Fingerprint.ResolvedConformance) void {
    for (items) |c| {
        if (c.conform.layout.len > 0) gpa.free(c.conform.layout);
        freeTouched(gpa, c.protocol_args);
        if (c.protocol_args.len > 0) gpa.free(@constCast(c.protocol_args));
    }
}

/// Index-free struct layout descriptor: name + per-field (name, kind, offset),
/// recursing nested structs, + size + align. Editing any of these flips the bytes.
pub fn structLayoutBytes(gpa: std.mem.Allocator, frozen: anytype, id: u32, buf: *std.ArrayList(u8)) !void {
    const l = frozen.layouts[id];
    try buf.appendSlice(gpa, l.name);
    try buf.append(gpa, 0);
    for (l.field_names, l.field_types, l.offsets) |fn_, fty, off| {
        try buf.appendSlice(gpa, fn_);
        try buf.append(gpa, 0);
        try buf.append(gpa, @intFromEnum(fty.kind));
        var ob: [4]u8 = undefined;
        std.mem.writeInt(u32, &ob, off, .little);
        try buf.appendSlice(gpa, &ob);
        if (fty.kind == .@"struct") try structLayoutBytes(gpa, frozen, fty.struct_id, buf);
    }
    var sz: [8]u8 = undefined;
    std.mem.writeInt(u32, sz[0..4], l.size, .little);
    std.mem.writeInt(u32, sz[4..8], l.@"align", .little);
    try buf.appendSlice(gpa, &sz);
}

/// Index-free enum layout descriptor: name + tag_size + payload_off + per-variant
/// (name + form byte + per payload field (name + kind + payload-local offset,
/// recursing nested struct/enum)) + size + align. Editing any variant/payload
/// flips the bytes, recompiling every using fn (cache soundness).
pub fn enumLayoutBytes(gpa: std.mem.Allocator, frozen: anytype, id: u32, buf: *std.ArrayList(u8)) !void {
    const e = frozen.enum_layouts[id];
    try buf.appendSlice(gpa, e.name);
    try buf.append(gpa, 0);
    var hdr: [8]u8 = undefined;
    std.mem.writeInt(u32, hdr[0..4], e.tag_size, .little);
    std.mem.writeInt(u32, hdr[4..8], e.payload_off, .little);
    try buf.appendSlice(gpa, &hdr);
    for (e.variants) |v| {
        try buf.appendSlice(gpa, v.name);
        try buf.append(gpa, 0);
        try buf.append(gpa, @intFromEnum(v.form));
        if (v.form == .@"struct") {
            for (v.field_names, v.field_types, v.offsets) |fn_, fty, off| {
                try buf.appendSlice(gpa, fn_);
                try buf.append(gpa, 0);
                try buf.append(gpa, @intFromEnum(fty.kind));
                var ob: [4]u8 = undefined;
                std.mem.writeInt(u32, &ob, off, .little);
                try buf.appendSlice(gpa, &ob);
                if (fty.kind == .@"struct") try structLayoutBytes(gpa, frozen, fty.struct_id, buf);
                if (fty.kind == .@"enum") try enumLayoutBytes(gpa, frozen, fty.enum_id, buf);
            }
        } else {
            for (v.field_types, v.offsets) |fty, off| {
                try buf.append(gpa, @intFromEnum(fty.kind));
                var ob: [4]u8 = undefined;
                std.mem.writeInt(u32, &ob, off, .little);
                try buf.appendSlice(gpa, &ob);
                if (fty.kind == .@"struct") try structLayoutBytes(gpa, frozen, fty.struct_id, buf);
                if (fty.kind == .@"enum") try enumLayoutBytes(gpa, frozen, fty.enum_id, buf);
            }
        }
    }
    var sz: [8]u8 = undefined;
    std.mem.writeInt(u32, sz[0..4], e.size, .little);
    std.mem.writeInt(u32, sz[4..8], e.@"align", .little);
    try buf.appendSlice(gpa, &sz);
}

const testing = std.testing;
const Lexer = @import("../lex.zig");
const Parser = @import("../parse.zig");

const Built = struct {
    tokens: []Token,
    tree: Ast.Tree,
    source: []const u8,

    fn deinit(self: *Built, gpa: std.mem.Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.tree.nodes);
        gpa.free(self.tree.extra);
    }

    fn fnDecl(self: *const Built, idx: usize) Ast.Index {
        const prog = self.tree.nodes[Ast.root(self.tree.nodes).int()];
        return Ast.rangeSlice(self.tree, prog.lhs.int())[idx];
    }

    fn src(self: *const Built) Source {
        return .{ .tree = self.tree, .tokens = self.tokens, .source = self.source };
    }
};

fn build(gpa: std.mem.Allocator, source: []const u8) !Built {
    const tokens = try Lexer.tokenize(gpa, source);
    errdefer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, source);
    return .{ .tokens = tokens, .tree = tree, .source = source };
}

/// The position-specific events the drift guard pins, paired with the node `idx`
/// each fires at. Recording the FULL ordered stream (not just `enter`) is what
/// makes the guard catch a mis-placed `.touch`/`.callee`/`.type_ref` dispatch —
/// not only a forked traversal order.
const StreamStep = struct {
    kind: enum { enter, touch, callee, type_ref, operator, try_operator },
    idx: Ast.Index,
};

/// Records the ordered `(kind, idx)` event stream — the load-bearing thing all
/// three real consumers must agree on. Wrapped around each real consumer in the
/// drift guard so any divergence (forked order OR a moved dispatch) is a failing
/// test, not a comment. `leaf`/`count`/`flag` are pure fold bytes with no node
/// identity, so they are not recorded here (the Fingerprint hash tests pin those).
const StreamRecorder = struct {
    out: *std.ArrayList(StreamStep),
    gpa: std.mem.Allocator,
    pub fn on(self: *StreamRecorder, ev: Event) error{OutOfMemory}!void {
        switch (ev) {
            .enter => |e| try self.out.append(self.gpa, .{ .kind = .enter, .idx = e.idx }),
            .touch => |t| try self.out.append(self.gpa, .{ .kind = .touch, .idx = t.idx }),
            .callee => |c| try self.out.append(self.gpa, .{ .kind = .callee, .idx = c.idx }),
            .type_ref => |r| try self.out.append(self.gpa, .{ .kind = .type_ref, .idx = r.idx }),
            .operator => |e| try self.out.append(self.gpa, .{ .kind = .operator, .idx = e.idx }),
            .try_operator => |e| try self.out.append(self.gpa, .{ .kind = .try_operator, .idx = e.idx }),
            .leaf, .raw_leaf, .count, .flag => {},
        }
    }
};

// A frozen view with all the fields the three visitors read, so the boundary test
// can drive CallVisitor/TouchedVisitor over a real parse. Empty resolution/type
// tables make the consumers no-op on appends while STILL walking every node, so
// the enter-order parity is what the test isolates.
const FakeFrozen = struct {
    tree: Ast.Tree,
    tokens: []const Token,
    source: []const u8,
    resolutions: []const @import("../symbols/Resolution.zig").Resolution = &.{},
    node_types: []const Typecheck.Type = &.{},
    layouts: []const Typecheck.Layout = &.{},
    enum_layouts: []const Typecheck.EnumLayout = &.{},
    names: []const @import("../link/Link.zig").SymName = &.{},
    sigs: []const Sig = &.{},
    instances: []const Mono.Instance = &.{},
    methods: []const Typecheck.Method = &.{},
    derives: []const Typecheck.DeriveRecipe = &.{},
    prelude_ids: Typecheck.PreludeProtocolIds = .{},

    pub fn genericTemplate(self: *const FakeFrozen, gid: u32) ?Mono.TemplateRef {
        if (gid >= self.sigs.len) return null;
        const sig = self.sigs[gid];
        if (!sig.hasTypeVar()) return null;
        return .{ .params = sig.params, .count = sig.genericParamCount() };
    }
};

test "[DRIFT GUARD] all three consumers observe the SAME event stream + dispatch positions" {
    const gpa = testing.allocator;
    // One fn exercising every shape: nested calls, binary a-b, struct + enum init,
    // a match with guard + or-pattern + sub-pattern, a labeled loop with break @l,
    // and a struct param + struct return. The match arm patterns are the load-
    // bearing `collect=false` case (their subtrees must NOT emit `.touch`).
    var b = try build(gpa,
        \\struct P { x: int }
        \\enum E { A, C(int) }
        \\fn g(p: P) -> P { return p }
        \\fn f(p: P, e: E) -> P {
        \\  q := g(g(p))
        \\  d := q.x - p.x
        \\  s := P { x: d }
        \\  eqp := q == s
        \\  m := match e { .A | .C(0) -> 1, .C(r) -> r }
        \\  @l loop { if m > 0 { break @l 1 } else { break @l 2 } }
        \\  return s
        \\}
    );
    defer b.deinit(gpa);
    const decl = b.fnDecl(3); // f (decls: struct P=0, enum E=1, fn g=2, fn f=3)

    // Size the frozen reads to the tree (production always has one entry per node),
    // so CallVisitor/TouchedVisitor read in-bounds. All unresolved/invalid, so no
    // appends fire — the test isolates the EVENT stream, not collection content.
    const resolutions = try gpa.alloc(@import("../symbols/Resolution.zig").Resolution, b.tree.nodes.len);
    defer gpa.free(resolutions);
    @memset(resolutions, .unresolved);
    const node_types = try gpa.alloc(Typecheck.Type, b.tree.nodes.len);
    defer gpa.free(node_types);
    @memset(node_types, Typecheck.Type{ .kind = .invalid });
    var frozen = FakeFrozen{ .tree = b.tree, .tokens = b.tokens, .source = b.source, .resolutions = resolutions, .node_types = node_types };

    var hash_stream: std.ArrayList(StreamStep) = .empty;
    defer hash_stream.deinit(gpa);
    var call_stream: std.ArrayList(StreamStep) = .empty;
    defer call_stream.deinit(gpa);
    var touched_stream: std.ArrayList(StreamStep) = .empty;
    defer touched_stream.deinit(gpa);

    // A combined visitor: the real consumer PLUS a full-stream recorder, so the
    // exact production walk (callee/touch/type_ref dispatch included) is recorded.
    const HashProbe = struct {
        rec: StreamRecorder,
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Event) error{OutOfMemory}!void {
            try self.rec.on(ev);
            self.h.on(ev);
        }
    };
    const CallProbe = struct {
        rec: StreamRecorder,
        c: CallVisitor(FakeFrozen),
        pub fn on(self: *@This(), ev: Event) error{OutOfMemory}!void {
            try self.rec.on(ev);
            try self.c.on(ev);
        }
    };
    const TouchedProbe = struct {
        rec: StreamRecorder,
        t: TouchedVisitor(FakeFrozen),
        pub fn on(self: *@This(), ev: Event) error{OutOfMemory}!void {
            try self.rec.on(ev);
            try self.t.on(ev);
        }
    };

    var h = std.hash.Wyhash.init(0);
    var hp = HashProbe{ .rec = .{ .out = &hash_stream, .gpa = gpa }, .h = .{ .h = &h } };
    try walk(b.src(), decl, &hp);

    var sigs: std.ArrayList(Sig) = .empty;
    defer sigs.deinit(gpa);
    var cp = CallProbe{ .rec = .{ .out = &call_stream, .gpa = gpa }, .c = .{ .gpa = gpa, .frozen = &frozen, .out = &sigs } };
    try walk(b.src(), decl, &cp);

    var touched: std.ArrayList(TouchedType) = .empty;
    defer touched.deinit(gpa);
    var tp = TouchedProbe{ .rec = .{ .out = &touched_stream, .gpa = gpa }, .t = .{ .gpa = gpa, .frozen = &frozen, .fn_sig = null, .out = &touched } };
    try walk(b.src(), decl, &tp);

    // The three consumers ride ONE walk, so their FULL event streams (enter +
    // touch + callee + type_ref, by kind AND idx) are identical. Comparing the
    // whole stream — not just `enter` — means a future edit that forks the order
    // OR moves a `.touch`/`.callee`/`.type_ref` dispatch for one consumer fails.
    try testing.expectEqualSlices(StreamStep, hash_stream.items, call_stream.items);
    try testing.expectEqualSlices(StreamStep, hash_stream.items, touched_stream.items);

    // Below: pin the dispatch POSITIONS so a mis-placed event is caught, not just a
    // divergence between consumers. One stream suffices (all three are equal).
    const stream = hash_stream.items;

    // Sanity: a non-trivial shape, and every kind actually fired.
    try testing.expect(stream.len > 40);
    var saw_enter = false;
    var saw_touch = false;
    var saw_callee = false;
    var saw_type_ref = false;
    // The fixture exercises all three operator categories: `q == s` (eq), `m > 0` (ord),
    // `q.x - p.x` (arith) — all now one `.operator` event, distinguished by token tag.
    var saw_eq_op = false;
    var saw_ord_op = false;
    var saw_arith_op = false;
    for (stream) |st| switch (st.kind) {
        .enter => saw_enter = true,
        .touch => saw_touch = true,
        .callee => saw_callee = true,
        .type_ref => saw_type_ref = true,
        .operator => switch (b.tokens[b.tree.nodes[st.idx.int()].main_token].tag) {
            .eq_eq, .bang_eq => saw_eq_op = true,
            .lt, .lt_eq, .gt, .gt_eq => saw_ord_op = true,
            .plus, .minus, .star, .slash => saw_arith_op = true,
            else => {},
        },
        .try_operator => {}, // this fixture has no `?`, so it never fires (proved by the placement loop below being vacuous)
    };
    try testing.expect(saw_enter and saw_touch and saw_callee and saw_type_ref and saw_eq_op and saw_ord_op and saw_arith_op);

    // (1) NO ORPHAN DISPATCH: every touch/callee/type_ref idx is a node that was
    // entered. A dispatch at a node outside the walked subtree would fail here.
    for (stream) |st| {
        if (st.kind == .enter) continue;
        var entered = false;
        for (stream) |e| {
            if (e.kind == .enter and e.idx == st.idx) {
                entered = true;
                break;
            }
        }
        try testing.expect(entered);
    }

    // (2) SUPPRESSION INVARIANT (`collect=false`): a `match_arm` pattern subtree
    // must NOT emit `.touch`, even though every pattern node IS entered. This is
    // the load-bearing thing `collect` controls; a future edit that forgets to
    // turn collection off would add a pattern node to the touch set and fail.
    var pattern_entered: u32 = 0;
    for (stream) |st| {
        if (st.kind != .enter) continue;
        if (isPatternTag(b.tree.nodes[st.idx.int()].tag)) {
            pattern_entered += 1;
            // This pattern node was entered; assert it is NOT in the touch set.
            for (stream) |t| {
                if (t.kind == .touch) try testing.expect(t.idx != st.idx);
            }
        }
    }
    // The fixture's match arms have an or-pattern, a literal sub-pattern, and a
    // binding — several pattern nodes — so suppression is genuinely exercised.
    try testing.expect(pattern_entered >= 3);

    // (3) `.callee` PLACEMENT: every `.callee` idx is the `lhs` (callee subtree) of
    // some entered `.call` node. A callee recorded at the wrong node fails here.
    for (stream) |st| {
        if (st.kind != .callee) continue;
        var matched = false;
        for (stream) |e| {
            if (e.kind == .enter and b.tree.nodes[e.idx.int()].tag == .call and b.tree.nodes[e.idx.int()].lhs == st.idx) {
                matched = true;
                break;
            }
        }
        try testing.expect(matched);
    }

    // (4) `.type_ref` PLACEMENT: every `.type_ref` idx is a param/return type node
    // of the fn_decl's proto. A type_ref emitted at the wrong position fails here.
    const proto = Ast.protoAt(b.tree, b.tree.nodes[decl.int()].lhs.int());
    for (stream) |st| {
        if (st.kind != .type_ref) continue;
        var is_proto_type = st.idx == proto.ret_type;
        for (proto.params) |p| {
            if (b.tree.nodes[p.int()].lhs == st.idx) is_proto_type = true;
        }
        try testing.expect(is_proto_type);
    }

    // (5) `.operator` PLACEMENT: every `.operator` idx is an entered `.binary` node whose
    // operator token has a candidate desugaring method. A signal at the wrong node — or
    // fired for a non-desugaring binary — fails here.
    for (stream) |st| {
        if (st.kind != .operator) continue;
        const bnode = b.tree.nodes[st.idx.int()];
        try testing.expect(bnode.tag == .binary);
        try testing.expect(opMethods(b.tokens[bnode.main_token].tag).len > 0);
    }
}

/// True for the pattern-subtree tags the touched walk descends with `collect=false`
/// (so they are entered + folded by the hash, but never emit `.touch`).
fn isPatternTag(tag: Ast.Node.Tag) bool {
    return switch (tag) {
        .pattern_variant, .pattern_wildcard, .pattern_binding, .pattern_literal, .pattern_or => true,
        else => false,
    };
}

test "[LEAK GUARD] TouchedVisitor layout slices round-trip through freeTouched" {
    const gpa = testing.allocator;
    // A fn touching a struct AND an enum so appendTouched allocates layout slices;
    // testing.allocator fails the test on any leak.
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    const layouts = [_]Typecheck.Layout{.{
        .name = "P",
        .field_names = @constCast(&[_][]const u8{"x"}),
        .field_types = @constCast(&[_]Typecheck.Type{Typecheck.Type.int}),
        .offsets = @constCast(&[_]u32{0}),
        .size = 8,
        .@"align" = 8,
    }};
    const variants = [_]Typecheck.VariantLayout{.{
        .name = "A",
        .form = .unit,
        .field_names = &.{},
        .field_types = &.{},
        .offsets = &.{},
    }};
    const enum_layouts = [_]Typecheck.EnumLayout{.{
        .name = "E",
        .tag_size = 1,
        .payload_off = 8,
        .variants = @constCast(&variants),
        .size = 8,
        .@"align" = 8,
    }};

    var touched: std.ArrayList(TouchedType) = .empty;
    defer touched.deinit(gpa);
    defer freeTouched(gpa, touched.items);

    var frozen = FakeFrozen{ .tree = b.tree, .tokens = b.tokens, .source = b.source, .layouts = &layouts, .enum_layouts = &enum_layouts };
    // A sig naming the struct param and enum return drives both layout builders
    // through the type_ref path.
    const sig = Sig{ .kind = .user_fn, .name = "f", .params = &.{Typecheck.Type.structT(0)}, .ret = Typecheck.Type.enumT(0) };
    var tv = TouchedVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .fn_sig = sig, .out = &touched };
    try walk(b.src(), decl, &tv);

    // At least the struct param and enum return were collected with owned layouts.
    var saw_struct = false;
    var saw_enum = false;
    for (touched.items) |t| {
        if (t.kind == .@"struct") saw_struct = true;
        if (t.kind == .@"enum") saw_enum = true;
    }
    try testing.expect(saw_struct);
    try testing.expect(saw_enum);
}

test "[CROSS-MODULE TYPE-REF] threaded fn_sig folds the sig's type, not a bare-name first-match" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(p: int) -> int {\n return p\n}\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Two same-named structs in the layout table; the threaded sig points at the
    // SECOND. typeRefToType's bare-name scan would pick the FIRST.
    const first = Typecheck.Layout{
        .name = "Rect",
        .field_names = @constCast(&[_][]const u8{"a"}),
        .field_types = @constCast(&[_]Typecheck.Type{Typecheck.Type.int}),
        .offsets = @constCast(&[_]u32{0}),
        .size = 8,
        .@"align" = 8,
    };
    const second = Typecheck.Layout{
        .name = "Rect",
        .field_names = @constCast(&[_][]const u8{ "w", "h" }),
        .field_types = @constCast(&[_]Typecheck.Type{ Typecheck.Type.int, Typecheck.Type.int }),
        .offsets = @constCast(&[_]u32{ 0, 8 }),
        .size = 16,
        .@"align" = 8,
    };
    const layouts = [_]Typecheck.Layout{ first, second };

    // Expected bytes: the SECOND struct's layout.
    var expect_buf: std.ArrayList(u8) = .empty;
    defer expect_buf.deinit(gpa);
    var fr = FakeFrozen{ .tree = b.tree, .tokens = b.tokens, .source = b.source, .layouts = &layouts };
    try structLayoutBytes(gpa, &fr, 1, &expect_buf);

    var touched: std.ArrayList(TouchedType) = .empty;
    defer touched.deinit(gpa);
    defer for (touched.items) |t| {
        if (t.layout.len > 0) gpa.free(t.layout);
    };
    // The param's bare type-ref node names "int" (so a bare-name scan can't find
    // Rect at all); the sig says param 0 is Rect#1. Folding the sig is what reaches
    // the right cross-module type.
    const sig = Sig{ .kind = .user_fn, .name = "f", .params = &.{Typecheck.Type.structT(1)}, .ret = Typecheck.Type.int };
    var tv = TouchedVisitor(FakeFrozen){ .gpa = gpa, .frozen = &fr, .fn_sig = sig, .out = &touched };
    try walk(b.src(), decl, &tv);

    var found: ?[]const u8 = null;
    for (touched.items) |t| {
        if (t.kind == .@"struct") found = t.layout;
    }
    try testing.expect(found != null);
    try testing.expectEqualSlices(u8, expect_buf.items, found.?);
}

test "the .operator event is IGNORED by the hash (int == fingerprint unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int, b: int) -> bool { a == b }\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.operator` after the operands.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk fed to a visitor that DROPS `.operator` before folding —
    // i.e. the hash as if the operator event had never been introduced. Byte-identity proves
    // the operator event never enters the fingerprint byte-stream, so every existing
    // int/bool/str/unit `==` fp is preserved and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Event) void {
            switch (ev) {
                .operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "the .operator event is IGNORED by the hash (int < fingerprint unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int, b: int) -> bool { a < b }\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.operator` after the operands.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk with `.operator` dropped before folding — the hash as if the
    // operator event had never been introduced. Byte-identity proves the operator event never
    // enters the fingerprint byte-stream, so every existing int/str/bool comparison fp is
    // preserved and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Event) void {
            switch (ev) {
                .operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "the .operator event is IGNORED by the hash (int + fingerprint unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa, "fn f(a: int, b: int) -> int { a + b }\n");
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.operator` after the operands.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk with `.operator` dropped before folding — the hash as if the
    // operator event had never been introduced. Byte-identity proves the operator event never
    // enters the fingerprint byte-stream, so every existing int arithmetic fp is preserved
    // and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Event) void {
            switch (ev) {
                .operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "`p == q` folds the SAME Eq witness Sig as `p.eq(q)` (fingerprint agreement)" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\struct P { x: int }
        \\impl P has Eq { fn eq(self, o: P) -> bool { self.x == o.x } }
        \\fn use_op(p: P, q: P) -> bool { p == q }
        \\fn use_method(p: P, q: P) -> bool { p.eq(q) }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    defer tc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = FakeFrozen{
        .tree = tree,
        .tokens = tokens,
        .source = src,
        .resolutions = res.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sigs = tc.sigs,
        .instances = tc.instances,
        .methods = tc.methods,
        .prelude_ids = tc.prelude_ids,
    };

    // The only TOP-LEVEL fn_decls are use_op then use_method (the impl's `eq` is nested
    // inside the impl_has_decl, not a program child), so they arrive in source order.
    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    var op_decl: Ast.Index = Ast.none;
    var meth_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, prog.lhs.int())) |idx| {
        if (tree.nodes[idx.int()].tag != .fn_decl) continue;
        if (op_decl == Ast.none) op_decl = idx else meth_decl = idx;
    }
    try testing.expect(op_decl != Ast.none and meth_decl != Ast.none);

    var op_sigs: std.ArrayList(Sig) = .empty;
    defer op_sigs.deinit(gpa);
    var vop = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .out = &op_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, op_decl, &vop);

    var meth_sigs: std.ArrayList(Sig) = .empty;
    defer meth_sigs.deinit(gpa);
    var vm = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .out = &meth_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, meth_decl, &vm);

    // Both fold exactly one witness Sig, and it is the SAME one: `p == q` desugars to the
    // same `Eq::eq` `p.eq(q)` dispatches to, so the fingerprints track one identity.
    try testing.expectEqual(@as(usize, 1), op_sigs.items.len);
    try testing.expectEqual(@as(usize, 1), meth_sigs.items.len);
    const a = op_sigs.items[0];
    const m = meth_sigs.items[0];
    try testing.expectEqual(a.kind, m.kind);
    try testing.expectEqualStrings(a.name, m.name);
    try testing.expectEqual(a.ret.kind, m.ret.kind);
    try testing.expectEqual(a.params.len, m.params.len);
}

test "`a < b` folds the SAME Ord::cmp witness Sig as `a.cmp(b)` (incremental soundness)" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\struct P { x: int }
        \\impl P has Ord {
        \\ fn cmp(self, o: P) -> Ordering {
        \\  if self.x < o.x { Ordering.lt } else if self.x == o.x { Ordering.eq } else { Ordering.gt }
        \\ }
        \\}
        \\fn use_op(a: P, b: P) -> bool { a < b }
        \\fn use_method(a: P, b: P) -> Ordering { a.cmp(b) }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    defer tc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = FakeFrozen{
        .tree = tree,
        .tokens = tokens,
        .source = src,
        .resolutions = res.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sigs = tc.sigs,
        .instances = tc.instances,
        .methods = tc.methods,
        .prelude_ids = tc.prelude_ids,
    };

    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    var op_decl: Ast.Index = Ast.none;
    var meth_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, prog.lhs.int())) |idx| {
        if (tree.nodes[idx.int()].tag != .fn_decl) continue;
        if (op_decl == Ast.none) op_decl = idx else meth_decl = idx;
    }
    try testing.expect(op_decl != Ast.none and meth_decl != Ast.none);

    var op_sigs: std.ArrayList(Sig) = .empty;
    defer op_sigs.deinit(gpa);
    var vop = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .out = &op_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, op_decl, &vop);

    var meth_sigs: std.ArrayList(Sig) = .empty;
    defer meth_sigs.deinit(gpa);
    var vm = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .out = &meth_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, meth_decl, &vm);

    // `a < b` folds the SAME `Ord::cmp` witness `a.cmp(b)` dispatches to, so editing the `cmp`
    // body invalidates a `<` caller. Each folds the cmp witness exactly once.
    try testing.expectEqual(@as(usize, 1), op_sigs.items.len);
    try testing.expectEqual(@as(usize, 1), meth_sigs.items.len);
    const a = op_sigs.items[0];
    const m = meth_sigs.items[0];
    try testing.expectEqual(a.kind, m.kind);
    try testing.expectEqualStrings(a.name, m.name);
    try testing.expectEqual(a.ret.kind, m.ret.kind);
    try testing.expectEqual(a.params.len, m.params.len);
}

test "`p + q` folds the SAME Add witness Sig as `p.add(q)` (incremental soundness)" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\struct V2 { x: int, y: int }
        \\impl V2 has Add { fn add(self, o: V2) -> V2 { V2{ x: self.x + o.x, y: self.y + o.y } } }
        \\fn use_op(p: V2, q: V2) -> V2 { p + q }
        \\fn use_method(p: V2, q: V2) -> V2 { p.add(q) }
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    defer tc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = FakeFrozen{
        .tree = tree,
        .tokens = tokens,
        .source = src,
        .resolutions = res.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sigs = tc.sigs,
        .instances = tc.instances,
        .methods = tc.methods,
        .prelude_ids = tc.prelude_ids,
    };

    const prog = tree.nodes[Ast.root(tree.nodes).int()];
    var op_decl: Ast.Index = Ast.none;
    var meth_decl: Ast.Index = Ast.none;
    for (Ast.rangeSlice(.{ .nodes = tree.nodes, .extra = tree.extra }, prog.lhs.int())) |idx| {
        if (tree.nodes[idx.int()].tag != .fn_decl) continue;
        if (op_decl == Ast.none) op_decl = idx else meth_decl = idx;
    }
    try testing.expect(op_decl != Ast.none and meth_decl != Ast.none);

    var op_sigs: std.ArrayList(Sig) = .empty;
    defer op_sigs.deinit(gpa);
    var vop = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .out = &op_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, op_decl, &vop);

    var meth_sigs: std.ArrayList(Sig) = .empty;
    defer meth_sigs.deinit(gpa);
    var vm = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .out = &meth_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, meth_decl, &vm);

    // `p + q` folds the SAME `Add::add` witness `p.add(q)` dispatches to, so editing the `add`
    // body invalidates a `+` caller (the stale-cache-soundness guarantee). Each folds once.
    try testing.expectEqual(@as(usize, 1), op_sigs.items.len);
    try testing.expectEqual(@as(usize, 1), meth_sigs.items.len);
    const a = op_sigs.items[0];
    const m = meth_sigs.items[0];
    try testing.expectEqual(a.kind, m.kind);
    try testing.expectEqualStrings(a.name, m.name);
    try testing.expectEqual(a.ret.kind, m.ret.kind);
    try testing.expectEqual(a.params.len, m.params.len);
}

test "the .try_operator event is IGNORED by the hash (a `?` fingerprint is unchanged)" {
    const gpa = testing.allocator;
    var b = try build(gpa,
        \\fn f(r: Result[int, int]) -> Result[int, int] {
        \\  v := r?
        \\  return Result.ok(v)
        \\}
    );
    defer b.deinit(gpa);
    const decl = b.fnDecl(0);

    // Hash 1: the real walk, which emits `.try_operator` after the `?` operand.
    var h1 = std.hash.Wyhash.init(0);
    var hv = HashVisitor{ .h = &h1 };
    try walk(b.src(), decl, &hv);

    // Hash 2: the SAME walk with `.try_operator` dropped before folding — the hash as if the
    // event had never been introduced. Byte-identity proves the operator event never
    // enters the fingerprint byte-stream, so every existing `?` fp (Option `?`, same-error
    // Result `?`, and this widening one) is preserved and warm caches never churn.
    var h2 = std.hash.Wyhash.init(0);
    const Filter = struct {
        h: HashVisitor,
        pub fn on(self: *@This(), ev: Event) void {
            switch (ev) {
                .try_operator => {},
                else => self.h.on(ev),
            }
        }
    };
    var fv = Filter{ .h = .{ .h = &h2 } };
    try walk(b.src(), decl, &fv);

    try testing.expectEqual(h1.final(), h2.final());
}

test "a WIDENING `?` folds the resolved `From` witness; the identity `?` folds nothing" {
    const gpa = testing.allocator;
    const Graph = @import("../driver/Graph.zig");
    const ResolveGraph = @import("../resolve_graph.zig");
    const TypecheckGraph = @import("../types_graph.zig");
    const Link = @import("../link/Link.zig");

    const src =
        \\enum SmallErr { bad }
        \\enum BigErr { small, other }
        \\impl BigErr has From[SmallErr] { fn from(s: SmallErr) -> BigErr { BigErr.small } }
        \\fn inner() -> Result[int, SmallErr] { return Result[int, SmallErr].err(SmallErr.bad) }
        \\fn outer() -> Result[int, BigErr] {
        \\  v := inner()?
        \\  return Result.ok(v)
        \\}
        \\fn same_err(r: Result[int, BigErr]) -> Result[int, BigErr] {
        \\  v := r?
        \\  return Result.ok(v)
        \\}
        \\
    ;
    const tokens = try Lexer.tokenize(gpa, src);
    defer gpa.free(tokens);
    const tree = try Parser.expectTree(gpa, tokens, src);
    defer {
        gpa.free(tree.nodes);
        gpa.free(tree.extra);
    }

    var g = try Graph.single(gpa, "main", "", src, tokens, tree.nodes, tree.extra, tree.pub_bits);
    defer g.deinit(gpa);
    var res = try ResolveGraph.resolveGraph(gpa, &g);
    defer res.deinit(gpa);
    var tc = try TypecheckGraph.checkGraph(gpa, &g, &res, null, 0);
    defer tc.deinit(gpa);
    try testing.expectEqual(@as(usize, 0), tc.diags.len);

    const names = try gpa.alloc(Link.SymName, res.fns.len);
    defer {
        for (names) |nm| gpa.free(nm.name);
        gpa.free(names);
    }
    for (res.fns, 0..) |gf, i| {
        const kind: Link.SymKind = if (gf.decl_node == Ast.none) .builtin else .user_fn;
        names[i] = .{ .kind = kind, .name = try gpa.dupe(u8, gf.name) };
    }

    const frozen = FakeFrozen{
        .tree = tree,
        .tokens = tokens,
        .source = src,
        .resolutions = res.resolutions[0],
        .node_types = tc.node_types[0],
        .layouts = tc.layouts,
        .enum_layouts = tc.enum_layouts,
        .names = names,
        .sigs = tc.sigs,
        .instances = tc.instances,
        .methods = tc.methods,
        .prelude_ids = tc.prelude_ids,
    };

    // Look fns up by NAME (the impl + inner shift source positions), not source order.
    var outer_decl: Ast.Index = Ast.none;
    var outer_sig: ?Sig = null;
    var same_decl: Ast.Index = Ast.none;
    var same_sig: ?Sig = null;
    var inner_sig: ?Sig = null;
    for (res.fns, 0..) |gf, i| {
        // Names are module-qualified (`main.outer`); match the `.<fn>` suffix.
        if (std.mem.endsWith(u8, gf.name, ".outer")) {
            outer_decl = gf.decl_node;
            outer_sig = tc.sigs[i];
        } else if (std.mem.endsWith(u8, gf.name, ".same_err")) {
            same_decl = gf.decl_node;
            same_sig = tc.sigs[i];
        } else if (std.mem.endsWith(u8, gf.name, ".inner")) {
            inner_sig = tc.sigs[i];
        }
    }
    try testing.expect(outer_decl != Ast.none and same_decl != Ast.none and inner_sig != null);

    // The witness the widen must fold, resolved the SAME way lower's buildResidual does:
    // BigErr's `From[SmallErr]::from`. The reified error types come from the `err` variant
    // (variants[1]) of each fn's reified return-Result enum — exactly what the fold reads.
    const rl = tc.enum_layouts[outer_sig.?.ret.enum_id];
    const ol = tc.enum_layouts[inner_sig.?.ret.enum_id];
    const ret_err = Typecheck.errPayload(rl);
    const op_err = Typecheck.errPayload(ol);
    const pick = Typecheck.resolveConformanceMethod(tc.methods, ret_err, "from", tc.prelude_ids.from, &.{op_err});
    try testing.expect(pick == .one);
    const want_name = if (pick.one.instance) |ii| tc.instances[ii].name.? else names[pick.one.fn_id].name;

    // outer's `?` WIDENS (SmallErr -> BigErr): its fold stream contains the `from` witness.
    var outer_sigs: std.ArrayList(Sig) = .empty;
    defer outer_sigs.deinit(gpa);
    var vo = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .fn_sig = outer_sig, .out = &outer_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, outer_decl, &vo);

    // same_err's `?` is the IDENTITY case (BigErr == BigErr): it folds NO `from` witness, so
    // its (b) callee-sig stream is byte-identical to a `?`-free fn (warm cache preserved).
    var same_sigs: std.ArrayList(Sig) = .empty;
    defer same_sigs.deinit(gpa);
    var vs = CallVisitor(FakeFrozen){ .gpa = gpa, .frozen = &frozen, .fn_sig = same_sig, .out = &same_sigs };
    try walk(.{ .tree = tree, .tokens = tokens, .source = src }, same_decl, &vs);

    var outer_has = false;
    for (outer_sigs.items) |s| {
        if (std.mem.eql(u8, s.name, want_name)) outer_has = true;
    }
    var same_has = false;
    for (same_sigs.items) |s| {
        if (std.mem.eql(u8, s.name, want_name)) same_has = true;
    }
    try testing.expect(outer_has);
    try testing.expect(!same_has);
}
