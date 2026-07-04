//! A flat, index-based AST.
//!
//! Nodes live in a single `[]Node` and reference their children by `u32` index
//! rather than by pointer. This is the same data-oriented layout Zig's own
//! compiler uses, and it buys us two things for free:
//!
//!   * **Cacheable.** A `[]Node` is a flat array of fixed-size values, so it
//!     serializes to (and loads from) the content cache exactly like `[]Token`
//!     — no pointer fix-ups, no bespoke (de)serializer.
//!   * **Parallel.** Each file parses into its own array on its own thread; the
//!     arrays never reference each other.
//!
//! Variable-arity children (a call's arguments, a block's statements, a
//! function's parameters) cannot fit in a `Node`'s two fixed slots, so they live
//! in a parallel `extra: []u32` side array — the same "extra_data" trick the Zig
//! compiler uses. A `Node` then stores an *index into `extra`* of a small header
//! describing a contiguous run of child node indices (see `Range`/`FnProto`).
//! Both arrays together form a `Tree`, and both serialize trivially.
//!
//! By construction the parser emits child nodes before their parents, and the
//! synthetic `program` node is appended last, so the **root is always the last
//! node** in the array (see `root`).

const std = @import("std");
const Token = @import("Token.zig").Token;

/// Index into the node array. `none` marks an absent child (e.g. a leaf's
/// operands, or a bare `return`'s missing expression).
///
/// A distinct `enum(u32)` newtype rather than a bare `u32`, so a slot that holds
/// a NODE index cannot be silently confused with a slot that holds a TOKEN index
/// (see `TokIndex`) or with the raw `u32` cells in the `extra` blob. The wire
/// layout is unchanged: `enum(u32){none=maxInt,_}` is layout-compatible with the
/// old `u32` (same size, same `none` sentinel bit-pattern), so cached blobs
/// round-trip identically. Convert only at true boundaries — `.int()` to index
/// the nodes array or write a cell, `Index.from()` to read one back.
pub const Index = enum(u32) {
    none = std.math.maxInt(u32),
    _,
    pub fn from(i: u32) Index {
        return @enumFromInt(i);
    }
    pub fn int(self: Index) u32 {
        return @intFromEnum(self);
    }
    pub fn unwrap(self: Index) ?Index {
        return if (self == .none) null else self;
    }
};
pub const none: Index = Index.none;

/// A TOKEN index stored in an lhs/rhs slot (as opposed to the usual NODE index).
/// A handful of tags overload a slot to name a token rather than a child node:
/// `break_stmt`/`continue_stmt`'s label, `import_decl`'s alias, and the segment
/// cells of an `import_decl`'s path range. Giving those a distinct newtype makes
/// a token-vs-node mixup a compile error. Same wire layout as `Index`/`u32`.
pub const TokIndex = enum(u32) {
    none = std.math.maxInt(u32),
    _,
    pub fn from(i: u32) TokIndex {
        return @enumFromInt(i);
    }
    pub fn int(self: TokIndex) u32 {
        return @intFromEnum(self);
    }
    pub fn unwrap(self: TokIndex) ?TokIndex {
        return if (self == .none) null else self;
    }
};

pub const Node = extern struct {
    tag: Tag,
    /// Index into the token array of the token that best represents this node:
    /// the operator for `unary`/`binary`, the name/literal token otherwise, the
    /// `(`/`{` for `call`/`block`, the function name for `fn_decl`.
    main_token: u32,
    /// Child node indices, OR an index into `extra` of a range/proto header.
    /// Meaning depends on `tag`; `none` when unused.
    lhs: Index,
    rhs: Index,

    comptime {
        // Pinned wire layout: `[]Node` is reinterpreted as raw cache bytes, so a
        // field change that shifts these offsets/size relocates padding and silently
        // invalidates old blobs — make it a deliberate (build-breaking) decision.
        if (@sizeOf(Node) != 16 or @offsetOf(Node, "main_token") != 4 or
            @offsetOf(Node, "lhs") != 8 or @offsetOf(Node, "rhs") != 12)
            @compileError("Node layout changed — bump the cache-blob format");
    }

    pub const Tag = enum(u8) {
        /// Integer literal. `main_token` is the number.
        literal_number,
        /// String literal. `main_token` is the string (quotes included).
        literal_string,
        /// `true` / `false`. `main_token` is the keyword.
        literal_bool,
        /// A name. `main_token` is the identifier.
        identifier,
        /// Prefix `op operand`. `main_token` is the operator; `lhs` is operand.
        unary,
        /// `lhs op rhs`. `main_token` is the operator.
        binary,

        // Appended below; the six above keep their ordinals (Tag is enum(u8) and
        // nodes are memcpy'd to/from the cache, so reordering breaks old blobs).

        /// Postfix `callee(args...)`. `main_token` is `(`. `lhs` is the callee
        /// expression node. `rhs` is the `extra` header of an args `Range`.
        call,
        /// `name := expr`. `main_token` is the name identifier. `lhs` is the
        /// initializer expression. `rhs` is `none`.
        var_decl,
        /// `name = expr`. `main_token` is the name identifier. `lhs` is the
        /// target (an `identifier` node). `rhs` is the value expression.
        assign,
        /// `return expr?`. `main_token` is `return`. `lhs` is the expression or
        /// `none` (bare return). `rhs` is `none`.
        return_stmt,
        /// An expression used as a statement. `main_token` is the expression's
        /// first token. `lhs` is the inner expression. `rhs` is `none`.
        expr_stmt,
        /// `{ stmts }`. `main_token` is `{`. `lhs` is the `extra` header of a
        /// statements `Range`. `rhs` is `none`.
        block,
        /// `name: Type`. `main_token` is the param name. `lhs` is the type-ref
        /// (an `identifier` node naming the type). `rhs` is `none`.
        param,
        /// `fn name(params) -> Ret { block }`. `main_token` is the name
        /// identifier. `lhs` is the `extra` header of a `FnProto`. `rhs` is the
        /// block node.
        fn_decl,
        /// The whole file: a sequence of function declarations. `main_token` is
        /// unused (0). `lhs` is the `extra` header of a fns `Range`. `rhs` is
        /// `none`.
        program,

        /// `while cond { body }`. `main_token` is `while`. `lhs` is the condition
        /// expression. `rhs` is the body `block` node.
        while_stmt,
        /// `if cond { then } [else (block|if)]`. `main_token` is `if`. `lhs` is
        /// the condition expression. `rhs` is the `extra` index of a 2-cell
        /// header `{then_block, else_node}`: `else_node` is a `block` node, a
        /// nested `if_stmt` node (for `else if`), or `Ast.none` (no else).
        if_stmt,

        /// The unit value `()` AND the unit type-ref `()`. A zero-sized leaf:
        /// `main_token` is the `(`; `lhs`/`rhs` are `none`. In value position it
        /// materializes to nothing; in type position it denotes the unit type.
        literal_unit,

        /// `loop { body }`. `main_token` is `loop`. `lhs` is the body `block`.
        /// `rhs` is `none`. A VALUE expression; its type is the merge of all
        /// `break <expr>` sites (or `never` when there is no value-break),
        /// modeled downstream in Typecheck.
        loop_expr,
        /// `for ident in lo..hi { body }`. `main_token` is the loop-var ident.
        /// `lhs` is the body `block`. `rhs` is the `extra` index of a 2-cell
        /// header `{lo, hi}`. Half-open `[lo, hi)`. A `()` statement.
        for_stmt,
        /// `break expr?`. `main_token` is `break`. `lhs` is the value expression
        /// or `none` (bare break). `rhs` overloads as a label TOKEN index: the
        /// identifier token after `@` for `break @name`, or `none`. This is
        /// the one node where `rhs` names a *token*, not a child node — break/
        /// continue have no node-child in `rhs`, so the slot is free.
        break_stmt,
        /// `continue`. `main_token` is `continue`. `lhs` is `none`. `rhs` overloads
        /// as a label TOKEN index (the identifier after `@`) for `continue @name`,
        /// or `none`. Like `break_stmt`, `rhs` names a *token*, not a node.
        continue_stmt,

        /// `@name <inner>`: a label prefixed onto a loop/while/for/bare-block.
        /// `main_token` is the identifier token after `@` (the label text, no
        /// `@`). `lhs` is the inner construct node (`loop_expr`, `while_stmt`,
        /// `for_stmt`, or `block`). `rhs` is `none`. A transparent wrapper for
        /// typing/codegen; its value (when the inner is value-yielding) is the
        /// inner's value merged with every `break @name <expr>` site.
        labeled,

        /// `struct Name { x: int, y: int }`. `main_token` is the struct name
        /// identifier. `lhs` is the `extra` header of a `Range` over `param`
        /// field nodes (in declaration order). `rhs` is `none`.
        struct_decl,
        /// `Name { x: 1, y: 2 }`. `main_token` is the `{`. `lhs` is the
        /// type-name `identifier` node. `rhs` is the `extra` header of a `Range`
        /// over `field_init` nodes.
        struct_init,
        /// A field initializer inside a `struct_init`: `name: value`, or the
        /// punning shorthand `name` (which synthesizes a real `identifier` leaf
        /// on the field token as `lhs`). `main_token` is the field-name ident.
        /// `lhs` is the value expression (never `none`). `rhs` is `none`.
        field_init,
        /// Field access `recv.field`. `main_token` is the field-name ident
        /// (after the `.`). `lhs` is the receiver expression (nests for
        /// `p.a.b`). `rhs` is `none`. Also the lhs of a field place-store
        /// (reusing `.assign`).
        field_access,

        /// `enum N { ... }`. `main_token` is the enum name. `lhs` is the `extra`
        /// header of a `Range` over variant nodes (in declaration order).
        /// `rhs` is `none`.
        enum_decl,
        /// A unit variant `Empty` in an enum decl. `main_token` is the variant
        /// name. `lhs`/`rhs` are `none`.
        enum_variant_unit,
        /// A tuple variant `Circle(int)` / `Line(int, int)`. `main_token` is the
        /// variant name. `lhs` is the `extra` header of a `Range` over type-ref
        /// nodes (positional payload types). `rhs` is `none`.
        enum_variant_tuple,
        /// A struct variant `Rect { w: int, h: int }`. `main_token` is the
        /// variant name. `lhs` is the `extra` header of a `Range` over `param`
        /// field nodes (REUSE). `rhs` is `none`.
        enum_variant_struct,
        /// Unit-variant construction `.Empty` or `N.Empty`. `main_token` is the
        /// variant-name ident. `lhs` is the type-name `identifier` node for a
        /// qualified `N.Empty`, or `Ast.none` for inferred `.Empty`. `rhs` none.
        enum_init_unit,
        /// Tuple-variant construction `.Circle(5)` / `N.Circle(5)`. `main_token`
        /// is the variant-name ident. `lhs` is the type-name ident OR `none`
        /// (inferred). `rhs` is the `extra` header of a `Range` over arg exprs.
        enum_init_tuple,
        /// Struct-variant construction `.Rect { w: 3, h: 4 }`. `main_token` is
        /// the variant-name ident. `lhs` is the type-name ident OR `none`. `rhs`
        /// is the `extra` header of a `Range` over `field_init` nodes (REUSE).
        enum_init_struct,
        /// `match scrut { arm, ... }`. `main_token` is the `match` keyword.
        /// `lhs` is the scrutinee expression. `rhs` is the `extra` header of a
        /// `Range` over `match_arm` nodes.
        match_expr,
        /// One match arm `pat [if guard] -> body`. `main_token` is the `->`.
        /// `lhs` is the pattern node. `rhs` is the `extra` header of a 2-cell
        /// `{guard, body}`: `guard` is the guard cond expr node or `none`
        /// (unguarded); `body` is the arm body expr (decode via `armHeaderAt`).
        match_arm,
        /// A variant pattern `.V` / `N.V` (optionally binding/matching payload).
        /// `main_token` is the variant-name ident. `lhs` is the type-name ident
        /// for a qualified `N.V`, or `none` for inferred `.V`. `rhs` is the
        /// `extra` header of a `Range` over arbitrary sub-pattern nodes (bindings,
        /// literals, wildcards, nested variants, or-patterns), or `none`.
        pattern_variant,
        /// The wildcard pattern `_`. `main_token` is the `_` ident. `lhs`/`rhs`
        /// are `none`.
        pattern_wildcard,
        /// A payload binding inside a `pattern_variant`. `main_token` is the
        /// bound local name. `lhs` is the source-field `identifier` node for a
        /// struct rename `field: alias` (then `main_token` is the alias), or
        /// `none` for a tuple-positional / struct-pun binding. `rhs` is an
        /// optional sub-pattern matched against the field/element (`.Rect { w: 0 }`
        /// or a nested pattern), or `none` to bind the whole value.
        pattern_binding,

        /// An int/bool literal pattern. `main_token` is the number/`true`/`false`
        /// token; matched against the scrutinee by equality. `lhs`/`rhs` are `none`.
        pattern_literal,
        /// An or-pattern `A | B | ...`. `main_token` is the first alt's first token.
        /// `lhs` is the `extra` header of a `Range` over >=2 alternative pattern
        /// nodes; `rhs` is `none`. All alts must bind the same names/types.
        pattern_or,

        /// `import a/b/c [as alias]`. `main_token` is the LAST path-segment
        /// identifier token (`c`) — the namespace this import binds by default.
        /// `lhs` is the `extra` header of a `Range` over the path-segment TOKEN
        /// indices (`a`,`b`,`c`, in order) — NOT node indices; the resolver
        /// rebuilds the `/`-joined module path and the qualified symbol prefix
        /// from them. `rhs` is the alias identifier TOKEN index for `as alias`,
        /// or `Ast.none` for no alias. Like `break_stmt`, `rhs` (and the `lhs`
        /// range contents) name *tokens*, not child nodes, so an `import_decl`
        /// has no node children and the children-before-parents invariant is
        /// vacuously satisfied.
        import_decl,

        /// A poison/error leaf the fault-tolerant parser emits at a parse error
        /// to keep building a partial tree instead of bailing. A leaf:
        /// `main_token` is the offending token; `lhs`/`rhs` are `Ast.none`. It is
        /// already-diagnosed by construction, so earlier stages treat it as an
        /// inert leaf that produces NO further diagnostics (types as the poison
        /// `Kind.invalid`); the tainted-tree gate keeps it out of lower, where
        /// `.error_node` is `unreachable`.
        error_node,

        // Generics front-end (M1). Appended at the END (frozen ordinals; `[]Node`
        // is memcpy'd to/from the content cache; `ParseHeader.version` bumped on
        // this change). Both PARSE into a well-formed tree but Typecheck rejects
        // them wholesale with T0013 before any lower/codegen — no semantics attach.

        /// A declared type parameter `T` in a `[T, U]` generic-param list.
        /// `main_token` is the param name identifier. `rhs` is `none`. `lhs` is
        /// `none` for an unbounded param, or (M13) a bound protocol-reference node
        /// (`identifier` / `field_access` dot-chain) for `[T has P]` / `[T has mod.P]`.
        generic_param,
        /// A type application `Base[Arg, ..]` — in TYPE position (`Box[int]`) or
        /// wrapping a callee in POSTFIX-CALL position (`id[int]` before `(..)`).
        /// `main_token` is the `[`. `lhs` is the base/callee node (an `identifier`
        /// or a `field_access` dot-chain). `rhs` is the `extra` header of a `Range`
        /// over the type-argument nodes.
        type_app,

        // Inherent methods (M8). Appended at the END (frozen ordinal; `[]Node` is
        // memcpy'd to/from the content cache; `ParseHeader.version` bumped 6->7 on
        // this change). An `impl` block desugars each method to an ordinary
        // `fn_decl` with a synthesized `self: <Receiver>` first param, so lower /
        // typecheck treat a method like any fn; only method-call DISPATCH is new.

        /// `impl Type { fn m(self, ..) -> R { .. } }` — an inherent-method block.
        /// `main_token` is the receiver type-name identifier token. `lhs` is the
        /// receiver type-ref `identifier` node. `rhs` is the `extra` header of a
        /// `Range` over the method `fn_decl` nodes (in declaration order).
        impl_decl,

        // Protocols + conformance (M11). Appended at the END (frozen ordinals;
        // `[]Node` is memcpy'd to/from the content cache; `ParseHeader.version`
        // bumped 8->9 on this change). A protocol declares signature-only methods;
        // an `impl T has P` block conforms a concrete type to a protocol. Each
        // conformance method desugars to an ordinary `fn_decl` exactly like an
        // inherent-method block, so lower/typecheck treat it like any method.

        /// `protocol P[X,..] { fn m(self, ..) -> R }` — signature-only method decls.
        /// `main_token` is the protocol-name identifier token. `lhs` is the `extra`
        /// header of a `Range` over the bodyless method-signature `fn_decl` nodes.
        /// `rhs` is the `extra` header of a `Range` over the protocol's generic-param
        /// `generic_param` nodes (M14, mirroring struct/enum templates), or `none` for
        /// a non-generic protocol (keeping every M11/M13 protocol byte-identical). The
        /// method sigs never enter the fn table (collectGlobals ignores `protocol_decl`),
        /// so their synthesized `self` type-ref is inert.
        protocol_decl,

        /// `impl T has P { fn m.. {} }` — a keyword-led conformance-impl block.
        /// `main_token` is the receiver type-name token (the LAST segment for a
        /// qualified `impl mod.T`). `lhs` is the receiver type-ref node (an
        /// `identifier`, or a `field_access` chain for `impl mod.T`). `rhs` is the
        /// `extra` index of a 3-cell header `{protocol_ref_node, methods_start,
        /// methods_len}`: the protocol reference is an `identifier`/`field_access`
        /// node, and the method `fn_decl`s live at `extra[start .. start + len]`.
        impl_has_decl,
    };
};

comptime {
    // The `extra` array is `[]u32`; pack/unpack memcpy these arrays as bytes, so
    // `Node` must be a whole number of 4-byte words and no more aligned.
    std.debug.assert(@sizeOf(Node) % 4 == 0);
    std.debug.assert(@alignOf(Node) <= 4);
}

/// A parse result: the node array plus its `extra` side array. Both are owned
/// together and (de)serialize together via `pack`/`unpack`.
///
/// `pub_bits` is a packed bitset (one bit per node index) marking which decl
/// nodes carry the `pub` modifier (export visibility). It is memcpy-trivial
/// like `nodes`/`extra` and round-trips through `pack`/`unpack`. It defaults to
/// the empty slice so the many partial `Tree` views built across the driver
/// (`.{ .nodes = ..., .extra = ... }`) keep compiling; `isPub` treats an absent
/// or short bitset as "not pub", so an empty bitset means no exports.
pub const Tree = struct {
    nodes: []Node,
    extra: []u32,
    pub_bits: []const u32 = &.{},

    /// Whether the decl node at `idx` carries `pub`. Out-of-range (or an empty
    /// bitset) reads as `false`.
    pub fn isPub(tree: Tree, idx: Index) bool {
        const i = idx.int();
        const word = i >> 5;
        if (word >= tree.pub_bits.len) return false;
        return (tree.pub_bits[word] >> @intCast(i & 31)) & 1 != 0;
    }
};

/// The number of `u32` words needed to hold one bit per node.
pub fn pubBitsLen(node_count: usize) usize {
    return (node_count + 31) >> 5;
}

/// A contiguous run of child node indices stored in `extra`, described by a
/// two-cell header `{start, len}`. The header cell index is what a `Node`
/// stores; the run lives at `extra[start .. start + len]`.
pub const Range = struct { start: u32, len: u32 };

/// A function's signature, decoded from a fixed 5-cell `FnProto` header in
/// `extra`: `{ret_type_node, params_start, params_len, generic_start, generic_len}`.
/// The layout is ADDITIVE: cells 0-2 are unchanged, so every existing decode site
/// that reads `.ret_type`/`.params` is byte-identical; cells 3-4 carry the
/// (usually empty) generic-param run appended in M1.
pub const FnProto = struct {
    /// type-ref node naming the return type, or `none` for unit `()`.
    ret_type: Index,
    /// `param` node indices, in source order.
    params: []const Index,
    /// `generic_param` node indices for `fn f[T, U](..)`, in source order; empty
    /// for a non-generic fn.
    generic_params: []const Index,
};

/// Decode the `{start, len}` range header at `header`.
pub fn rangeAt(tree: Tree, header: u32) Range {
    return .{ .start = tree.extra[header], .len = tree.extra[header + 1] };
}

/// The slice of child node indices described by the range header at `header`.
/// The `extra` cells are raw `u32`s that ARE node indices, so reinterpret the run
/// as `[]const Index` — layout-identical since `Index` is `enum(u32)`.
pub fn rangeSlice(tree: Tree, header: u32) []const Index {
    const r = rangeAt(tree, header);
    return @ptrCast(tree.extra[r.start .. r.start + r.len]);
}

/// Decode the 2-cell `if_stmt` header at `header`: `{then_block, else_node}`.
pub fn ifHeaderAt(tree: Tree, header: u32) struct { then_block: Index, else_node: Index } {
    return .{ .then_block = Index.from(tree.extra[header]), .else_node = Index.from(tree.extra[header + 1]) };
}

/// Decode the 2-cell `for_stmt` range header at `header`: `{lo, hi}`.
pub fn forHeaderAt(tree: Tree, header: u32) struct { lo: Index, hi: Index } {
    return .{ .lo = Index.from(tree.extra[header]), .hi = Index.from(tree.extra[header + 1]) };
}

/// Decode the 2-cell `match_arm` header at `header`: `{guard, body}`.
pub fn armHeaderAt(tree: Tree, header: u32) struct { guard: Index, body: Index } {
    return .{ .guard = Index.from(tree.extra[header]), .body = Index.from(tree.extra[header + 1]) };
}

/// Decode the 5-cell `FnProto` header at `header`. Cells 3-4 hold the generic-param
/// run; `gl == 0` yields an empty slice (`gs` may equal `extra.len`, so `extra[gs..gs]`
/// is a safe empty view).
pub fn protoAt(tree: Tree, header: u32) FnProto {
    const ps = tree.extra[header + 1];
    const pl = tree.extra[header + 2];
    const gs = tree.extra[header + 3];
    const gl = tree.extra[header + 4];
    return .{
        .ret_type = Index.from(tree.extra[header]),
        .params = @ptrCast(tree.extra[ps .. ps + pl]),
        .generic_params = @ptrCast(tree.extra[gs .. gs + gl]),
    };
}

/// True when the `param` node at `param_idx` is a `mut`-qualified receiver: the
/// token immediately before its name is `kw_mut` (M9). Token-adjacency detection
/// mirrors the by-text `self`/`_` recognition — no dedicated node/cell, so no
/// `ParseHeader.version` bump. `tokens` is passed explicitly because a `Tree` holds
/// only nodes/extra, never the token stream.
pub fn isMutParam(tree: Tree, tokens: []const Token, param_idx: Index) bool {
    const n = tree.nodes[param_idx.int()];
    return n.tag == .param and n.main_token > 0 and tokens[n.main_token - 1].tag == .kw_mut;
}

/// The root (top-level) node of a non-empty tree — by construction the last.
pub fn root(nodes: []const Node) Index {
    return Index.from(@intCast(nodes.len - 1));
}

// A few tags store a TOKEN index in an lhs/rhs slot instead of a child node
// index (see the `break_stmt`/`continue_stmt`/`import_decl` doc comments). The
// slot's storage is still an `Index` field, but these accessors reinterpret it
// as a `TokIndex` so downstream code that means "token" cannot accidentally feed
// the value back into a node-index API (and vice versa).

/// The `@label` token of a `break_stmt`/`continue_stmt` (in the `rhs` slot), or
/// `TokIndex.none` when unlabeled.
pub fn labelTok(node: Node) TokIndex {
    return TokIndex.from(node.rhs.int());
}

/// The `as alias` token of an `import_decl` (in the `rhs` slot), or
/// `TokIndex.none` when the import has no alias.
pub fn importAliasTok(node: Node) TokIndex {
    return TokIndex.from(node.rhs.int());
}

/// The `/`-separated path-segment TOKEN indices of an `import_decl`, decoded from
/// the range header in its `lhs` slot. These are token indices, NOT node indices.
pub fn importPathToks(tree: Tree, node: Node) []const TokIndex {
    const r = rangeAt(tree, node.lhs.int());
    return @ptrCast(tree.extra[r.start .. r.start + r.len]);
}

/// Decode the 3-cell `impl_has_decl` header at `header` (its `rhs` slot):
/// `{protocol_ref_node, methods_start, methods_len}`. `protocol` is the
/// (bare or qualified) protocol-reference node; `methods` is the run of method
/// `fn_decl` node indices at `extra[start .. start + len]`.
pub fn implHasAt(tree: Tree, header: u32) struct { protocol: Index, methods: []const Index } {
    const ms = tree.extra[header + 1];
    const ml = tree.extra[header + 2];
    return .{
        .protocol = Index.from(tree.extra[header]),
        .methods = @ptrCast(tree.extra[ms .. ms + ml]),
    };
}

/// The method `fn_decl` node indices of an inherent (`impl_decl`) OR conformance
/// (`impl_has_decl`) block — one accessor so callers don't branch on the two
/// impl shapes (they store their method run differently: a plain `Range` in
/// `impl_decl.rhs`, a 3-cell header in `impl_has_decl.rhs`). Empty for any other tag.
pub fn implMethods(tree: Tree, node: Node) []const Index {
    return switch (node.tag) {
        .impl_decl => rangeSlice(tree, node.rhs.int()),
        .impl_has_decl => implHasAt(tree, node.rhs.int()).methods,
        else => &.{},
    };
}

/// The bound protocol-reference node of a `generic_param` (`[T has P]`, M13), stored
/// in its `lhs` slot — an `identifier` (bare `P`) or a `field_access` chain (`mod.P`).
/// Null for an unbounded param (`lhs == none`) or a non-`generic_param` node.
pub fn genericParamBound(tree: Tree, node: Index) ?Index {
    const n = tree.nodes[node.int()];
    if (n.tag != .generic_param) return null;
    return n.lhs.unwrap();
}

/// The protocol-reference node an `impl_has_decl` conforms to, or `null` for a
/// non-conformance (`impl_decl`) or any other tag.
pub fn implProtocol(tree: Tree, node: Node) ?Index {
    return switch (node.tag) {
        .impl_has_decl => implHasAt(tree, node.rhs.int()).protocol,
        else => null,
    };
}

/// The generic-parameter `generic_param` node indices of a `protocol_decl` (`protocol
/// Into[U]`, M14), stored as a `Range` in its `rhs` slot — empty for a non-generic
/// protocol (`rhs == none`) or a non-`protocol_decl` node. Mirrors the struct/enum
/// template read (their generic params ride the `rhs` `Range` too).
pub fn protocolGenericParams(tree: Tree, node: Index) []const Index {
    const n = tree.nodes[node.int()];
    if (n.tag != .protocol_decl or n.rhs == none) return &.{};
    return rangeSlice(tree, n.rhs.int());
}

/// Unwrap a protocol-reference node (`P` / `mod.P` / `P[int]` / `mod.P[int]`, M14) to
/// its BASE name node — the `identifier`/`field_access` a bare/qualified ref already
/// is, or the `type_app`'s `lhs` for a generic protocol-ref. `protocolIdFromNode`
/// resolves the base; `protocolRefArgs` reads the type-args. A bare ref is its own base.
pub fn protocolRefBase(tree: Tree, node: Index) Index {
    const n = tree.nodes[node.int()];
    return if (n.tag == .type_app) n.lhs else node;
}

/// The type-argument node indices of a protocol-reference (`P[int]` -> `[int]`, M14),
/// or empty for a bare/qualified ref (no `[..]`). The args are a `Range` in the
/// `type_app`'s `rhs`; each element is an ordinary type-ref node resolved by
/// `typeFromNode`.
pub fn protocolRefArgs(tree: Tree, node: Index) []const Index {
    const n = tree.nodes[node.int()];
    if (n.tag != .type_app) return &.{};
    return rangeSlice(tree, n.rhs.int());
}

/// "TOYP" — a magic so a foreign/corrupt blob is treated as a cache miss.
pub const parse_magic: u32 = 0x544f5950;

/// Header prefixing a packed `Tree` blob. `extern` so it serializes by memcpy.
pub const ParseHeader = extern struct {
    magic: u32,
    /// Bumped to 4 to add the trailing `pub_bits` section; older v3 blobs
    /// (no `pub_bits`) miss cleanly via the version check in `unpack`. Bumped to 5
    /// when the `error_node` Tag ordinal was appended, so a blob produced by an
    /// older compiler is rejected rather than reused across the Tag change. Bumped
    /// to 6 for the M1 generics front-end: the `FnProto` header grew 3->5 cells and
    /// the `generic_param`/`type_app` Tag ordinals were appended, so a v5 3-cell
    /// proto read by the 5-cell `protoAt` would alias neighbouring `extra` bytes —
    /// a v5 blob must miss cleanly. Bumped to 7 for the M8 `impl_decl` Tag ordinal
    /// appended at the end: a v6 blob predating that tag must miss cleanly rather
    /// than misdecode a cell whose meaning the new tag changed. Bumped to 8 for the
    /// M10 generic-impl parse change: an `impl_decl`'s `lhs` may now be a `type_app`
    /// (`impl Box[T]`) and a method carries an impl-derived `generic_param` run in its
    /// FnProto (cells 3-4), so a v7 blob — which never produced either shape — must
    /// miss cleanly rather than feed a stale AST into the M10 method-monomorphizer.
    /// Bumped to 9 for the M11 protocols front-end: the `protocol_decl` and
    /// `impl_has_decl` Tag ordinals were appended, and `impl_has_decl` stores a new
    /// 3-cell header in its `rhs`, so a v8 blob predating these tags must miss cleanly
    /// rather than misdecode a node whose tag/cell meaning the new tags changed.
    /// Bumped to 10 for the M13 constrained-generics parse change: a `generic_param`'s
    /// `lhs` may now carry a bound protocol-ref node (`[T has P]`), where a v9 blob
    /// always left it `none` — a v9 blob must miss cleanly so a stale parse never feeds
    /// an unbounded generic-param shape into the M13 bound-resolution machinery.
    /// Bumped to 11 for the M14 generic-protocols parse change: `protocol_decl`'s `rhs`
    /// may now carry a generic-param Range (`protocol Into[U]`), and a `type_app` may now
    /// appear in an `impl .. has P[int]` protocol slot and a `[T has P[int]]` generic-param
    /// bound — where a v10 blob always left `protocol_decl.rhs` `none` and never wrapped a
    /// protocol-ref in a `type_app`. A v10 blob must miss cleanly so a stale parse never
    /// feeds a bare protocol-ref shape into the M14 protocol-args machinery.
    version: u32 = 11,
    node_count: u32,
    extra_count: u32,
    /// Number of `u32` words in the `pub_bits` section (`pubBitsLen(node_count)`).
    pub_words: u32,
};

/// A padding-free structural fingerprint of a parsed `Tree`. `Node` is an
/// `extern struct` with 3 uninitialized padding bytes between `tag` and
/// `main_token`; hashing the packed blob (or `sliceAsBytes(nodes)`) folds that
/// garbage in, so the fp differs run-to-run on a COLD build. The content-fp cache
/// keys codegen entries off this fp, so a non-deterministic parse fp is a permanent
/// cache miss (or, worse, a cross-build alias). This folds ONLY the
/// semantically-meaningful fields (every field that round-trips through
/// pack/unpack), keeping the canonical field list co-located with pack/unpack.
pub fn contentFp(tree: Tree) u64 {
    var h = std.hash.Wyhash.init(0x44_41_47_46); // "DAGF"
    for (tree.nodes) |n| {
        h.update(&[_]u8{@intFromEnum(n.tag)});
        h.update(std.mem.asBytes(&n.main_token));
        h.update(std.mem.asBytes(&n.lhs));
        h.update(std.mem.asBytes(&n.rhs));
    }
    h.update(std.mem.sliceAsBytes(tree.extra));
    h.update(std.mem.sliceAsBytes(tree.pub_bits));
    return h.final();
}

/// Pack a `Tree` into one flat byte blob: header, then nodes, then extra, then
/// the `pub_bits` bitset.
pub fn pack(gpa: std.mem.Allocator, tree: Tree) ![]u8 {
    const total = @sizeOf(ParseHeader) +
        tree.nodes.len * @sizeOf(Node) +
        tree.extra.len * 4 +
        tree.pub_bits.len * 4;
    const buf = try gpa.alloc(u8, total);
    const hdr = ParseHeader{
        .magic = parse_magic,
        .node_count = @intCast(tree.nodes.len),
        .extra_count = @intCast(tree.extra.len),
        .pub_words = @intCast(tree.pub_bits.len),
    };
    @memcpy(buf[0..@sizeOf(ParseHeader)], std.mem.asBytes(&hdr));
    var off: usize = @sizeOf(ParseHeader);
    const nb = std.mem.sliceAsBytes(tree.nodes);
    @memcpy(buf[off .. off + nb.len], nb);
    off += nb.len;
    const eb = std.mem.sliceAsBytes(tree.extra);
    @memcpy(buf[off .. off + eb.len], eb);
    off += eb.len;
    const pb = std.mem.sliceAsBytes(tree.pub_bits);
    @memcpy(buf[off .. off + pb.len], pb);
    return buf;
}

/// Unpack a blob produced by `pack`. Returns null on any mismatch (corrupt or
/// foreign blob) so the caller treats it as a cache miss. Caller owns the Tree.
pub fn unpack(gpa: std.mem.Allocator, bytes: []const u8) !?Tree {
    if (bytes.len < @sizeOf(ParseHeader)) return null;
    var hdr: ParseHeader = undefined;
    @memcpy(std.mem.asBytes(&hdr), bytes[0..@sizeOf(ParseHeader)]);
    if (hdr.magic != parse_magic or hdr.version != 11) return null;
    const need = @sizeOf(ParseHeader) +
        @as(usize, hdr.node_count) * @sizeOf(Node) +
        @as(usize, hdr.extra_count) * 4 +
        @as(usize, hdr.pub_words) * 4;
    if (bytes.len != need) return null;

    const nodes = try gpa.alloc(Node, hdr.node_count);
    errdefer gpa.free(nodes);
    const extra = try gpa.alloc(u32, hdr.extra_count);
    errdefer gpa.free(extra);
    const pub_bits = try gpa.alloc(u32, hdr.pub_words);

    var off: usize = @sizeOf(ParseHeader);
    @memcpy(std.mem.sliceAsBytes(nodes), bytes[off .. off + hdr.node_count * @sizeOf(Node)]);
    off += hdr.node_count * @sizeOf(Node);
    @memcpy(std.mem.sliceAsBytes(extra), bytes[off .. off + hdr.extra_count * 4]);
    off += hdr.extra_count * 4;
    @memcpy(std.mem.sliceAsBytes(pub_bits), bytes[off .. off + hdr.pub_words * 4]);
    return Tree{ .nodes = nodes, .extra = extra, .pub_bits = pub_bits };
}

/// Write the whole program (rooted at the last node) as an S-expression.
pub fn render(out: *std.Io.Writer, tree: Tree, tokens: []const Token, source: []const u8) !void {
    if (tree.nodes.len == 0) return;
    try renderNode(out, tree, tokens, source, root(tree.nodes));
}

fn renderNode(out: *std.Io.Writer, tree: Tree, tokens: []const Token, source: []const u8, idx: Index) !void {
    const nodes = tree.nodes;
    const n = nodes[idx.int()];
    const tok_text = tokens[n.main_token].text(source);
    switch (n.tag) {
        .literal_number, .literal_string, .literal_bool, .identifier => try out.writeAll(tok_text),
        .literal_unit => try out.writeAll("()"),
        // A poison leaf renders as a fixed `(error)` marker (its `main_token` is
        // the offending token, but the marker deliberately elides its text).
        .error_node => try out.writeAll("(error)"),
        // A declared type parameter renders as its bare name (like an identifier).
        .generic_param => try out.writeAll(tok_text),
        .type_app => {
            try out.writeAll("(tyapp ");
            try renderNode(out, tree, tokens, source, n.lhs);
            for (rangeSlice(tree, n.rhs.int())) |arg| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, arg);
            }
            try out.writeByte(')');
        },
        .impl_decl => {
            try out.print("(impl {s}", .{tok_text});
            for (rangeSlice(tree, n.rhs.int())) |method| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, method);
            }
            try out.writeByte(')');
        },
        .protocol_decl => {
            try out.print("(protocol {s}", .{tok_text});
            // Generic params (M14) ride the `rhs` Range, rendered only when present so
            // non-generic protocol goldens stay byte-identical (mirrors `struct_decl`).
            if (n.rhs != none) {
                try out.writeAll(" [");
                for (rangeSlice(tree, n.rhs.int()), 0..) |gp, i| {
                    if (i != 0) try out.writeByte(' ');
                    try renderNode(out, tree, tokens, source, gp);
                }
                try out.writeByte(']');
            }
            for (rangeSlice(tree, n.lhs.int())) |method| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, method);
            }
            try out.writeByte(')');
        },
        .impl_has_decl => {
            const h = implHasAt(tree, n.rhs.int());
            try out.writeAll("(impl-has ");
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, h.protocol);
            for (h.methods) |method| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, method);
            }
            try out.writeByte(')');
        },
        .unary => {
            try out.print("({s} ", .{tok_text});
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(')');
        },
        .binary => {
            try out.print("({s} ", .{tok_text});
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, n.rhs);
            try out.writeByte(')');
        },
        .call => {
            try out.writeAll("(call ");
            try renderNode(out, tree, tokens, source, n.lhs);
            for (rangeSlice(tree, n.rhs.int())) |arg| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, arg);
            }
            try out.writeByte(')');
        },
        .var_decl => {
            try out.print("(:= {s} ", .{tok_text});
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(')');
        },
        .assign => {
            try out.writeAll("(= ");
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, n.rhs);
            try out.writeByte(')');
        },
        .return_stmt => {
            if (n.lhs == none) {
                try out.writeAll("(return)");
            } else {
                try out.writeAll("(return ");
                try renderNode(out, tree, tokens, source, n.lhs);
                try out.writeByte(')');
            }
        },
        // An expression statement is a transparent wrapper: render the inner expr.
        .expr_stmt => try renderNode(out, tree, tokens, source, n.lhs),
        .block => {
            try out.writeAll("(block");
            for (rangeSlice(tree, n.lhs.int())) |stmt| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, stmt);
            }
            try out.writeByte(')');
        },
        .param => {
            try out.print("(param {s} ", .{tok_text});
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(')');
        },
        .fn_decl => {
            const proto = protoAt(tree, n.lhs.int());
            try out.print("(fn {s}", .{tok_text});
            // Generics render only when present, so existing goldens are unchanged.
            if (proto.generic_params.len > 0) {
                try out.writeAll(" [");
                for (proto.generic_params, 0..) |gp, i| {
                    if (i != 0) try out.writeByte(' ');
                    try renderNode(out, tree, tokens, source, gp);
                }
                try out.writeByte(']');
            }
            try out.writeAll(" (");
            for (proto.params, 0..) |pidx, i| {
                if (i != 0) try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, pidx);
            }
            try out.writeAll(") ");
            if (proto.ret_type == none) {
                try out.writeByte('_');
            } else {
                try renderNode(out, tree, tokens, source, proto.ret_type);
            }
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, n.rhs);
            try out.writeByte(')');
        },
        .program => {
            try out.writeAll("(program");
            for (rangeSlice(tree, n.lhs.int())) |decl_idx| {
                try out.writeByte(' ');
                // A `pub` decl renders as `(pub <decl>)` so the visibility surface
                // is visible in the S-expression (and asserted by parse tests).
                if (tree.isPub(decl_idx)) {
                    try out.writeAll("(pub ");
                    try renderNode(out, tree, tokens, source, decl_idx);
                    try out.writeByte(')');
                } else {
                    try renderNode(out, tree, tokens, source, decl_idx);
                }
            }
            try out.writeByte(')');
        },
        .import_decl => {
            try out.writeAll("(import");
            // Path segments are TOKEN indices in the range; render them `/`-joined.
            for (importPathToks(tree, n), 0..) |seg_tok, i| {
                try out.writeByte(if (i == 0) ' ' else '/');
                try out.writeAll(tokens[seg_tok.int()].text(source));
            }
            if (importAliasTok(n).unwrap()) |alias| {
                try out.print(" as {s}", .{tokens[alias.int()].text(source)});
            }
            try out.writeByte(')');
        },
        .while_stmt => {
            try out.writeAll("(while ");
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, n.rhs);
            try out.writeByte(')');
        },
        .if_stmt => {
            const h = ifHeaderAt(tree, n.rhs.int());
            try out.writeAll("(if ");
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, h.then_block);
            if (h.else_node != none) {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, h.else_node);
            }
            try out.writeByte(')');
        },
        .loop_expr => {
            try out.writeAll("(loop ");
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(')');
        },
        .for_stmt => {
            const h = forHeaderAt(tree, n.rhs.int());
            try out.print("(for {s} ", .{tok_text});
            try renderNode(out, tree, tokens, source, h.lo);
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, h.hi);
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(')');
        },
        .break_stmt => {
            try out.writeAll("(break");
            if (labelTok(n).unwrap()) |label| {
                try out.print(" @{s}", .{tokens[label.int()].text(source)});
            }
            if (n.lhs != none) {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, n.lhs);
            }
            try out.writeByte(')');
        },
        .continue_stmt => {
            try out.writeAll("(continue");
            if (labelTok(n).unwrap()) |label| {
                try out.print(" @{s}", .{tokens[label.int()].text(source)});
            }
            try out.writeByte(')');
        },
        .labeled => {
            try out.print("(label {s} ", .{tok_text});
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(')');
        },
        .struct_decl => {
            try out.print("(struct {s}", .{tok_text});
            // Generics live in the `rhs` slot as a Range header, rendered only when
            // present (`rhs != none`) so existing non-generic goldens are unchanged.
            if (n.rhs != none) {
                try out.writeAll(" [");
                for (rangeSlice(tree, n.rhs.int()), 0..) |gp, i| {
                    if (i != 0) try out.writeByte(' ');
                    try renderNode(out, tree, tokens, source, gp);
                }
                try out.writeByte(']');
            }
            for (rangeSlice(tree, n.lhs.int())) |field| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, field);
            }
            try out.writeByte(')');
        },
        .struct_init => {
            try out.writeAll("(new ");
            try renderNode(out, tree, tokens, source, n.lhs);
            for (rangeSlice(tree, n.rhs.int())) |fi| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, fi);
            }
            try out.writeByte(')');
        },
        .field_init => {
            try out.print("(field {s} ", .{tok_text});
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.writeByte(')');
        },
        .field_access => {
            try out.writeAll("(. ");
            try renderNode(out, tree, tokens, source, n.lhs);
            try out.print(" {s})", .{tok_text});
        },
        .enum_decl => {
            try out.print("(enum {s}", .{tok_text});
            // Generics live in the `rhs` slot as a Range header (see struct_decl).
            if (n.rhs != none) {
                try out.writeAll(" [");
                for (rangeSlice(tree, n.rhs.int()), 0..) |gp, i| {
                    if (i != 0) try out.writeByte(' ');
                    try renderNode(out, tree, tokens, source, gp);
                }
                try out.writeByte(']');
            }
            for (rangeSlice(tree, n.lhs.int())) |v| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, v);
            }
            try out.writeByte(')');
        },
        .enum_variant_unit => try out.print("(variant.unit {s})", .{tok_text}),
        .enum_variant_tuple => {
            try out.print("(variant.tuple {s}", .{tok_text});
            for (rangeSlice(tree, n.lhs.int())) |ty| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, ty);
            }
            try out.writeByte(')');
        },
        .enum_variant_struct => {
            try out.print("(variant.struct {s}", .{tok_text});
            for (rangeSlice(tree, n.lhs.int())) |f| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, f);
            }
            try out.writeByte(')');
        },
        .enum_init_unit => {
            try out.writeAll("(enew.unit");
            if (n.lhs != none) {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, n.lhs);
            }
            try out.print(" {s})", .{tok_text});
        },
        .enum_init_tuple => {
            try out.writeAll("(enew.tuple");
            if (n.lhs != none) {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, n.lhs);
            }
            try out.print(" {s}", .{tok_text});
            for (rangeSlice(tree, n.rhs.int())) |a| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, a);
            }
            try out.writeByte(')');
        },
        .enum_init_struct => {
            try out.writeAll("(enew.struct");
            if (n.lhs != none) {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, n.lhs);
            }
            try out.print(" {s}", .{tok_text});
            for (rangeSlice(tree, n.rhs.int())) |fi| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, fi);
            }
            try out.writeByte(')');
        },
        .match_expr => {
            try out.writeAll("(match ");
            try renderNode(out, tree, tokens, source, n.lhs);
            for (rangeSlice(tree, n.rhs.int())) |arm| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, arm);
            }
            try out.writeByte(')');
        },
        .match_arm => {
            const h = armHeaderAt(tree, n.rhs.int());
            try out.writeAll("(arm ");
            try renderNode(out, tree, tokens, source, n.lhs);
            if (h.guard != none) {
                try out.writeAll(" if ");
                try renderNode(out, tree, tokens, source, h.guard);
            }
            try out.writeByte(' ');
            try renderNode(out, tree, tokens, source, h.body);
            try out.writeByte(')');
        },
        .pattern_variant => {
            try out.writeAll("(pvar");
            if (n.lhs != none) {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, n.lhs);
            }
            try out.print(" {s}", .{tok_text});
            if (n.rhs != none) {
                for (rangeSlice(tree, n.rhs.int())) |b| {
                    try out.writeByte(' ');
                    try renderNode(out, tree, tokens, source, b);
                }
            }
            try out.writeByte(')');
        },
        .pattern_wildcard => try out.writeAll("(_)"),
        .pattern_binding => {
            try out.print("(bind {s}", .{tok_text});
            if (n.lhs != none) {
                try out.writeAll(" from ");
                try renderNode(out, tree, tokens, source, n.lhs);
            }
            if (n.rhs != none) {
                try out.writeAll(" = ");
                try renderNode(out, tree, tokens, source, n.rhs);
            }
            try out.writeByte(')');
        },
        .pattern_literal => try out.print("(lit {s})", .{tok_text}),
        .pattern_or => {
            try out.writeAll("(por");
            for (rangeSlice(tree, n.lhs.int())) |a| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, a);
            }
            try out.writeByte(')');
        },
    }
}

const testing = std.testing;

test "rangeSlice and protoAt accessors round-trip on a hand-built tree" {
    // Build extra for: a NON-generic fn with two params (param nodes 0,1),
    // ret_type node 2. The 5-cell FnProto's generic run is empty (glen=0), with
    // gstart pointing at the current extra length (a safe empty view).
    // extra layout:
    //   [0]=0,[1]=1              params run (param node indices 0,1)
    //   [2]=0,[3]=2              Range header {start=0, len=2}
    //   [4]=2,[5]=0,[6]=2,       FnProto ret_type=2, params_start=0, params_len=2,
    //   [7]=4,[8]=0                       generic_start=4 (empty), generic_len=0
    var extra = [_]u32{ 0, 1, 0, 2, 2, 0, 2, 4, 0 };
    var nodes = [_]Node{
        .{ .tag = .param, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .param, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .identifier, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .block, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .fn_decl, .main_token = 0, .lhs = Index.from(4), .rhs = Index.from(3) },
    };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };

    const params = rangeSlice(tree, 2);
    try testing.expectEqual(@as(usize, 2), params.len);
    try testing.expectEqual(Index.from(0), params[0]);
    try testing.expectEqual(Index.from(1), params[1]);

    const proto = protoAt(tree, 4);
    try testing.expectEqual(Index.from(2), proto.ret_type);
    try testing.expectEqual(@as(usize, 2), proto.params.len);
    try testing.expectEqual(Index.from(1), proto.params[1]);
    // The additive cells decode to an empty generic-param slice for a plain fn.
    try testing.expectEqual(@as(usize, 0), proto.generic_params.len);
}

test "isMutParam detects a kw_mut token immediately before the param name" {
    // tokens: [0]=kw_mut [1]=identifier(self) [2]=identifier(other)
    const toks = [_]Token{
        .{ .tag = .kw_mut, .start = 0, .end = 3 },
        .{ .tag = .identifier, .start = 4, .end = 8 },
        .{ .tag = .identifier, .start = 9, .end = 14 },
    };
    var nodes = [_]Node{
        .{ .tag = .param, .main_token = 1, .lhs = none, .rhs = none }, // `mut self`
        .{ .tag = .param, .main_token = 2, .lhs = none, .rhs = none }, // plain param
    };
    const tree = Tree{ .nodes = &nodes, .extra = &.{} };
    try testing.expect(isMutParam(tree, &toks, Index.from(0)));
    try testing.expect(!isMutParam(tree, &toks, Index.from(1)));
}

test "protoAt decodes a non-empty generic-param run" {
    // A generic fn `f[T]()`: one generic_param node (index 0), no params, no
    // ret_type. The generic run lives at extra[0..1]; the 5-cell header follows.
    // extra layout:
    //   [0]=0                    generic run (generic_param node index 0)
    //   [1]=maxInt(ret none),    FnProto ret_type=none, params_start=1, params_len=0,
    //   [2]=1,[3]=0,                      generic_start=0, generic_len=1
    //   [4]=0,[5]=1
    var extra = [_]u32{ 0, std.math.maxInt(u32), 1, 0, 0, 1 };
    var nodes = [_]Node{
        .{ .tag = .generic_param, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .block, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .fn_decl, .main_token = 0, .lhs = Index.from(1), .rhs = Index.from(1) },
    };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };

    const proto = protoAt(tree, 1);
    try testing.expectEqual(none, proto.ret_type);
    try testing.expectEqual(@as(usize, 0), proto.params.len);
    try testing.expectEqual(@as(usize, 1), proto.generic_params.len);
    try testing.expectEqual(Index.from(0), proto.generic_params[0]);
}

test "pack/unpack byte round-trip" {
    const gpa = testing.allocator;
    var nodes = [_]Node{
        .{ .tag = .literal_number, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(0), .rhs = none },
    };
    var extra = [_]u32{ 0, 1, 0 };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };

    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer gpa.free(got.nodes);
    defer gpa.free(got.extra);

    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

test "renders the unit literal" {
    var nodes = [_]Node{
        .{ .tag = .literal_unit, .main_token = 0, .lhs = none, .rhs = none },
    };
    var extra = [_]u32{};
    const tree = Tree{ .nodes = &nodes, .extra = &extra };
    var buf: [16]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try renderNode(&w, tree, &.{Token{ .tag = .l_paren, .start = 0, .end = 1 }}, "()", Index.from(0));
    try testing.expectEqualStrings("()", w.buffered());
}

test "unpack rejects a foreign blob" {
    const gpa = testing.allocator;
    try testing.expect((try unpack(gpa, "not a tree")) == null);
    try testing.expect((try unpack(gpa, &.{})) == null);
}

test "pack/unpack round-trips a tree containing an impl_decl (v7)" {
    const gpa = testing.allocator;
    // pack/unpack is a pure byte round-trip (memcpy of the node/extra arrays), so
    // the tree need not be well-formed — it only has to contain the new tag so the
    // v7 blob exercises `impl_decl`'s ordinal.
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .impl_decl, .main_token = 0, .lhs = Index.from(0), .rhs = Index.from(1) },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(1), .rhs = none },
    };
    var extra = [_]u32{ 0, 1, 1 }; // an arbitrary Range header {start=0,len=1}
    const tree = Tree{ .nodes = &nodes, .extra = &extra };

    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer gpa.free(got.nodes);
    defer gpa.free(got.extra);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

test "unpack rejects a v7 blob (pre-generic-impl)" {
    const gpa = testing.allocator;
    var nodes = [_]Node{
        .{ .tag = .literal_number, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(0), .rhs = none },
    };
    var extra = [_]u32{ 0, 1, 0 };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };
    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    // Rewrite the header `version` field (the second u32) to 7: a blob from a
    // compiler predating the M10 generic-impl parse change must miss cleanly, not
    // misdecode an `impl`'s receiver / a method's generic run.
    std.mem.writeInt(u32, blob[4..8], 7, @import("builtin").cpu.arch.endian());
    try testing.expect((try unpack(gpa, blob)) == null);
}

test "pack/unpack round-trips a tree containing an impl_has_decl (v9)" {
    const gpa = testing.allocator;
    // A pure byte round-trip (memcpy of the node/extra arrays), so the tree need not
    // be well-formed — it only has to contain the new tag + its 3-cell header so the
    // v9 blob exercises `impl_has_decl`'s ordinal.
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 0, .lhs = none, .rhs = none }, // recv type-ref
        .{ .tag = .identifier, .main_token = 1, .lhs = none, .rhs = none }, // protocol ref
        .{ .tag = .impl_has_decl, .main_token = 0, .lhs = Index.from(0), .rhs = Index.from(2) },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(3), .rhs = none },
    };
    // extra: [0..1]=methods run (empty here, start=0,len=0 lands at header cell 2),
    // header at cell 2 = {protocol_ref=1, methods_start=0, methods_len=0};
    // program's Range header {start=5,len=1} over decl node 2.
    var extra = [_]u32{ 0, 0, 1, 0, 0, 2, 1 };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };

    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer gpa.free(got.nodes);
    defer gpa.free(got.extra);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqualSlices(u32, tree.extra, got.extra);
}

test "unpack rejects a v8 blob (pre-protocols)" {
    const gpa = testing.allocator;
    var nodes = [_]Node{
        .{ .tag = .literal_number, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(0), .rhs = none },
    };
    var extra = [_]u32{ 0, 1, 0 };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };
    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    // Rewrite `version` to 8: a blob from a compiler predating the M11 protocol tags
    // must miss cleanly, not misdecode an `impl_has_decl`'s 3-cell header.
    std.mem.writeInt(u32, blob[4..8], 8, @import("builtin").cpu.arch.endian());
    try testing.expect((try unpack(gpa, blob)) == null);
}

test "pack/unpack round-trips a tree with a bound generic_param (v10)" {
    const gpa = testing.allocator;
    // A pure byte round-trip — the tree only has to contain a `generic_param` whose
    // `lhs` names a bound protocol-ref node so the v10 shape is exercised.
    var nodes = [_]Node{
        .{ .tag = .identifier, .main_token = 1, .lhs = none, .rhs = none }, // bound protocol ref `P`
        .{ .tag = .generic_param, .main_token = 0, .lhs = Index.from(0), .rhs = none }, // `T has P`
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(2), .rhs = none },
    };
    var extra = [_]u32{ 1, 1 }; // program's Range header {start=... } — arbitrary
    const tree = Tree{ .nodes = &nodes, .extra = &extra };

    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer gpa.free(got.nodes);
    defer gpa.free(got.extra);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqual(@as(?Index, Index.from(0)), genericParamBound(got, Index.from(1)));
}

test "unpack rejects a v9 blob (pre-generic-bounds)" {
    const gpa = testing.allocator;
    var nodes = [_]Node{
        .{ .tag = .literal_number, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(0), .rhs = none },
    };
    var extra = [_]u32{ 0, 1, 0 };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };
    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    // Rewrite `version` to 9: a blob from a compiler predating the M13 generic-bound
    // parse change never left a bound in `generic_param.lhs`, so it must miss cleanly.
    std.mem.writeInt(u32, blob[4..8], 9, @import("builtin").cpu.arch.endian());
    try testing.expect((try unpack(gpa, blob)) == null);
}

test "unpack rejects a v10 blob (pre-generic-protocols)" {
    const gpa = testing.allocator;
    var nodes = [_]Node{
        .{ .tag = .literal_number, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(0), .rhs = none },
    };
    var extra = [_]u32{ 0, 1, 0 };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };
    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    // Rewrite `version` to 10: a blob from a compiler predating the M14 generic-protocol
    // parse change never carried a `protocol_decl.rhs` generic-param Range nor a
    // `type_app` in a protocol-ref slot, so it must miss cleanly.
    std.mem.writeInt(u32, blob[4..8], 10, @import("builtin").cpu.arch.endian());
    try testing.expect((try unpack(gpa, blob)) == null);
}

test "pack/unpack round-trips a tree with a generic protocol_decl (v11)" {
    const gpa = testing.allocator;
    // A pure byte round-trip exercising the M14 `protocol_decl.rhs` generic-param Range.
    var nodes = [_]Node{
        .{ .tag = .generic_param, .main_token = 1, .lhs = none, .rhs = none }, // `U`
        .{ .tag = .protocol_decl, .main_token = 0, .lhs = Index.from(3), .rhs = Index.from(5) },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(7), .rhs = none },
    };
    // extra: [0]=sigs Range start, [1]=len(0) at 3; generic Range {start=0,len=1} at 5;
    // program Range {start=1,len=1} at 7. Only the round-trip bytes matter here.
    var extra = [_]u32{ 0, 0, 0, 0, 0, 0, 1, 1, 1 };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };
    const blob = try pack(gpa, tree);
    defer gpa.free(blob);
    const got = (try unpack(gpa, blob)) orelse return error.UnexpectedMiss;
    defer gpa.free(got.nodes);
    defer gpa.free(got.extra);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(tree.nodes), std.mem.sliceAsBytes(got.nodes));
    try testing.expectEqual(@as(usize, 1), protocolGenericParams(got, Index.from(1)).len);
}

test "contentFp ignores Node padding (cold-build fp determinism foundation)" {
    // The `Node` extern struct has 3 padding bytes between `tag` and `main_token`
    // that struct literals leave undefined; on a cold build those carry stack
    // garbage, so hashing the raw bytes flips the fp run-to-run. `contentFp` folds
    // only the four semantic fields, so two trees with IDENTICAL fields but DISTINCT
    // padding must produce the SAME fp — otherwise the content-fp cache is inert
    // (permanent miss) or unsound (cross-build alias). Forge the padding via byte access.
    var a = [_]Node{
        .{ .tag = .binary, .main_token = 1, .lhs = Index.from(0), .rhs = Index.from(2) },
        .{ .tag = .program, .main_token = 0, .lhs = Index.from(0), .rhs = none },
    };
    var b = a;
    // Stamp differing garbage into the padding bytes (offsets 1..3 of each Node).
    const ab = std.mem.sliceAsBytes(a[0..]);
    const bb = std.mem.sliceAsBytes(b[0..]);
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        const base = i * @sizeOf(Node);
        ab[base + 1] = 0xAA;
        ab[base + 2] = 0xBB;
        ab[base + 3] = 0xCC;
        bb[base + 1] = 0x11;
        bb[base + 2] = 0x22;
        bb[base + 3] = 0x33;
    }
    var extra = [_]u32{0};
    const ta = Tree{ .nodes = &a, .extra = &extra };
    const tb = Tree{ .nodes = &b, .extra = &extra };

    // Raw-byte hashing WOULD differ (padding leaked); contentFp must NOT.
    try testing.expect(std.hash.Wyhash.hash(0, ab) != std.hash.Wyhash.hash(0, bb));
    try testing.expectEqual(contentFp(ta), contentFp(tb));

    // And it still discriminates a real field change.
    var c = a;
    c[0].main_token = 99;
    const tc = Tree{ .nodes = &c, .extra = &extra };
    try testing.expect(contentFp(ta) != contentFp(tc));
}
