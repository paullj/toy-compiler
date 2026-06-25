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
pub const Index = u32;
pub const none: Index = std.math.maxInt(Index);

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

        // ---- M10: enums + match (appended last; ordinals frozen) -----------

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
        /// or a nested pattern), or `none` to bind the whole value (M10 leaf bind).
        pattern_binding,

        // ---- M11: literal + or patterns (appended last; ordinals frozen) ----

        /// An int/bool literal pattern. `main_token` is the number/`true`/`false`
        /// token; matched against the scrutinee by equality. `lhs`/`rhs` are `none`.
        pattern_literal,
        /// An or-pattern `A | B | ...`. `main_token` is the first alt's first token.
        /// `lhs` is the `extra` header of a `Range` over >=2 alternative pattern
        /// nodes; `rhs` is `none`. All alts must bind the same names/types.
        pattern_or,

        // ---- M14: module imports (appended last; ordinals frozen) -----------

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
/// nodes carry the `pub` modifier (M14 export visibility). It is memcpy-trivial
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
        const word = idx >> 5;
        if (word >= tree.pub_bits.len) return false;
        return (tree.pub_bits[word] >> @intCast(idx & 31)) & 1 != 0;
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

/// A function's signature, decoded from a fixed 3-cell `FnProto` header in
/// `extra`: `{ret_type_node, params_start, params_len}`.
pub const FnProto = struct {
    /// type-ref node naming the return type, or `none` for unit `()`.
    ret_type: Index,
    /// `param` node indices, in source order.
    params: []const Index,
};

/// Decode the `{start, len}` range header at `header`.
pub fn rangeAt(tree: Tree, header: u32) Range {
    return .{ .start = tree.extra[header], .len = tree.extra[header + 1] };
}

/// The slice of child node indices described by the range header at `header`.
pub fn rangeSlice(tree: Tree, header: u32) []const Index {
    const r = rangeAt(tree, header);
    return tree.extra[r.start .. r.start + r.len]; // u32 == Index
}

/// Decode the 2-cell `if_stmt` header at `header`: `{then_block, else_node}`.
pub fn ifHeaderAt(tree: Tree, header: u32) struct { then_block: Index, else_node: Index } {
    return .{ .then_block = tree.extra[header], .else_node = tree.extra[header + 1] };
}

/// Decode the 2-cell `for_stmt` range header at `header`: `{lo, hi}`.
pub fn forHeaderAt(tree: Tree, header: u32) struct { lo: Index, hi: Index } {
    return .{ .lo = tree.extra[header], .hi = tree.extra[header + 1] };
}

/// Decode the 2-cell `match_arm` header at `header`: `{guard, body}`.
pub fn armHeaderAt(tree: Tree, header: u32) struct { guard: Index, body: Index } {
    return .{ .guard = tree.extra[header], .body = tree.extra[header + 1] };
}

/// Decode the `FnProto` header at `header`.
pub fn protoAt(tree: Tree, header: u32) FnProto {
    const ps = tree.extra[header + 1];
    const pl = tree.extra[header + 2];
    return .{ .ret_type = tree.extra[header], .params = tree.extra[ps .. ps + pl] };
}

/// The root (top-level) node of a non-empty tree — by construction the last.
pub fn root(nodes: []const Node) Index {
    return @intCast(nodes.len - 1);
}

/// "TOYP" — a magic so a foreign/corrupt blob is treated as a cache miss.
pub const parse_magic: u32 = 0x544f5950;

/// Header prefixing a packed `Tree` blob. `extern` so it serializes by memcpy.
pub const ParseHeader = extern struct {
    magic: u32,
    /// Bumped to 4 in M14 to add the trailing `pub_bits` section; older v3 blobs
    /// (no `pub_bits`) miss cleanly via the version check in `unpack`.
    version: u32 = 4,
    node_count: u32,
    extra_count: u32,
    /// Number of `u32` words in the `pub_bits` section (`pubBitsLen(node_count)`).
    pub_words: u32,
};

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
    if (hdr.magic != parse_magic or hdr.version != 4) return null;
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
    const n = nodes[idx];
    const tok_text = tokens[n.main_token].text(source);
    switch (n.tag) {
        .literal_number, .literal_string, .literal_bool, .identifier => try out.writeAll(tok_text),
        .literal_unit => try out.writeAll("()"),
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
            for (rangeSlice(tree, n.rhs)) |arg| {
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
            for (rangeSlice(tree, n.lhs)) |stmt| {
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
            const proto = protoAt(tree, n.lhs);
            try out.print("(fn {s} (", .{tok_text});
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
            for (rangeSlice(tree, n.lhs)) |decl_idx| {
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
            for (rangeSlice(tree, n.lhs), 0..) |seg_tok, i| {
                try out.writeByte(if (i == 0) ' ' else '/');
                try out.writeAll(tokens[seg_tok].text(source));
            }
            if (n.rhs != none) {
                try out.print(" as {s}", .{tokens[n.rhs].text(source)});
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
            const h = ifHeaderAt(tree, n.rhs);
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
            const h = forHeaderAt(tree, n.rhs);
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
            if (n.rhs != none) {
                try out.print(" @{s}", .{tokens[n.rhs].text(source)});
            }
            if (n.lhs != none) {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, n.lhs);
            }
            try out.writeByte(')');
        },
        .continue_stmt => {
            try out.writeAll("(continue");
            if (n.rhs != none) {
                try out.print(" @{s}", .{tokens[n.rhs].text(source)});
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
            for (rangeSlice(tree, n.lhs)) |field| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, field);
            }
            try out.writeByte(')');
        },
        .struct_init => {
            try out.writeAll("(new ");
            try renderNode(out, tree, tokens, source, n.lhs);
            for (rangeSlice(tree, n.rhs)) |fi| {
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
            for (rangeSlice(tree, n.lhs)) |v| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, v);
            }
            try out.writeByte(')');
        },
        .enum_variant_unit => try out.print("(variant.unit {s})", .{tok_text}),
        .enum_variant_tuple => {
            try out.print("(variant.tuple {s}", .{tok_text});
            for (rangeSlice(tree, n.lhs)) |ty| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, ty);
            }
            try out.writeByte(')');
        },
        .enum_variant_struct => {
            try out.print("(variant.struct {s}", .{tok_text});
            for (rangeSlice(tree, n.lhs)) |f| {
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
            for (rangeSlice(tree, n.rhs)) |a| {
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
            for (rangeSlice(tree, n.rhs)) |fi| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, fi);
            }
            try out.writeByte(')');
        },
        .match_expr => {
            try out.writeAll("(match ");
            try renderNode(out, tree, tokens, source, n.lhs);
            for (rangeSlice(tree, n.rhs)) |arm| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, arm);
            }
            try out.writeByte(')');
        },
        .match_arm => {
            const h = armHeaderAt(tree, n.rhs);
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
                for (rangeSlice(tree, n.rhs)) |b| {
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
            for (rangeSlice(tree, n.lhs)) |a| {
                try out.writeByte(' ');
                try renderNode(out, tree, tokens, source, a);
            }
            try out.writeByte(')');
        },
    }
}

const testing = std.testing;

test "rangeSlice and protoAt accessors round-trip on a hand-built tree" {
    // Build extra for: a fn with two params (param nodes 0,1), ret_type node 2.
    // extra layout:
    //   [0]=0,[1]=1            params run (param node indices 0,1)
    //   [2]=0,[3]=2            Range header {start=0, len=2}
    //   [4]=2,[5]=0,[6]=2      FnProto {ret_type=2, params_start=0, params_len=2}
    var extra = [_]u32{ 0, 1, 0, 2, 2, 0, 2 };
    var nodes = [_]Node{
        .{ .tag = .param, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .param, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .identifier, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .block, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .fn_decl, .main_token = 0, .lhs = 4, .rhs = 3 },
    };
    const tree = Tree{ .nodes = &nodes, .extra = &extra };

    const params = rangeSlice(tree, 2);
    try testing.expectEqual(@as(usize, 2), params.len);
    try testing.expectEqual(@as(Index, 0), params[0]);
    try testing.expectEqual(@as(Index, 1), params[1]);

    const proto = protoAt(tree, 4);
    try testing.expectEqual(@as(Index, 2), proto.ret_type);
    try testing.expectEqual(@as(usize, 2), proto.params.len);
    try testing.expectEqual(@as(Index, 1), proto.params[1]);
}

test "pack/unpack byte round-trip" {
    const gpa = testing.allocator;
    var nodes = [_]Node{
        .{ .tag = .literal_number, .main_token = 0, .lhs = none, .rhs = none },
        .{ .tag = .program, .main_token = 0, .lhs = 0, .rhs = none },
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
    try renderNode(&w, tree, &.{Token{ .tag = .l_paren, .start = 0, .end = 1 }}, "()", 0);
    try testing.expectEqualStrings("()", w.buffered());
}

test "unpack rejects a foreign blob" {
    const gpa = testing.allocator;
    try testing.expect((try unpack(gpa, "not a tree")) == null);
    try testing.expect((try unpack(gpa, &.{})) == null);
}
