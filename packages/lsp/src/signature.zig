//! textDocument/signatureHelp: when the cursor sits inside a call's argument list, the
//! callee's signature plus the ACTIVE PARAMETER index.
//!
//! Detection runs FIRST over a fault-tolerant raw LEX of the buffer: the enclosing call is
//! found from the token stream, so an outside-any-call cursor is rejected before any
//! compile. Like completion, this must survive a broken buffer: signature help fires
//! mid-type (`add(1, ` with no closing paren), so an unterminated call is REPAIRED by
//! appending its missing closers at EOF (offset-preserving) and checked as its own
//! `Analysis`; a balanced buffer reads the document's. Resolve/type errors are tolerated —
//! an arity error on the repaired call must not suppress the declared signature.
//!
//! The call is found by its `(`, so any callee shape gets help. A free fn's signature is
//! the one its Callee names (`Analysis.freeCallee`, as hover reads it): a non-generic fn's,
//! the concrete one of the instance a generic call runs, or a generic fn's as declared when
//! no instance is known (in a template's own body, or mid-typing), rendered through
//! `render.bareName` so the module qualifier never leaks. A method call's comes from
//! the checker's Callee table (`Analysis.calleeAt`), shown without its `self`: the receiver
//! is left of the `.`, never inside the parens. A call of a function value (`f(`, `h.run(`,
//! `fs[0](`, `mk()(`) shows the value's type, `fn(int) -> int`, whose params are labelled by
//! type, since a fn type has no param names. A local bound once to a named fn or a bound
//! method (`f := add`, `g := acc.plus`) shows that fn's or method's label instead, as a call
//! of it would, and a local fn its own. Param labels are [start,end) byte spans into the
//! rendered label.
//!
//! KNOWN LIMITATIONS (documented, not bugs):
//!   * A local's binding is followed only when nothing writes the local again; any write
//!     (an assignment, a `mut` argument) leaves its calls labelled by their type. A param, a
//!     field or an element is always labelled by its type.
//!   * A method call in a generic body reads what the body's instances selected; with no
//!     instance it yields no help, and a generic protocol's method shows the protocol's own
//!     params: see `Analysis.CalleeSig`'s limitations.
//!   * A bare pipe target `x |> f` has no `(`, so it yields no help of its own.
//!   * A parse error OTHER than the appended closers (or a stray unmatched closer) re-triggers
//!     discovery's whole-tree discard -> null (the single-edit repair limit, as in completion).

const std = @import("std");
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const render = @import("render.zig");
const places = @import("places.zig");
const Analysis = @import("Analysis.zig");
const Workspace = @import("Workspace.zig");

const Callee = toyc.Callee;
const Token = toyc.Token;
const Type = toyc.Typecheck.Type;
const Layout = toyc.Typecheck.Layout;
const EnumLayout = toyc.Typecheck.EnumLayout;

/// `lparen` is the source offset of the call's `(`.
const Enclosing = struct { lparen: u32, active: u32 };

/// The innermost enclosing call the cursor is inside, or null. A call's `(` is never
/// preceded by a statement-terminator newline, so the IMMEDIATELY preceding token decides
/// `is_call` — no newline-skip (that would misread `foo⏎(x)`, two statements, as a call).
/// An identifier before `(` covers `add(` and the member callee of `recv.f(`; a `]` or `)`
/// covers a callee that is an element or a call's result (`fs[0](`, `mk()(`), and a
/// turbofish. Grouping / `if` / `while` parens are excluded. Only tokens starting strictly
/// before `off` drive the stack, so a `(` exactly at the cursor is "not yet entered" and a
/// cursor just before it is outside.
fn enclosingCall(gpa: std.mem.Allocator, toks: []const Token, off: u32) !?Enclosing {
    const Frame = struct { is_call: bool, lparen_i: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(gpa);
    for (toks, 0..) |t, i| {
        if (t.start >= off) break;
        switch (t.tag) {
            .l_paren => try stack.append(gpa, .{
                .is_call = i > 0 and switch (toks[i - 1].tag) {
                    .identifier, .r_bracket, .r_paren => true,
                    else => false,
                },
                .lparen_i = i,
            }),
            .l_bracket, .l_brace => try stack.append(gpa, .{ .is_call = false, .lparen_i = i }),
            // Pop on any closer regardless of type: a mismatched closer in a broken prefix
            // also fails to re-parse (whole-tree discard -> null), so no extra handling pays.
            .r_paren, .r_bracket, .r_brace => if (stack.items.len > 0) {
                _ = stack.pop();
            },
            else => {},
        }
    }
    var k = stack.items.len;
    while (k > 0) {
        k -= 1;
        if (stack.items[k].is_call) {
            const fr = stack.items[k];
            return .{ .lparen = toks[fr.lparen_i].start, .active = activeParam(toks, fr.lparen_i, off) };
        }
    }
    return null;
}

/// The count of top-level argument separators between the call `(` and `off`. Commas nested
/// in inner ()/[]/{} are at depth > 0 and excluded; an unbalanced inner prefix only drives
/// depth negative, which never over-counts.
fn activeParam(toks: []const Token, lparen_i: usize, off: u32) u32 {
    var depth: i32 = 0;
    var commas: u32 = 0;
    var i = lparen_i + 1;
    while (i < toks.len and toks[i].start < off) : (i += 1) switch (toks[i].tag) {
        .l_paren, .l_bracket, .l_brace => depth += 1,
        .r_paren, .r_bracket, .r_brace => depth -= 1,
        .comma => if (depth == 0) {
            commas += 1;
        },
        else => {},
    };
    return commas;
}

/// Append-only EOF balance. Returns `source` unchanged (same ptr) when already balanced (the
/// common complete-call case -> no copy); appends the unclosed openers' matching closers
/// innermost-first at EOF (offset-preserving, so every original token start stays valid in
/// the recompiled stream); null on an extra closer (unrepairable single-edit).
fn repair(gpa: std.mem.Allocator, source: []const u8, toks: []const Token) !?[]const u8 {
    var stack: std.ArrayList(u8) = .empty;
    defer stack.deinit(gpa);
    for (toks) |t| switch (t.tag) {
        .l_paren => try stack.append(gpa, ')'),
        .l_bracket => try stack.append(gpa, ']'),
        .l_brace => try stack.append(gpa, '}'),
        .r_paren, .r_bracket, .r_brace => {
            if (stack.items.len == 0) return null;
            _ = stack.pop();
        },
        else => {},
    };
    if (stack.items.len == 0) return source;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, source);
    var i = stack.items.len;
    while (i > 0) : (i -= 1) try out.append(gpa, stack.items[i - 1]);
    return try out.toOwnedSlice(gpa);
}

/// Mirrors `render.renderSig` byte-for-byte (for empty `var_names`) while capturing each
/// param's [start,end) span, so a param label always points at the RIGHT occurrence even when
/// two params share a type name. Type/name bytes are borrowed off the compiler tables; the
/// appends copy them into `a`.
fn renderLabel(
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    name: []const u8,
    params: []const Type,
    mut_mask: u64,
    ret: Type,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
    var_names: []const []const u8,
    ranges: *std.ArrayList([2]u32),
) !void {
    try buf.appendSlice(a, "fn ");
    try buf.appendSlice(a, name);
    try buf.append(a, '(');
    for (params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        const s: u32 = @intCast(buf.items.len);
        if (toyc.Modes.isMut(mut_mask, i)) try buf.appendSlice(a, "mut ");
        try render.renderTypeNamed(a, buf, p, layouts, enum_layouts, var_names);
        try ranges.append(a, .{ s, @intCast(buf.items.len) });
    }
    try buf.appendSlice(a, ") -> ");
    try render.renderTypeNamed(a, buf, ret, layouts, enum_layouts, var_names);
}

/// A label spelled from its declaration, as hover shows it: a user protocol's method, or a
/// generic fn. Their generic types (`Option[int]`, `T`) are opaque once typecheck is done,
/// so the checked types would render `?`.
fn renderDeclLabel(
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    pm: *const toyc.Graph.Module,
    decl: u32,
    name: []const u8,
    has_self: bool,
    ranges: *std.ArrayList([2]u32),
) !void {
    const proto = toyc.Ast.protoAt(pm.tree(), pm.nodes[decl].lhs.int());
    try buf.appendSlice(a, "fn ");
    try buf.appendSlice(a, name);
    try buf.append(a, '(');
    const params = proto.params[@min(@intFromBool(has_self), proto.params.len)..];
    for (params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        const s: u32 = @intCast(buf.items.len);
        if (toyc.Ast.isMutParam(pm.tree(), pm.tokens, p)) try buf.appendSlice(a, "mut ");
        try appendTypeText(a, buf, pm, pm.nodes[p.int()].lhs);
        try ranges.append(a, .{ s, @intCast(buf.items.len) });
    }
    try buf.appendSlice(a, ") -> ");
    if (proto.ret_type == toyc.Ast.none) try buf.appendSlice(a, "()") else try appendTypeText(a, buf, pm, proto.ret_type);
}

fn appendTypeText(a: std.mem.Allocator, buf: *std.ArrayList(u8), pm: *const toyc.Graph.Module, ty: toyc.Ast.Index) !void {
    const sp = toyc.Walk.nodeSpan(.{ .tree = pm.tree(), .tokens = pm.tokens, .source = pm.source }, ty);
    try render.appendCollapsed(a, buf, pm.source[sp.start..sp.end]);
}

/// What a call reaches, as its label spells it.
const Hit = union(enum) {
    /// A free fn: its global id. A generic one whose instance is not known shows as
    /// declared.
    func: u32,
    /// A generic fn's instance, with its concrete types.
    instance: u32,
    /// A method, and the member node whose Callee selected it (a bound method value's own
    /// `field_access` for a local bound to one).
    method: struct { sig: Analysis.CalleeSig, node: u32 },
    /// A function value: its type, and the name of the local fn it is, if it is one.
    value: struct { name: ?[]const u8, sig: toyc.LayoutEngine.FnSig },
};

/// The call node whose `(` starts at `lparen`: the call it heads, or the piped call whose
/// `|>` it follows.
fn callAt(an: *const Analysis, lparen: u32) ?u32 {
    const m = an.module();
    if (position.tokenAt(m.tokens, lparen)) |t| for (an.nodesAt(t)) |i| {
        const n = m.nodes[i];
        if (n.tag == .call and n.lhs != toyc.Ast.none) return i;
    };
    for (m.nodes, 0..) |n, i| {
        if (n.tag != .call or n.lhs == toyc.Ast.none or m.tokens[n.main_token].tag != .pipe_gt) continue;
        const open = toyc.Ast.pipeArgsOpen(m.tokens, n.main_token) orelse continue;
        if (m.tokens[open].start == lparen) return @intCast(i);
    }
    return null;
}

/// What `call` reaches. A function value's label is its type, which has no parameter
/// names. A local bound once to a named fn or a bound method (`f := add`, `g := acc.plus`)
/// is labelled as a call of that fn or method would be, and a local fn by its own name.
fn hitOf(a: std.mem.Allocator, an: *Analysis, call: u32) !?Hit {
    const m = an.module();
    const tree = m.tree();
    const resolutions = an.resolutions();
    const callees = an.callees();
    const tc = an.tc();
    const cn = m.nodes[call];
    const slot = (Callee.callSlot(tree, cn) orelse return null).int();
    if (slot >= callees.len) return null;
    if (an.freeCallee(slot)) |fc| return switch (fc) {
        .direct, .generic => |gid| if (gid < tc.sigs.len) .{ .func = gid } else null,
        .instance => |ii| .{ .instance = ii },
    };
    // A free fn call the checker selected nothing for (one in a generic body, which only its
    // instances check) still names its fn.
    if (resolutions[slot] == .func) return if (resolutions[slot].func < tc.sigs.len) .{ .func = resolutions[slot].func } else null;
    if (Callee.memberNode(tree, cn)) |fa| {
        if (try an.calleeAt(a, fa.int())) |ms| return .{ .method = .{ .sig = ms, .node = fa.int() } };
    }
    // The callee's type, not the `.value` tag: a call with an argument error, mid-typing,
    // still has a typed callee.
    const nts = an.nodeTypes();
    const fs = render.fnSigOf(nts[slot], tc.layouts) orelse return null;
    if (m.nodes[slot].tag == .identifier) if (try places.soleBinding(an, slot)) |b| {
        const bn = m.nodes[b];
        if (bn.tag == .local_fn) return .{ .value = .{ .name = m.tokens[bn.main_token].text(m.source), .sig = fs } };
        const init = bn.lhs.int();
        if (init < callees.len) switch (callees[init].view()) {
            .fn_ref => |gid| if (gid < tc.sigs.len) return .{ .func = gid },
            .bound_method => if (try an.calleeAt(a, init)) |ms| return .{ .method = .{ .sig = ms, .node = init } },
            else => {},
        };
    };
    return .{ .value = .{ .name = null, .sig = fs } };
}

/// A function value's label: its type as `render.renderType` spells it, a unit return
/// left off, with each param's span.
fn renderFnTypeLabel(
    a: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    fs: toyc.LayoutEngine.FnSig,
    layouts: []const Layout,
    enum_layouts: []const EnumLayout,
    ranges: *std.ArrayList([2]u32),
) !void {
    try buf.appendSlice(a, "fn(");
    for (fs.params, 0..) |p, i| {
        if (i != 0) try buf.appendSlice(a, ", ");
        const s: u32 = @intCast(buf.items.len);
        try render.renderType(a, buf, p, layouts, enum_layouts);
        try ranges.append(a, .{ s, @intCast(buf.items.len) });
    }
    try buf.append(a, ')');
    if (fs.ret.kind == .unit) return;
    try buf.appendSlice(a, " -> ");
    try render.renderType(a, buf, fs.ret, layouts, enum_layouts);
}

/// The signature help at (`line`, `character`) in `doc`'s document, allocated in `a`, or
/// null when the cursor is not inside a resolvable call. A repaired buffer is checked with
/// `gpa` in `ws` and torn down before this returns.
pub fn at(a: std.mem.Allocator, gpa: std.mem.Allocator, ws: Workspace, doc: *Analysis, line: u32, character: u32) !?protocol.SignatureHelp {
    const source = doc.text;
    const toks = try toyc.Lexer.tokenize(a, source);
    const off = doc.offsetAt(line, character) orelse return null;
    const enc = (try enclosingCall(a, toks, off)) orelse return null;

    const repaired = (try repair(a, source, toks)) orelse return null;
    if (repaired.ptr == source.ptr) return help(a, doc, enc);
    const an = try Analysis.init(gpa, ws, doc.uri, repaired, doc.version);
    defer an.deinit();
    return help(a, an, enc);
}

fn help(a: std.mem.Allocator, an: *Analysis, enc: Enclosing) !?protocol.SignatureHelp {
    if (an.checked == null) return null;
    const m = an.module();
    const tc = an.tc();
    const call = callAt(an, enc.lparen) orelse return null;
    const callee = (try hitOf(a, an, call)) orelse return null;

    // In `x |> f(a, ..)` the piped `x` is argument 0 but sits left of the `(`, so the comma
    // count is one short. The tree, not the tokens, says whether the call is piped: a token
    // look-back from the callee would miss an `m.f` callee. A value written at a `_` is
    // already counted.
    var active = enc.active;
    if (toyc.Ast.pipePrepends(m.tree(), m.tokens, m.source, toyc.Ast.Index.from(call))) active += 1;

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    switch (callee) {
        .func => |fid| {
            const sig = tc.sigs[fid];
            const name = render.unmangled(render.bareName(sig.name));
            const gf = an.res().fns[fid];
            // A generic fn's checked types are opaque: it shows as declared.
            if (sig.isGenericTemplate() and gf.decl_node != toyc.Ast.none)
                try renderDeclLabel(a, &label, &an.graph.modules[gf.module], gf.decl_node.int(), name, false, &ranges)
            else
                try renderLabel(a, &label, name, sig.params, sig.mut_mask, sig.ret, tc.layouts, tc.enum_layouts, &.{}, &ranges);
        },
        .instance => |ii| {
            const inst = tc.instances[ii];
            const sig = tc.sigs[inst.template_gid];
            try renderLabel(a, &label, render.unmangled(render.bareName(sig.name)), inst.params, sig.mut_mask, inst.ret, tc.layouts, tc.enum_layouts, &.{}, &ranges);
        },
        .method => |hm| {
            const ms = hm.sig;
            if (ms.decl != null and ms.spell != .checked)
                try renderDeclLabel(a, &label, &an.graph.modules[ms.decl.?.module], ms.decl.?.node, ms.name, ms.has_self, &ranges)
            else
                try renderLabel(a, &label, ms.name, ms.params[@intFromBool(ms.has_self)..], ms.mut_mask >> @intFromBool(ms.has_self), ms.ret, tc.layouts, tc.enum_layouts, ms.var_names, &ranges);
        },
        .value => |v| if (v.name) |name|
            try renderLabel(a, &label, name, v.sig.params, 0, v.sig.ret, tc.layouts, tc.enum_layouts, &.{}, &ranges)
        else
            try renderFnTypeLabel(a, &label, v.sig, tc.layouts, tc.enum_layouts, &ranges),
    }

    const params = try a.alloc(protocol.ParameterInformation, ranges.items.len);
    for (ranges.items, 0..) |r, i| params[i] = .{ .label = r };

    if (ranges.items.len == 0) {
        active = 0;
    } else if (active > ranges.items.len - 1) {
        active = @intCast(ranges.items.len - 1);
    }

    const sigs = try a.alloc(protocol.SignatureInformation, 1);
    sigs[0] = .{ .label = label.items, .parameters = params };
    return .{ .signatures = sigs, .activeSignature = 0, .activeParameter = active };
}

const testing = std.testing;
const test_util = @import("test_util.zig");

/// A test's signature help: its analysis and the answer, torn down together.
const Result = struct {
    an: *Analysis,
    arena: std.heap.ArenaAllocator,
    help: protocol.SignatureHelp,

    fn deinit(r: *Result) void {
        r.arena.deinit();
        r.an.deinit();
    }
};

/// The signature help at (`line`, `character`) in `source`, checked as the document
/// `doc_uri` in `ws`.
fn signatureHelpAt(gpa: std.mem.Allocator, ws: Workspace, source: []const u8, line: u32, character: u32, doc_uri: []const u8) !?Result {
    const an = try test_util.analyzeIn(gpa, ws, doc_uri, source);
    errdefer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const h = (try at(arena.allocator(), gpa, ws, an, line, character)) orelse {
        arena.deinit();
        an.deinit();
        return null;
    };
    return .{ .an = an, .arena = arena, .help = h };
}

fn tok(tag: toyc.Tag, start: u32, end: u32) Token {
    return .{ .tag = tag, .start = start, .end = end };
}

test "activeParam: counts only top-level commas" {
    // `add(1, 2)` — add[0,3) ([3,4) 1[4,5) ,[5,6) 2[7,8) )[8,9)
    const toks = [_]Token{
        tok(.identifier, 0, 3), tok(.l_paren, 3, 4), tok(.number, 4, 5),
        tok(.comma, 5, 6),      tok(.number, 7, 8),  tok(.r_paren, 8, 9),
    };
    try testing.expectEqual(@as(u32, 0), activeParam(&toks, 1, 5)); // before the comma
    try testing.expectEqual(@as(u32, 1), activeParam(&toks, 1, 7)); // after the comma
}

test "activeParam: a comma nested in inner parens does not count" {
    // `f(g(1,2), 3)` — f( g( 1 , 2 ) , 3 )
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2),   tok(.identifier, 2, 3),
        tok(.l_paren, 3, 4),    tok(.number, 4, 5),    tok(.comma, 5, 6),
        tok(.number, 6, 7),     tok(.r_paren, 7, 8),   tok(.comma, 8, 9),
        tok(.number, 10, 11),   tok(.r_paren, 11, 12),
    };
    // Cursor past the OUTER comma (offset 10): the inner `,` at depth 1 is excluded.
    try testing.expectEqual(@as(u32, 1), activeParam(&toks, 1, 10));
}

test "activeParam: a bracket comma does not count" {
    // `f(a[i,j], 3)`
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2),    tok(.identifier, 2, 3),
        tok(.l_bracket, 3, 4),  tok(.identifier, 4, 5), tok(.comma, 5, 6),
        tok(.identifier, 6, 7), tok(.r_bracket, 7, 8),  tok(.comma, 8, 9),
        tok(.number, 10, 11),   tok(.r_paren, 11, 12),
    };
    try testing.expectEqual(@as(u32, 1), activeParam(&toks, 1, 10));
}

test "enclosingCall: a bare call is entered at its paren" {
    // `add(1)` — add[0,3) ([3,4) 1[4,5) )[5,6)
    const toks = [_]Token{ tok(.identifier, 0, 3), tok(.l_paren, 3, 4), tok(.number, 4, 5), tok(.r_paren, 5, 6) };
    const enc = (try enclosingCall(testing.allocator, &toks, 4)).?;
    try testing.expectEqual(@as(u32, 3), enc.lparen);
    try testing.expectEqual(@as(u32, 0), enc.active);
}

test "enclosingCall: the INNERMOST call wins for a nested call" {
    // `f(g(1))` — f( g( 1 ) ) with g's identifier at [2,3).
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.l_paren, 1, 2), tok(.identifier, 2, 3),
        tok(.l_paren, 3, 4),    tok(.number, 4, 5),  tok(.r_paren, 5, 6),
        tok(.r_paren, 6, 7),
    };
    const enc = (try enclosingCall(testing.allocator, &toks, 5)).?; // inside the inner call
    try testing.expectEqual(@as(u32, 3), enc.lparen); // g's, not f's
}

test "enclosingCall: a grouping paren is not a call" {
    // `x := (a)` — x[0,1) :=[2,4) ([5,6) a[6,7) )[7,8)
    const toks = [_]Token{
        tok(.identifier, 0, 1), tok(.colon_eq, 2, 4), tok(.l_paren, 5, 6),
        tok(.identifier, 6, 7), tok(.r_paren, 7, 8),
    };
    try testing.expectEqual(@as(?Enclosing, null), try enclosingCall(testing.allocator, &toks, 7));
}

test "enclosingCall: a newline before the paren is not a call" {
    // `foo⏎(x)` — foo[0,3) newline[3,3] ([4,5) x[5,6) )[6,7). Two statements, not a call.
    const toks = [_]Token{
        tok(.identifier, 0, 3), tok(.newline, 3, 3), tok(.l_paren, 4, 5),
        tok(.identifier, 5, 6), tok(.r_paren, 6, 7),
    };
    try testing.expectEqual(@as(?Enclosing, null), try enclosingCall(testing.allocator, &toks, 5));
}

test "enclosingCall: an element or a call result before the paren is a callee" {
    // `fs[0](1)` — fs[0,2) [[2,3) 0[3,4) ][4,5) ([5,6) 1[6,7) )[7,8)
    const elem = [_]Token{
        tok(.identifier, 0, 2), tok(.l_bracket, 2, 3), tok(.number, 3, 4),  tok(.r_bracket, 4, 5),
        tok(.l_paren, 5, 6),    tok(.number, 6, 7),    tok(.r_paren, 7, 8),
    };
    try testing.expectEqual(@as(u32, 5), (try enclosingCall(testing.allocator, &elem, 6)).?.lparen);
    // `mk()(1)` — mk[0,2) ([2,3) )[3,4) ([4,5) 1[5,6) )[6,7)
    const result = [_]Token{
        tok(.identifier, 0, 2), tok(.l_paren, 2, 3), tok(.r_paren, 3, 4), tok(.l_paren, 4, 5),
        tok(.number, 5, 6),     tok(.r_paren, 6, 7),
    };
    try testing.expectEqual(@as(u32, 4), (try enclosingCall(testing.allocator, &result, 5)).?.lparen);
}

test "enclosingCall: an unterminated call at EOF still resolves with the active param" {
    // `add(1, ` — add[0,3) ([3,4) 1[4,5) ,[5,6) eof[7,7)
    const toks = [_]Token{
        tok(.identifier, 0, 3), tok(.l_paren, 3, 4), tok(.number, 4, 5),
        tok(.comma, 5, 6),      tok(.eof, 7, 7),
    };
    const enc = (try enclosingCall(testing.allocator, &toks, 7)).?;
    try testing.expectEqual(@as(u32, 3), enc.lparen);
    try testing.expectEqual(@as(u32, 1), enc.active);
}

test "enclosingCall: no call yields null" {
    const toks = [_]Token{ tok(.identifier, 0, 6), tok(.identifier, 7, 8), tok(.eof, 8, 8) };
    try testing.expectEqual(@as(?Enclosing, null), try enclosingCall(testing.allocator, &toks, 8));
}

test "repair: a balanced source is returned unchanged (same pointer)" {
    const gpa = testing.allocator;
    const src = "fn main() -> int { return 0 }";
    const toks = try toyc.Lexer.tokenize(gpa, src);
    defer gpa.free(toks);
    const out = (try repair(gpa, src, toks)).?;
    try testing.expect(out.ptr == src.ptr);
}

test "repair: an unterminated call appends the missing closers at EOF" {
    const gpa = testing.allocator;
    const src = "fn f() -> int { g(1, ";
    const toks = try toyc.Lexer.tokenize(gpa, src);
    defer gpa.free(toks);
    const out = (try repair(gpa, src, toks)).?;
    defer gpa.free(out);
    try testing.expect(out.ptr != src.ptr);
    // Two open parens + one open brace, closed innermost-first: `)` then `}`.
    try testing.expect(std.mem.startsWith(u8, out, src));
    try testing.expectEqualStrings(")}", out[src.len..]);
}

test "repair: an extra closer is unrepairable" {
    const gpa = testing.allocator;
    const src = ")";
    const toks = try toyc.Lexer.tokenize(gpa, src);
    defer gpa.free(toks);
    try testing.expectEqual(@as(?[]const u8, null), try repair(gpa, src, toks));
}

test "renderLabel: matches renderSig and spans the correct param occurrences" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const int = Type.int;
    const Sig = struct { name: []const u8, params: []const Type, ret: Type, mut_mask: u64 = 0 };
    const sig: Sig = .{ .name = "m.add", .params = &.{ int, int }, .ret = int };

    const no_layouts: []const Layout = &.{};
    const no_enums: []const EnumLayout = &.{};

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    try renderLabel(a, &label, render.bareName(sig.name), sig.params, sig.mut_mask, sig.ret, no_layouts, no_enums, &.{}, &ranges);

    var expect: std.ArrayList(u8) = .empty;
    try render.renderSig(a, &expect, sig, no_layouts, no_enums);
    try testing.expectEqualStrings(expect.items, label.items);
    try testing.expectEqualStrings("fn add(int, int) -> int", label.items);

    // Two params; the SECOND `int` span is the later occurrence — proves the offset label and
    // that `bareName` dropped the `m.` qualifier (label starts `fn add`, not `fn m.add`).
    try testing.expectEqual(@as(usize, 2), ranges.items.len);
    try testing.expectEqual([2]u32{ 7, 10 }, ranges.items[0]);
    try testing.expectEqual([2]u32{ 12, 15 }, ranges.items[1]);
}

test "renderLabel: a zero-param sig has no ranges" {
    const gpa = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    const int = Type.int;
    const Sig = struct { name: []const u8, params: []const Type, ret: Type, mut_mask: u64 = 0 };
    const sig: Sig = .{ .name = "foo", .params = &.{}, .ret = int };

    const no_layouts: []const Layout = &.{};
    const no_enums: []const EnumLayout = &.{};

    var label: std.ArrayList(u8) = .empty;
    var ranges: std.ArrayList([2]u32) = .empty;
    try renderLabel(a, &label, render.bareName(sig.name), sig.params, sig.mut_mask, sig.ret, no_layouts, no_enums, &.{}, &ranges);
    try testing.expectEqualStrings("fn foo() -> int", label.items);
    try testing.expectEqual(@as(usize, 0), ranges.items.len);
}

test "signatureHelpAt: a piped call's active param counts the piped value" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn add3(a: int, b: int, c: int) -> int { return a + b + c }
        \\fn main() -> int {
        \\    h := 5 |> add(2, _)
        \\    k := 5 |> add(_, 2)
        \\    x := 1 |> add(2)
        \\    y := 1 |> add3(2, 3)
        \\    z := add(1, 2) |> add(3)
        \\    return x + y + z + h + k
        \\}
        \\
    ;
    const cases = [_]struct { after: []const u8, label: []const u8, active: u32 }{
        .{ .after = "h := 5 |> add(2, ", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "h := 5 |> add(", .label = "fn add(int, int) -> int", .active = 0 },
        .{ .after = "k := 5 |> add(_, ", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "k := 5 |> add(", .label = "fn add(int, int) -> int", .active = 0 },
        .{ .after = "1 |> add(", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "1 |> add3(2, ", .label = "fn add3(int, int, int) -> int", .active = 2 },
        .{ .after = "z := add(1, ", .label = "fn add(int, int) -> int", .active = 1 },
        .{ .after = "z := add(", .label = "fn add(int, int) -> int", .active = 0 },
        .{ .after = "|> add(", .label = "fn add(int, int) -> int", .active = 1 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a piped module-qualified call counts the piped value" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    try docs.put(gpa, "file:///p/helper.toy", "pub fn add(a: int, b: int, c: int) -> int { return a + b + c }\n", 1);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = "import helper\nfn main() -> int {\n    return 1 |> helper.add(2, 3)\n}\n";
    const cases = [_]struct { after: []const u8, active: u32 }{
        .{ .after = "helper.add(", .active = 1 },
        .{ .after = "helper.add(2, ", .active = 2 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse
            return error.NoSignatureHelp;
        defer r.deinit();
        try testing.expectEqualStrings("fn add(int, int, int) -> int", r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: method calls show the selected method, piped or not" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = test_util.method_fixture;
    const plus = "fn plus(int, int) -> int";
    const cases = [_]struct { after: []const u8, label: []const u8, active: u32 }{
        .{ .after = "x := acc.plus(", .label = plus, .active = 0 },
        .{ .after = "acc.plus(1, ", .label = plus, .active = 1 },
        .{ .after = "1 |> acc.plus(", .label = plus, .active = 1 },
        .{ .after = "Box[int].make(", .label = "fn make(int) -> Box[int]", .active = 0 },
        .{ .after = "b.get(", .label = "fn get() -> int", .active = 0 },
        .{ .after = "v.dbl(", .label = "fn dbl() -> int", .active = 0 },
        .{ .after = "\"hi\".concat(", .label = "fn concat(str) -> str", .active = 0 },
        .{ .after = "(3).eq(", .label = "fn eq(int) -> bool", .active = 0 },
        .{ .after = "o.or(", .label = "fn or(int) -> int", .active = 0 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a generic method's own type params show while its call is incomplete" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const pre =
        \\struct Box[T] { v: T }
        \\impl Box[T] {
        \\    fn map[U](self, f: fn(T) -> U) -> Box[U] { Box { v: f(self.v) } }
        \\    fn fold[A](self, init: A, f: fn(A, T) -> A) -> A { f(init, self.v) }
        \\}
        \\fn main() -> int {
        \\    b := Box { v: 2 }
        \\
    ;
    const cases = [_]struct { tail: []const u8, label: []const u8, active: u32 }{
        .{ .tail = "    k := b.map(", .label = "fn map(fn(T) -> U) -> Box[U]", .active = 0 },
        .{ .tail = "    k := b.fold(0, ", .label = "fn fold(A, fn(A, T) -> A) -> A", .active = 1 },
        .{ .tail = "    k := b.map(fn(s) { s.nope() }", .label = "fn map(fn(T) -> U) -> Box[U]", .active = 0 },
    };
    for (cases) |c| {
        const src = try std.mem.concat(gpa, u8, &.{ pre, c.tail });
        defer gpa.free(src);
        const p = test_util.posAfterLast(src, c.tail);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.tail});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a method pipe's piped value is the receiver, not an argument" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct Acc { n: int }
        \\impl Acc {
        \\    fn plus(self, a: int, b: int) -> int { return self.n + a + b }
        \\}
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\    y := 1 |> acc.plus(2)
        \\    z := acc |> .plus(1, 2)
        \\    return y + z
        \\}
        \\
    ;
    const cases = [_]struct { after: []const u8, active: u32 }{
        .{ .after = "|> .plus(", .active = 0 },
        .{ .after = "|> .plus(1, ", .active = 1 },
        .{ .after = "1 |> acc.plus(", .active = 1 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings("fn plus(int, int) -> int", r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
    }
}

test "signatureHelpAt: a user protocol's method is labelled from its declaration" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\protocol Finder { fn find(self, k: Option[int], n: int) -> Option[int] }
        \\struct P { x: int }
        \\impl P has Finder { fn find(self, k: Option[int], n: int) -> Option[int] { k } }
        \\fn look[T has Finder](v: T) -> int { return v.find(Option.some(1), 2).or(0) }
        \\fn main() -> int { return look(P { x: 1 }) }
        \\
    ;
    const cases = [_]struct { after: []const u8, active: u32 }{
        .{ .after = "v.find(", .active = 0 },
        .{ .after = "v.find(Option.some(1), ", .active = 1 },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse
            return error.NoSignatureHelp;
        defer r.deinit();
        const label = "fn find(Option[int], int) -> Option[int]";
        try testing.expectEqualStrings(label, r.help.signatures[0].label);
        try testing.expectEqual(c.active, r.help.activeParameter);
        const params = r.help.signatures[0].parameters;
        try testing.expectEqual(@as(usize, 2), params.len);
        try testing.expectEqualStrings("Option[int]", label[params[0].label[0]..params[0].label[1]]);
        try testing.expectEqualStrings("int", label[params[1].label[0]..params[1].label[1]]);
    }
}

test "signatureHelpAt: an unterminated method call keeps its selection" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    // The repaired `acc.plus(1, )` is an arity error; the selection survives it.
    const src =
        \\struct Acc { n: int }
        \\impl Acc {
        \\    fn plus(self, a: int, b: int) -> int { return self.n + a + b }
        \\}
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\
    ++ "    x := acc.plus(1, ";
    const p = test_util.posAfterLast(src, "acc.plus(1, ");
    var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse
        return error.NoSignatureHelp;
    defer r.deinit();
    try testing.expectEqualStrings("fn plus(int, int) -> int", r.help.signatures[0].label);
    try testing.expectEqual(@as(u32, 1), r.help.activeParameter);
}

test "signatureHelpAt: a mut param's label says mut" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\struct P { x: int }
        \\impl P {
        \\    fn bump(mut self, mut d: int) { self.x = self.x + d }
        \\}
        \\fn inc(mut n: int, k: int) { n = n + k }
        \\fn main() -> int {
        \\    c := 1
        \\    inc(c, 2)
        \\    p := P { x: 1 }
        \\    p.bump(c)
        \\    return c + p.x
        \\}
        \\
    ;
    const cases = [_]struct { after: []const u8, label: []const u8 }{
        .{ .after = "inc(", .label = "fn inc(mut int, int) -> ()" },
        .{ .after = "p.bump(", .label = "fn bump(mut int) -> ()" },
    };
    for (cases) |c| {
        const p = test_util.posAfterLast(src, c.after);
        var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
            std.debug.print("no signature help after '{s}'\n", .{c.after});
            return error.NoSignatureHelp;
        };
        defer r.deinit();
        try testing.expectEqualStrings(c.label, r.help.signatures[0].label);
    }
}

test "renderFnTypeLabel: spells the type as renderType does and spans each param" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pair_names = [_][]const u8{ "code", "env" };
    const pair_types = [_]Type{ Type.int, Type.rawptr };
    const pair_offs = [_]u32{ 0, 8 };
    const inner: Layout = .{ .name = "fn$int$int", .field_names = @constCast(&pair_names), .field_types = @constCast(&pair_types), .offsets = @constCast(&pair_offs), .size = 16, .@"align" = 8, .fn_sig = .{ .params = &.{Type.int}, .ret = Type.int } };
    var outer = inner;
    outer.name = "fn$s0$int$unit";
    outer.fn_sig = .{ .params = &.{ Type.structT(0), Type.int }, .ret = Type.unit };
    const layouts = [_]Layout{ inner, outer };
    const cases = [_]struct { t: Type, want: []const u8, spans: []const [2]u32 }{
        .{ .t = Type.structT(0), .want = "fn(int) -> int", .spans = &.{.{ 3, 6 }} },
        // A unit return is left off; a fn-typed param spans its whole type.
        .{ .t = Type.structT(1), .want = "fn(fn(int) -> int, int)", .spans = &.{ .{ 3, 17 }, .{ 19, 22 } } },
    };
    for (cases) |c| {
        var label: std.ArrayList(u8) = .empty;
        var ranges: std.ArrayList([2]u32) = .empty;
        try renderFnTypeLabel(a, &label, render.fnSigOf(c.t, &layouts).?, &layouts, &.{}, &ranges);
        var expect: std.ArrayList(u8) = .empty;
        try render.renderType(a, &expect, c.t, &layouts, &.{});
        try testing.expectEqualStrings(expect.items, label.items);
        try testing.expectEqualStrings(c.want, label.items);
        try testing.expectEqualSlices([2]u32, c.spans, ranges.items);
    }
}

/// Signature help just past the last `after` in `src`.
fn expectHelp(src: []const u8, after: []const u8, label: []const u8, active: u32) !void {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const p = test_util.posAfterLast(src, after);
    var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")) orelse {
        std.debug.print("no signature help after '{s}'\n", .{after});
        return error.NoSignatureHelp;
    };
    defer r.deinit();
    testing.expectEqualStrings(label, r.help.signatures[0].label) catch |e| {
        std.debug.print("after '{s}'\n", .{after});
        return e;
    };
    try testing.expectEqual(active, r.help.activeParameter);
}

/// `expectHelp` at the end of `pre ++ tail`, a buffer the case leaves mid-call.
fn expectHelpAfter(pre: []const u8, tail: []const u8, label: []const u8, active: u32) !void {
    const src = try std.mem.concat(testing.allocator, u8, &.{ pre, tail });
    defer testing.allocator.free(src);
    try expectHelp(src, tail, label, active);
}

const value_fixture =
    \\import std/vec
    \\struct Acc { n: int }
    \\impl Acc {
    \\    fn plus(self, k: int) -> int { self.n + k }
    \\    fn two(self, a: int, b: str) -> int { self.n + a + b.len() }
    \\}
    \\struct H { run: fn(int) -> int, show: fn(str) }
    \\fn add(a: int, b: int) -> int { a + b }
    \\fn dbl(x: int) -> int { x * 2 }
    \\fn say(s: str) { }
    \\fn mk() -> fn(int, int) -> int { add }
    \\fn relabel(mut f: fn(int) -> int) { f = dbl }
    \\fn apply(f: fn(int, int) -> int, v: int) -> int { f(v, 1) }
    \\fn main() -> int {
    \\    acc := Acc { n: 1 }
    \\    other := Acc { n: 2 }
    \\    g := acc.plus
    \\    t := acc.two
    \\    f := add
    \\    h := H { run: dbl, show: say }
    \\    fs := [dbl, dbl]
    \\    i := 0
    \\    re := acc.plus
    \\    re = other.plus
    \\    mu := acc.plus
    \\    relabel(mu)
    \\    fn step(k: int) -> int { k + 1 }
    \\
;

test "signatureHelpAt: a call of a function value shows its type, params labelled by type" {
    const cases = [_]struct { tail: []const u8, label: []const u8, active: u32 }{
        .{ .tail = "    x := h.run(", .label = "fn(int) -> int", .active = 0 },
        .{ .tail = "    h.show(", .label = "fn(str)", .active = 0 },
        .{ .tail = "    x := fs[0](", .label = "fn(int) -> int", .active = 0 },
        .{ .tail = "    x := fs[i](", .label = "fn(int) -> int", .active = 0 },
        .{ .tail = "    x := mk()(1, ", .label = "fn(int, int) -> int", .active = 1 },
        // Written after its binding, so its value may come from anywhere.
        .{ .tail = "    x := re(", .label = "fn(int) -> int", .active = 0 },
        .{ .tail = "    x := mu(", .label = "fn(int) -> int", .active = 0 },
        // The piped value is argument 0; the active param stops at the last one.
        .{ .tail = "    x := 1 |> h.run(", .label = "fn(int) -> int", .active = 0 },
    };
    for (cases) |c| try expectHelpAfter(value_fixture, c.tail, c.label, c.active);
    try expectHelp(value_fixture, "f(v, ", "fn(int, int) -> int", 1);
    try expectHelp(value_fixture, "f(", "fn(int, int) -> int", 0);

    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = value_fixture ++ "    x := fs[0](1)\n";
    const p = test_util.posAfterLast(src, "fs[0](");
    var r = (try signatureHelpAt(gpa, ws, src, p.line, p.character, "file:///p/main.toy")).?;
    defer r.deinit();
    const params = r.help.signatures[0].parameters;
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expectEqualStrings("int", r.help.signatures[0].label[params[0].label[0]..params[0].label[1]]);
}

test "signatureHelpAt: a local bound once to a named fn, a bound method or a local fn shows that declaration" {
    const cases = [_]struct { tail: []const u8, label: []const u8, active: u32 }{
        .{ .tail = "    x := g(", .label = "fn plus(int) -> int", .active = 0 },
        .{ .tail = "    x := t(1, ", .label = "fn two(int, str) -> int", .active = 1 },
        .{ .tail = "    x := f(", .label = "fn add(int, int) -> int", .active = 0 },
        .{ .tail = "    x := step(", .label = "fn step(int) -> int", .active = 0 },
        .{ .tail = "    x := 3 |> t(", .label = "fn two(int, str) -> int", .active = 1 },
        .{ .tail = "    x := 3 |> f(", .label = "fn add(int, int) -> int", .active = 1 },
        // Captured by a closure, it is still the local it was bound as.
        .{ .tail = "    x := apply(fn(a, b) { g(", .label = "fn plus(int) -> int", .active = 0 },
        .{ .tail = "    x := add(g(", .label = "fn plus(int) -> int", .active = 0 },
    };
    for (cases) |c| try expectHelpAfter(value_fixture, c.tail, c.label, c.active);
}

test "signatureHelpAt: a function value's call keeps its help while incomplete or erroneous" {
    const cases = [_]struct { tail: []const u8, label: []const u8, active: u32 }{
        .{ .tail = "    x := h.run(1, ", .label = "fn(int) -> int", .active = 0 },
        .{ .tail = "    x := fs[0](nope, ", .label = "fn(int) -> int", .active = 0 },
        .{ .tail = "    x := mk()(", .label = "fn(int, int) -> int", .active = 0 },
        .{ .tail = "    x := g(\"s\", ", .label = "fn plus(int) -> int", .active = 0 },
        .{ .tail = "    x := t(nope, ", .label = "fn two(int, str) -> int", .active = 1 },
    };
    for (cases) |c| try expectHelpAfter(value_fixture, c.tail, c.label, c.active);
}

test "signatureHelpAt: a paren that opens no call yields null" {
    const gpa = testing.allocator;
    var docs: @import("Documents.zig") = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\fn add(a: int, b: int) -> int { a + b }
        \\fn id[T](x: T) -> T { x }
        \\fn main() -> int { return add(1, 2) + id(3) }
        \\
    ;
    for ([_][]const u8{ "add(a: int, ", "id[T](" }) |after| {
        const p = test_util.posOf(src, after);
        const r = try signatureHelpAt(gpa, ws, src, p.line, p.character + @as(u32, @intCast(after.len)), "file:///p/main.toy");
        try testing.expect(r == null);
    }
}

test "signatureHelpAt: a generic fn's call shows the instance it runs, explicit type args too; in a template body, as declared" {
    const src =
        \\fn id[T](x: T) -> T { x }
        \\fn pair[T has Hash](v: T) -> int { id(v).hash() }
        \\fn main() -> int {
        \\    a := id(7)
        \\    b := id[str]("s")
        \\    return a + b.len() + pair(3)
        \\}
        \\
    ;
    try expectHelp(src, "a := id(", "fn id(int) -> int", 0);
    try expectHelp(src, "id[str](", "fn id(str) -> str", 0);
    try expectHelp(src, "{ id(", "fn id(T) -> T", 0);
    const boxed = "struct Box[T] { v: T }\nimpl Box[T] { fn wrap[U](self, u: U, t: T) -> U { u } }\nfn main() -> int {\n    b := Box { v: 1 }\n";
    // Incomplete, so it picked no instance: shown as declared.
    try expectHelpAfter(boxed, "    x := b.wrap[str](", "fn wrap(U, T) -> U", 0);
}

test "signatureHelpAt: method calls inside generic bodies (#33), and free fn calls there" {
    const src = test_util.generic_body_fixture;
    try expectHelp(src, "self.get(", "fn get() -> T", 0);
    try expectHelp(src, "b.get(", "fn get() -> int", 0);
    try expectHelp(src, "{ b.get(", "fn get() -> T", 0);
    try expectHelp(src, "acc.plus(", "fn plus(int) -> int", 0);
    const free =
        \\fn id[T](x: T) -> T { x }
        \\fn dbl(x: int) -> int { x * 2 }
        \\fn wrap[T](v: T) -> T {
        \\    k := dbl(1)
        \\    id(v)
        \\}
        \\fn main() -> int { return wrap(1) + wrap("s").len() }
        \\
    ;
    try expectHelp(free, "dbl(", "fn dbl(int) -> int", 0);
    try expectHelp(free, "    id(", "fn id(T) -> T", 0);
}
