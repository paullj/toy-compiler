//! The fingerprint's body walks, lifted out of Driver.zig into the query engine.
//!
//! These collect the inputs `Fingerprint.fingerprint` folds: (b) the callee
//! signatures in body-walk order (`walkCalls`, which MUST mirror
//! `Fingerprint.walk` positionally so the order-sensitive [C7] fold lines up) and
//! (c) the touched-type layout descriptors (`walkTouched`/`walkTouchedSig` +
//! `appendTouched` + `structLayoutBytes`/`enumLayoutBytes`). `typeRefToType`
//! resolves a declared param/return type-ref by name; `freeTouched` frees the
//! layout slices a `walkTouched` result owns.
//!
//! Moved VERBATIM as one block (only the `frozen` param is now `anytype` so the
//! single-file and graph callers can share it — both pass `*const Driver.Frozen`).
//! NEVER edit either switch independently of `Fingerprint.walk`: the two are
//! hand-mirrored and any drift silently breaks the [C7] order-sensitive fold.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Typecheck = @import("../types.zig");
const Fingerprint = @import("Fingerprint.zig");

/// Collect the signatures of every function this fn calls, in body walk order
/// (matching `Fingerprint`'s walk) so the fingerprint's (b) component lines up.
pub fn walkCalls(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, out: *std.ArrayList(Fingerprint.Sig)) !void {
    if (idx == Ast.none) return;
    const tree = frozen.tree;
    const n = tree.nodes[idx];
    switch (n.tag) {
        .literal_number, .literal_string, .literal_bool, .identifier, .literal_unit => {},
        .unary => try walkCalls(gpa, frozen, n.lhs, out),
        .binary => {
            try walkCalls(gpa, frozen, n.lhs, out);
            try walkCalls(gpa, frozen, n.rhs, out);
        },
        .call => {
            // The callee leaf is the lhs identifier; recurse it first (matches the
            // Fingerprint walk), then record the callee identity+sig, then the args.
            try walkCalls(gpa, frozen, n.lhs, out);
            const res = frozen.resolutions[n.lhs];
            // `res.func` indexes BOTH `names` (the resolved SymName{kind,name}, what
            // the .func reloc target carries) and `sigs` (params/ret). Fold the full
            // identity so a builtin↔user_fn shadow switch flips the caller's hash. [Cx]
            if (res == .func and res.func < frozen.sigs.len and res.func < frozen.names.len) {
                const sig = frozen.sigs[res.func];
                const nm = frozen.names[res.func];
                try out.append(gpa, .{ .kind = nm.kind, .name = nm.name, .params = sig.params, .ret = sig.ret });
            }
            for (Ast.rangeSlice(tree, n.rhs)) |a| try walkCalls(gpa, frozen, a, out);
        },
        .var_decl => try walkCalls(gpa, frozen, n.lhs, out),
        .assign => {
            try walkCalls(gpa, frozen, n.lhs, out);
            try walkCalls(gpa, frozen, n.rhs, out);
        },
        .return_stmt => if (n.lhs != Ast.none) try walkCalls(gpa, frozen, n.lhs, out),
        .expr_stmt => try walkCalls(gpa, frozen, n.lhs, out),
        .block => for (Ast.rangeSlice(tree, n.lhs)) |s| try walkCalls(gpa, frozen, s, out),
        .param => try walkCalls(gpa, frozen, n.lhs, out),
        .fn_decl => {
            const proto = Ast.protoAt(tree, n.lhs);
            for (proto.params) |p| try walkCalls(gpa, frozen, p, out);
            if (proto.ret_type != Ast.none) try walkCalls(gpa, frozen, proto.ret_type, out);
            try walkCalls(gpa, frozen, n.rhs, out);
        },
        .while_stmt => {
            try walkCalls(gpa, frozen, n.lhs, out);
            try walkCalls(gpa, frozen, n.rhs, out);
        },
        .if_stmt => {
            try walkCalls(gpa, frozen, n.lhs, out);
            const head = Ast.ifHeaderAt(tree, n.rhs);
            try walkCalls(gpa, frozen, head.then_block, out);
            if (head.else_node != Ast.none) try walkCalls(gpa, frozen, head.else_node, out);
        },
        .loop_expr => try walkCalls(gpa, frozen, n.lhs, out),
        .for_stmt => {
            const head = Ast.forHeaderAt(tree, n.rhs);
            try walkCalls(gpa, frozen, head.lo, out);
            try walkCalls(gpa, frozen, head.hi, out);
            try walkCalls(gpa, frozen, n.lhs, out);
        },
        .break_stmt => if (n.lhs != Ast.none) try walkCalls(gpa, frozen, n.lhs, out),
        .continue_stmt => {},
        .labeled => try walkCalls(gpa, frozen, n.lhs, out),
        // M9: order must match Fingerprint.walk (type-name lhs + inits; field recv).
        .struct_init => {
            try walkCalls(gpa, frozen, n.lhs, out);
            for (Ast.rangeSlice(tree, n.rhs)) |fi| try walkCalls(gpa, frozen, fi, out);
        },
        .field_init => try walkCalls(gpa, frozen, n.lhs, out),
        .field_access => try walkCalls(gpa, frozen, n.lhs, out),
        .struct_decl => {},
        // M10: order must MIRROR Fingerprint.walk (the callee_sigs sequence must
        // line up). Calls hide in payload exprs and arm bodies; patterns and
        // decls carry no calls.
        .enum_decl, .enum_variant_unit, .enum_variant_tuple, .enum_variant_struct => {},
        .enum_init_unit, .pattern_variant, .pattern_wildcard, .pattern_binding, .pattern_literal, .pattern_or => {},
        .enum_init_tuple => for (Ast.rangeSlice(tree, n.rhs)) |a| try walkCalls(gpa, frozen, a, out),
        .enum_init_struct => for (Ast.rangeSlice(tree, n.rhs)) |fi| try walkCalls(gpa, frozen, fi, out),
        .match_expr => {
            try walkCalls(gpa, frozen, n.lhs, out); // scrutinee
            for (Ast.rangeSlice(tree, n.rhs)) |arm| try walkCalls(gpa, frozen, arm, out);
        },
        // guard before body — must mirror Fingerprint.walk's order so the
        // positional callee_sigs sequence lines up.
        .match_arm => {
            const h = Ast.armHeaderAt(tree, n.rhs);
            if (h.guard != Ast.none) try walkCalls(gpa, frozen, h.guard, out);
            try walkCalls(gpa, frozen, h.body, out);
        },
        // Top-level decls; never reached inside a fn-body walk.
        .program, .import_decl => {},
    }
}

/// Collect the types this fn touches (the node types under its subtree, in walk
/// order), each as a `TouchedType` carrying — for a struct — an index-free layout
/// descriptor so a struct field-layout edit flips every using fn's hash (M9). The
/// fold is what makes the M5 "touched type layouts" hook REAL. Caller frees each
/// `layout` slice (see `freeTouched`).
pub fn walkTouched(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, out: *std.ArrayList(Fingerprint.TouchedType)) error{OutOfMemory}!void {
    return walkTouchedSig(gpa, frozen, idx, null, out);
}

/// `walkTouched` with the OWNING fn's signature threaded in (so the fn_decl case
/// folds the ABI-correct param/return types). `fn_sig` is the fn's typecheck Sig
/// (`frozen.sigs[idx]` single-file / `gf.sigs[gid]` graph) when known; null
/// elsewhere. The sig's `params`/`ret` carry the GLOBAL struct/enum ids the
/// typechecker resolved — including a CROSS-MODULE qualified type-ref `b: rect.Rect`
/// (a `field_access` in type position) which `typeRefToType`'s bare-name scan would
/// otherwise mis-resolve to the FIRST same-named type in the program-wide layout
/// table (the cross-module M9 / TOP-RISK-#1 hole). Folding the sig types makes a
/// pub-type LAYOUT edit reach EXACTLY the importers that name it. [design 10]
pub fn walkTouchedSig(gpa: std.mem.Allocator, frozen: anytype, idx: Ast.Index, fn_sig: ?Fingerprint.Sig, out: *std.ArrayList(Fingerprint.TouchedType)) error{OutOfMemory}!void {
    if (idx == Ast.none) return;
    const tree = frozen.tree;
    const n = tree.nodes[idx];
    if (idx < frozen.node_types.len) try appendTouched(gpa, frozen, frozen.node_types[idx], out);
    switch (n.tag) {
        .unary, .var_decl, .expr_stmt, .param => try walkTouched(gpa, frozen, n.lhs, out),
        .binary, .assign, .while_stmt => {
            try walkTouched(gpa, frozen, n.lhs, out);
            try walkTouched(gpa, frozen, n.rhs, out);
        },
        .call => {
            try walkTouched(gpa, frozen, n.lhs, out);
            for (Ast.rangeSlice(tree, n.rhs)) |a| try walkTouched(gpa, frozen, a, out);
        },
        .return_stmt => if (n.lhs != Ast.none) try walkTouched(gpa, frozen, n.lhs, out),
        .block => for (Ast.rangeSlice(tree, n.lhs)) |s| try walkTouched(gpa, frozen, s, out),
        .fn_decl => {
            const proto = Ast.protoAt(tree, n.lhs);
            // Fold the DECLARED type of each param and the return type. Typecheck
            // pass A never records node_types on the param/ret type-ref nodes (they
            // stay .invalid), so the generic walk below would miss a struct touched
            // ONLY via a param/return type. Resolve the type-ref by name against the
            // struct table and fold its layout — closing the param/ret-only
            // stale-hit hole (M9 cache soundness: fold EVERY struct a fn touches,
            // including params and return, so an ABI-boundary edit recompiles it).
            for (proto.params, 0..) |p, i| {
                const pty_node = tree.nodes[p].lhs;
                if (pty_node != Ast.none) {
                    // Prefer the typecheck-resolved sig type (carries the right
                    // GLOBAL id, incl. a cross-module qualified `mod.Type`); fall
                    // back to the bare-name re-resolution when no sig is threaded.
                    const pty = if (fn_sig) |s| (if (i < s.params.len) s.params[i] else typeRefToType(frozen, pty_node)) else typeRefToType(frozen, pty_node);
                    try appendTouched(gpa, frozen, pty, out);
                }
                try walkTouched(gpa, frozen, p, out);
            }
            if (proto.ret_type != Ast.none) {
                const rty = if (fn_sig) |s| s.ret else typeRefToType(frozen, proto.ret_type);
                try appendTouched(gpa, frozen, rty, out);
                try walkTouched(gpa, frozen, proto.ret_type, out);
            }
            try walkTouched(gpa, frozen, n.rhs, out);
        },
        .if_stmt => {
            try walkTouched(gpa, frozen, n.lhs, out);
            const head = Ast.ifHeaderAt(tree, n.rhs);
            try walkTouched(gpa, frozen, head.then_block, out);
            if (head.else_node != Ast.none) try walkTouched(gpa, frozen, head.else_node, out);
        },
        .loop_expr => try walkTouched(gpa, frozen, n.lhs, out),
        .for_stmt => {
            const head = Ast.forHeaderAt(tree, n.rhs);
            try walkTouched(gpa, frozen, head.lo, out);
            try walkTouched(gpa, frozen, head.hi, out);
            try walkTouched(gpa, frozen, n.lhs, out);
        },
        .break_stmt => if (n.lhs != Ast.none) try walkTouched(gpa, frozen, n.lhs, out),
        .labeled => try walkTouched(gpa, frozen, n.lhs, out),
        // M9: a struct construction touches its type (recorded on the node above)
        // and the field-init values; a field access touches the receiver.
        .struct_init => {
            try walkTouched(gpa, frozen, n.lhs, out);
            for (Ast.rangeSlice(tree, n.rhs)) |fi| try walkTouched(gpa, frozen, fi, out);
        },
        .field_init => try walkTouched(gpa, frozen, n.lhs, out),
        .field_access => try walkTouched(gpa, frozen, n.lhs, out),
        // M10: a variant construction touches its enum (folded on the node above)
        // and the payload values; a match touches the scrutinee + each arm body.
        .enum_init_tuple => for (Ast.rangeSlice(tree, n.rhs)) |a| try walkTouched(gpa, frozen, a, out),
        .enum_init_struct => for (Ast.rangeSlice(tree, n.rhs)) |fi| try walkTouched(gpa, frozen, fi, out),
        .match_expr => {
            try walkTouched(gpa, frozen, n.lhs, out);
            for (Ast.rangeSlice(tree, n.rhs)) |arm| try walkTouched(gpa, frozen, arm, out);
        },
        .match_arm => {
            const h = Ast.armHeaderAt(tree, n.rhs);
            if (h.guard != Ast.none) try walkTouched(gpa, frozen, h.guard, out);
            try walkTouched(gpa, frozen, h.body, out);
        },
        else => {},
    }
}

/// Resolve a DECLARED type-ref node (a param/return type annotation) to a Type by
/// name, using the struct table. Builtins map to their scalar/str kinds; a struct
/// name resolves to its `struct_id`; anything else (incl. `()`) → unit. Mirrors
/// `lower`'s type-ref resolution so the fingerprint folds the SAME layout codegen
/// will use, independent of node_types (which Typecheck never sets here).
pub fn typeRefToType(frozen: anytype, type_node: Ast.Index) Typecheck.Type {
    const n = frozen.tree.nodes[type_node];
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

/// Append a `TouchedType` for `ty`, building the struct layout descriptor bytes.
pub fn appendTouched(gpa: std.mem.Allocator, frozen: anytype, ty: Typecheck.Type, out: *std.ArrayList(Fingerprint.TouchedType)) !void {
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
    try out.append(gpa, .{ .kind = ty.kind });
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
/// flips the bytes, recompiling every using fn (M10 cache soundness).
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
            // A tuple variant has no field names; fold its payload types/offsets.
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

/// Free the layout slices owned by a `walkTouched` result.
pub fn freeTouched(gpa: std.mem.Allocator, items: []const Fingerprint.TouchedType) void {
    for (items) |t| if (t.layout.len > 0) gpa.free(t.layout);
}
