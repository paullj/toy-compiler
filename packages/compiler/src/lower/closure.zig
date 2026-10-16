//! `lower/closure`: function values. A lambda or local fn unit's own lowering (its env
//! prologue), and in any body the values that build a `{code, env}` pair: a named fn, a
//! lambda with its env, a bound method, plus calls of a function value (straight to its
//! code in a spec unit) and `==` on two of them. Split out of `lower.zig` and reading its
//! `Builder` like `derive_emit` and `adapter_emit` do.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const L = @import("../lower.zig");
const places = @import("places.zig");
const Typecheck = @import("../types.zig");
const Closure = @import("../symbols/Closure.zig");
const Spec = @import("../symbols/Spec.zig");
const Link = @import("../link/Link.zig");
const Ir = @import("../ir/Ir.zig");
const Abi = @import("../codegen/abi/Abi.zig");
const Diagnostic = @import("../diagnostics/Diagnostic.zig").Diagnostic;
const Builder = L.Builder;
const Inputs = L.Inputs;

/// Lower one lambda unit: the lambda's (or local fn's) body under its owner's tables, as a
/// function of `(env, params..)`. Its params bind the resolver slots the owner's body gave
/// them, and its return type is the unit's (a lambda's missing `-> R` is inferred, not
/// unit, so `returnType`/`paramType` do not apply).
pub fn lowerLambda(
    gpa: std.mem.Allocator,
    in: Inputs,
    unit_i: u32,
    sym: Link.SymName,
    out_diags: *std.ArrayList(Diagnostic),
) error{OutOfMemory}!Ir.Function {
    std.debug.assert(in.nodes.callees.len == in.nodes.types.len);
    const unit = in.closure.lambda_units[unit_i];
    var b: Builder = .{ .gpa = gpa, .in = in, .ret_type = unit.sig.ret, .diags = out_diags };
    errdefer b.deinit();

    var params: std.ArrayList(Ir.SlotId) = .empty;
    errdefer params.deinit(gpa);
    const env_slot = try b.addSlot(Typecheck.Type.rawptr);
    try params.append(gpa, env_slot);
    for (unit.sig.params, unit.param_slots) |pty, li| {
        const sid = try b.addSlot(pty);
        try params.append(gpa, sid);
        try b.homes.put(gpa, li, .{ .slot = sid });
    }
    const lam = in.tree.nodes[unit.node.int()];
    const blocks = try L.openBodyBlocks(&b, unit.sig.ret);
    try bindCaptures(&b, unit, env_slot);
    if (unit.self_slot) |ss| try bindSelf(&b, unit, ss, env_slot);
    try places.boxParams(&b, unit.param_slots, params.items[1..], lam.main_token);
    try places.noteMutArgRoots(&b, lam.rhs);
    try places.noteUnboxedRoots(&b);
    try L.lowerBodyInto(&b, lam.rhs, unit.sig.ret, blocks.exit);
    return try L.finishFn(&b, gpa, sym, &params, blocks.entry, blocks.exit);
}

/// A lambda unit's prologue: copy each capture out of the env into a local slot of its
/// own and bind the owner's slot to it, so the body (and any lambda nested in it, which
/// copies from these slots in turn) reads captures like any local. A capture held in a cell
/// lives at the cell pointer instead, shared with the owner; one held by address (a `mut`
/// param, or a local this lowering keeps in the owner's frame) at that address.
fn bindCaptures(b: *Builder, unit: Closure.LambdaUnit, env_slot: Ir.SlotId) error{OutOfMemory}!void {
    if (unit.env_size == 0) return;
    const tok = b.in.tree.nodes[unit.node.int()].main_token;
    const env_sa = try b.emit(.{ .slot_addr = env_slot }, Typecheck.Type.int);
    const env = try b.emit(.{ .load = .{ .addr = env_sa, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.rawptr);
    for (unit.env) |es| switch (b.in.closure.storage.hold(es.slot)) {
        // A non-self `mut` param's stores are staged in its owner, so they are here too.
        .param_addr => try bindAddressCapture(b, es, env, !isSelfParam(b, unit.fn_node, es.slot)),
        .frame_addr => try bindAddressCapture(b, es, env, false),
        .cell => {
            const sid = try b.addSlot(Typecheck.Type.rawptr);
            try b.homes.put(b.gpa, es.slot, .{ .cell = sid });
            const src = try b.emit(.{ .field_addr = .{ .base = env, .off = es.off, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.int);
            const cell = try b.emit(.{ .load = .{ .addr = src, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.rawptr);
            const dst = try b.emit(.{ .slot_addr = sid }, Typecheck.Type.int);
            _ = try b.emit(.{ .store = .{ .addr = dst, .val = cell, .ty = Typecheck.Type.rawptr } }, null);
        },
        .copy => {
            const sid = try b.addSlot(es.ty);
            try b.homes.put(b.gpa, es.slot, .{ .slot = sid });
            const src = try b.emit(.{ .field_addr = .{ .base = env, .off = es.off, .ty = es.ty } }, Typecheck.Type.int);
            const dst = try b.emit(.{ .slot_addr = sid }, Typecheck.Type.int);
            try copyCapture(b, dst, src, es.ty, tok);
        },
    };
}

/// Bind the address env slot `es` holds as an `addr` home, the shape a `mut` param has, so
/// every use loads through it.
fn bindAddressCapture(b: *Builder, es: Closure.EnvSlot, env: Ir.ValueId, staged: bool) error{OutOfMemory}!void {
    const sid = try b.addSlot(Typecheck.Type.int);
    try b.homes.put(b.gpa, es.slot, .{ .addr = sid });
    if (staged) b.staged_places = true;
    const src = try b.emit(.{ .field_addr = .{ .base = env, .off = es.off, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
    const addr = try b.emit(.{ .load = .{ .addr = src, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
    const dst = try b.emit(.{ .slot_addr = sid }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = dst, .val = addr, .ty = Typecheck.Type.int } }, null);
}

/// Whether resolver slot `slot` is the `self` receiver of the fn at `fn_node`.
fn isSelfParam(b: *const Builder, fn_node: Ast.Index, slot: u32) bool {
    if (slot != 0) return false;
    const proto = Ast.protoAt(b.in.tree, b.in.tree.nodes[fn_node.int()].lhs.int());
    if (proto.params.len == 0) return false;
    return Ast.isSelfParam(b.in.tree, b.in.tokens, b.in.source, proto.params[0]);
}

/// A local fn's prologue: its own name reads as the pair of its code and the env it was
/// called with, which is exactly the value its caller holds, so recursion needs no capture.
fn bindSelf(b: *Builder, unit: Closure.LambdaUnit, self_slot: u32, env_slot: Ir.SlotId) error{OutOfMemory}!void {
    const env_sa = try b.emit(.{ .slot_addr = env_slot }, Typecheck.Type.int);
    const env = try b.emit(.{ .load = .{ .addr = env_sa, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.rawptr);
    const pair = try emitFnPair(b, unit.name, b.in.nodes.typeOf(unit.node), env);
    try b.homes.put(b.gpa, self_slot, .{ .slot = pair.slot });
}

/// `fn name(..) { .. }` in a block: make its closure like a lambda's, and let the pair's
/// slot be the name's home. Nothing assigns a local fn, so it is never boxed.
pub fn lowerLocalFn(b: *Builder, stmt_idx: Ast.Index) error{OutOfMemory}!void {
    const v = try lowerLambdaValue(b, stmt_idx, b.in.nodes.typeOf(stmt_idx));
    const res = b.in.resolutions[(stmt_idx).int()];
    if (v != .slot or res != .local) return;
    std.debug.assert(!b.isBoxedLocal(res.local));
    try b.homes.put(b.gpa, res.local, .{ .slot = v.slot });
}

/// A call of a function-typed value: evaluate the callee to its `{code, env}` pair, then
/// the args left to right, and `blr` the code with the env as the hidden first argument.
pub fn lowerValueCall(b: *Builder, node_idx: Ast.Index, n: Ast.Node) error{OutOfMemory}!Ir.Operand {
    if (specTarget(b, n.lhs)) |t| return lowerSpecTargetCall(b, node_idx, n, t);
    const int_ty = Typecheck.Type.int;
    const pair = try L.operandPtr(b, try L.lowerExpr(b, n.lhs));
    if (pair == Ir.none_value) {
        try b.note(n.main_token, "called function value is not a slot in lower");
        return .none;
    }
    const code = try b.emit(.{ .load = .{ .addr = pair, .ty = int_ty } }, int_ty);
    const env_addr = try b.emit(.{ .field_addr = .{ .base = pair, .off = 8, .ty = Typecheck.Type.rawptr } }, int_ty);
    const env = try b.emit(.{ .load = .{ .addr = env_addr, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.rawptr);
    const arg_nodes = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    const args = try b.gpa.alloc(Ir.Operand, arg_nodes.len + 1);
    errdefer b.gpa.free(args);
    args[0] = .{ .value = env };
    for (arg_nodes, 1..) |arg, i| args[i] = try L.lowerExpr(b, arg);
    return try L.emitCall(b, .{ .indirect = code }, args, b.in.nodes.typeOf(node_idx), n.main_token);
}

/// The target of a spec unit's param when `callee` names one directly. Resolver slots
/// hand params out first, so param `p` is slot `p`; a param is never assigned in a body
/// that specializes it, so the slot still holds the caller's argument.
fn specTarget(b: *const Builder, callee: Ast.Index) ?Spec.Target {
    if (b.in.closure.spec_targets.len == 0) return null;
    if (b.in.tree.nodes[callee.int()].tag != .identifier) return null;
    const slot = switch (b.in.resolutions[callee.int()]) {
        .local => |s| s,
        else => return null,
    };
    for (b.in.closure.spec_targets) |t| if (t.param == slot) return t;
    return null;
}

/// A spec unit's call of param `t.param`: a direct call of the code the caller passed. A
/// named fn takes no env; a lambda takes the env of the pair the param holds.
fn lowerSpecTargetCall(b: *Builder, node_idx: Ast.Index, n: Ast.Node, t: Spec.Target) error{OutOfMemory}!Ir.Operand {
    const arg_nodes = Ast.rangeSlice(b.in.tree, (n.rhs).int());
    const lead: usize = @intFromBool(t.kind == .lambda);
    const args = try b.gpa.alloc(Ir.Operand, arg_nodes.len + lead);
    errdefer b.gpa.free(args);
    if (t.kind == .lambda) {
        const pair = try L.operandPtr(b, try L.lowerExpr(b, n.lhs));
        if (pair == Ir.none_value) {
            try b.note(n.main_token, "called function value is not a slot in lower");
            b.gpa.free(args);
            return .none;
        }
        const env_addr = try b.emit(.{ .field_addr = .{ .base = pair, .off = 8, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.int);
        args[0] = .{ .value = try b.emit(.{ .load = .{ .addr = env_addr, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.rawptr) };
    }
    for (arg_nodes, lead..) |arg, i| args[i] = try L.lowerExpr(b, arg);
    return try L.emitCall(b, .{ .direct = t.sym }, args, b.in.nodes.typeOf(node_idx), n.main_token);
}

/// A named fn in value position: the pair `{its thunk's address, null env}` in a fresh
/// slot. The thunk gives every function value the same `(env, params..)` call shape.
pub fn lowerFnValue(b: *Builder, node_idx: Ast.Index, gid: u32, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const n = b.in.tree.nodes[(node_idx).int()];
    const ai = Closure.findAdapter(b.in.closure.adapters, .fn_value, gid) orelse {
        try b.note(n.main_token, "function value has no thunk in lower");
        return .none;
    };
    return try emitFnPair(b, b.in.closure.adapters[ai].name, ty, null);
}

/// A lambda in value position: the pair `{its unit's address, env}`. A capturing lambda
/// allocates its env cell first, then stores each capture in at the table's offset: a
/// copy of its current value, or for a boxed local the pointer to its cell. A
/// capture-free one carries a null env.
pub fn lowerLambdaValue(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const n = b.in.tree.nodes[(node_idx).int()];
    const owner = b.in.closure.owner orelse {
        try b.note(n.main_token, "lambda outside a closure owner in lower");
        return .none;
    };
    const ui = Closure.findUnit(b.in.closure.lambda_units, owner, node_idx) orelse {
        try b.note(n.main_token, "lambda has no unit in lower");
        return .none;
    };
    const unit = b.in.closure.lambda_units[ui];
    if (unit.env_size == 0) return try emitFnPair(b, unit.name, ty, null);
    const env = if (b.in.closure.frame_envs and unit.frame_env != null)
        try b.emit(.{ .slot_addr = try b.addSlot(Typecheck.Type.structT(unit.frame_env.?)) }, Typecheck.Type.int)
    else
        try L.emitGcAlloc(b, unit.env_size);
    for (unit.env) |es| {
        const home = b.homes.get(es.slot) orelse {
            try b.note(n.main_token, "captured local has no slot in lower");
            continue;
        };
        switch (b.in.closure.storage.hold(es.slot)) {
            .param_addr, .frame_addr => try storeEnvAddress(b, env, es, home),
            .cell => {
                if (home != .cell) {
                    try b.note(n.main_token, "boxed capture has no cell in lower");
                    continue;
                }
                const dst = try b.emit(.{ .field_addr = .{ .base = env, .off = es.off, .ty = Typecheck.Type.rawptr } }, Typecheck.Type.int);
                _ = try b.emit(.{ .store = .{ .addr = dst, .val = try places.homeAddr(b, home), .ty = Typecheck.Type.rawptr } }, null);
            },
            .copy => {
                // Only an address word may come from an address home: a copy would read the
                // caller's storage once, and T0052 keeps an escaping closure from holding
                // the address.
                std.debug.assert(home != .addr);
                const dst = try b.emit(.{ .field_addr = .{ .base = env, .off = es.off, .ty = es.ty } }, Typecheck.Type.int);
                try copyCapture(b, dst, try places.homeAddr(b, home), es.ty, n.main_token);
            },
        }
    }
    return try emitFnPair(b, unit.name, ty, env);
}

/// Store the address of the local at `home` into env slot `es`, typed `int` so the word is
/// never a root.
fn storeEnvAddress(b: *Builder, env: Ir.ValueId, es: Closure.EnvSlot, home: places.LocalHome) error{OutOfMemory}!void {
    const dst = try b.emit(.{ .field_addr = .{ .base = env, .off = es.off, .ty = Typecheck.Type.int } }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = dst, .val = try places.homeAddr(b, home), .ty = Typecheck.Type.int } }, null);
}

/// Copy one captured value of type `ty` from `src` to `dst` (both addresses): a scalar or
/// box word by load/store, an aggregate by `copy`.
pub fn copyCapture(b: *Builder, dst: Ir.ValueId, src: Ir.ValueId, ty: Typecheck.Type, tok: u32) error{OutOfMemory}!void {
    const int_ty = Typecheck.Type.int;
    switch (L.passKind(b, ty)) {
        .box => {
            const v = try b.emit(.{ .load = .{ .addr = src, .ty = int_ty } }, L.box_repr_ty);
            _ = try b.emit(.{ .store = .{ .addr = dst, .val = v, .ty = int_ty } }, null);
        },
        .int, .bool, .float, .rawptr => {
            const v = try b.emit(.{ .load = .{ .addr = src, .ty = ty } }, ty);
            _ = try b.emit(.{ .store = .{ .addr = dst, .val = v, .ty = ty } }, null);
        },
        .str, .@"struct", .@"enum" => _ = try b.emit(.{ .copy = .{ .dst = dst, .src = src, .ty = ty } }, null),
        .unit => {},
        .never, .invalid => try b.note(tok, "captured type unsupported in lower"),
    }
}

/// The function value `{address of code_sym, env}` in a fresh slot of pair type `ty`; a
/// null `env` stores a null env word.
fn emitFnPair(b: *Builder, code_sym: []const u8, ty: Typecheck.Type, env: ?Ir.ValueId) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const slot = try b.addSlot(ty);
    const base = try b.emit(.{ .slot_addr = slot }, int_ty);
    const code = try b.emit(.{ .fn_addr = .{ .kind = .user_fn, .name = code_sym } }, int_ty);
    _ = try b.emit(.{ .store = .{ .addr = base, .val = code, .ty = int_ty } }, null);
    const env_addr = try b.emit(.{ .field_addr = .{ .base = base, .off = 8, .ty = Typecheck.Type.rawptr } }, int_ty);
    const env_v = env orelse try b.emit(.{ .iconst = 0 }, Typecheck.Type.rawptr);
    _ = try b.emit(.{ .store = .{ .addr = env_addr, .val = env_v, .ty = Typecheck.Type.rawptr } }, null);
    return .{ .slot = slot };
}

/// The fn a value-position name refers to, when the checker typed it as a fn value.
pub fn fnValueRef(b: *const Builder, node_idx: Ast.Index) ?u32 {
    if (b.in.resolutions[(node_idx).int()] != .func) return null;
    return switch (b.in.nodes.callee(node_idx).view()) {
        .fn_ref => |gid| gid,
        else => null,
    };
}

/// Whether `field_access` `node_idx` names a field of its struct base.
pub fn namesField(b: *const Builder, node_idx: Ast.Index) bool {
    const n = b.in.tree.nodes[(node_idx).int()];
    if (n.tag != .field_access) return true;
    const base_ty = b.in.nodes.typeOf(n.lhs);
    if (base_ty.kind != .@"struct" or base_ty.struct_id >= b.in.layouts.len) return false;
    const fname = b.in.tokens[n.main_token].text(b.in.source);
    for (b.in.layouts[base_ty.struct_id].field_names) |dn| if (std.mem.eql(u8, dn, fname)) return true;
    return false;
}

/// The method a value-position `recv.m` binds, when the checker typed it as one.
pub fn boundMethodRef(b: *const Builder, node_idx: Ast.Index) ?u32 {
    return switch (b.in.nodes.callee(node_idx).view()) {
        .bound_method => |mi| mi,
        else => null,
    };
}

/// A bound method `recv.m` in value position: the pair `{its adapter's address, env}`.
/// The env holds the receiver: the cell of a boxed local receiver, so the closure and its
/// owner see each other's writes, else a copy of the receiver value, evaluated once here.
pub fn lowerBoundValue(b: *Builder, node_idx: Ast.Index, mi: u32, ty: Typecheck.Type) error{OutOfMemory}!Ir.Operand {
    const n = b.in.tree.nodes[(node_idx).int()];
    const recv_local = Closure.boundRecvLocal(b.in.tree, b.in.resolutions, node_idx);
    const kind = Closure.boundKind(recv_local, b.in.closure.storage);
    const ai = Closure.findAdapter(b.in.closure.adapters, kind, mi) orelse {
        try b.note(n.main_token, "bound method has no adapter in lower");
        return .none;
    };
    const recv_ty = b.in.nodes.typeOf(n.lhs);
    switch (kind) {
        .bound_box => {
            const home = try places.bindingHome(b, n.lhs, recv_ty);
            if (home != .cell) {
                try b.note(n.main_token, "bound receiver has no cell in lower");
                return .none;
            }
            const cell = try places.homeAddr(b, home);
            const env = try L.emitGcAlloc(b, 8);
            _ = try b.emit(.{ .store = .{ .addr = env, .val = cell, .ty = Typecheck.Type.rawptr } }, null);
            return try emitFnPair(b, b.in.closure.adapters[ai].name, ty, env);
        },
        .bound_ref => {
            const home = try places.bindingHome(b, n.lhs, recv_ty);
            if (home != .addr) {
                try b.note(n.main_token, "bound mut receiver has no place in lower");
                return .none;
            }
            const env = try L.emitGcAlloc(b, 8);
            _ = try b.emit(.{ .store = .{ .addr = env, .val = try places.homeAddr(b, home), .ty = Typecheck.Type.int } }, null);
            return try emitFnPair(b, b.in.closure.adapters[ai].name, ty, env);
        },
        .bound_copy => {
            const env = try L.emitGcAlloc(b, @max(8, Abi.typeSize(recv_ty, b.in.layouts, b.in.enum_layouts)));
            try L.lowerExprInto(b, n.lhs, env, recv_ty);
            return try emitFnPair(b, b.in.closure.adapters[ai].name, ty, env);
        },
        .fn_value => unreachable,
    }
}

/// `==` on function values is identity: both the code and the env words match.
pub fn lowerFnIdentityEq(b: *Builder, lhs_node: Ast.Index, rhs_node: Ast.Index, negate: bool) error{OutOfMemory}!Ir.Operand {
    const int_ty = Typecheck.Type.int;
    const bool_ty = Typecheck.Type.bool;
    const lp = try L.operandPtr(b, try L.lowerExpr(b, lhs_node));
    const rp = try L.operandPtr(b, try L.lowerExpr(b, rhs_node));
    if (lp == Ir.none_value or rp == Ir.none_value) {
        try b.note(b.in.tree.nodes[(lhs_node).int()].main_token, "function value operand is not a slot in lower");
        return .{ .value = try b.emit(.{ .bconst = false }, bool_ty) };
    }
    const lc = try b.emit(.{ .load = .{ .addr = lp, .ty = int_ty } }, int_ty);
    const rc = try b.emit(.{ .load = .{ .addr = rp, .ty = int_ty } }, int_ty);
    const le_addr = try b.emit(.{ .field_addr = .{ .base = lp, .off = 8, .ty = int_ty } }, int_ty);
    const le = try b.emit(.{ .load = .{ .addr = le_addr, .ty = int_ty } }, int_ty);
    const re_addr = try b.emit(.{ .field_addr = .{ .base = rp, .off = 8, .ty = int_ty } }, int_ty);
    const re = try b.emit(.{ .load = .{ .addr = re_addr, .ty = int_ty } }, int_ty);
    // Both words at once, branch-free: (lc ^ rc) | (le ^ re) is zero iff identical.
    const dc = try b.emit(.{ .bxor = .{ .lhs = lc, .rhs = rc } }, int_ty);
    const de = try b.emit(.{ .bxor = .{ .lhs = le, .rhs = re } }, int_ty);
    const any = try b.emit(.{ .bor = .{ .lhs = dc, .rhs = de } }, int_ty);
    const zero = try b.emit(.{ .iconst = 0 }, int_ty);
    const cc: Ir.Cond = if (negate) .ne else .eq;
    return .{ .value = try b.emit(.{ .icmp = .{ .cc = cc, .lhs = any, .rhs = zero } }, bool_ty) };
}
