//! `lower/places`: where a local lives and how a place is addressed. Each local's home
//! (its own slot, a GC cell, or an address), binding and boxing locals, the address of a
//! local-rooted place, and the copy-on-read and staged-store rules that keep a store
//! through an address from aliasing a read. Split out of `lower.zig` and reading its
//! `Builder` like `derive_emit` and `adapter_emit` do.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const L = @import("../lower.zig");
const Typecheck = @import("../types.zig");
const Modes = @import("../symbols/Modes.zig");
const Callee = @import("../symbols/Callee.zig").Callee;
const symbols_res = @import("../symbols/Resolution.zig");
const Walk = @import("../ast/Walk.zig");
const Ir = @import("../ir/Ir.zig");
const Abi = @import("../codegen/abi/Abi.zig");
const closure = @import("closure.zig");
const Builder = L.Builder;

/// Where a resolver local lives in one lowering. `slot` is the local's own slot. `cell` and
/// `addr` are ADDRESS slots, holding a pointer to the local's storage, which `homeAddr`
/// loads through so every use reaches it. `cell` is a boxed local's: typed `rawptr`, so the
/// precise root map keeps its GC cell alive. `addr` is typed `int`: a `mut` param's (`mut
/// self` included), so the scalar 1-GPR ABI carries the caller's place address, and in a
/// lambda unit each capture its env passes by address (a captured `mut` param, or a local
/// its owner keeps in the frame). That address may point into a frame or into a cell's
/// interior, so it must never be a root.
pub const LocalHome = union(enum) {
    slot: Ir.SlotId,
    cell: Ir.SlotId,
    addr: Ir.SlotId,
};

/// The address a local's value is read from or stored to. For its own slot that is
/// `slot_addr(slot)`. An address slot holds a POINTER to the storage instead, so the
/// address is that pointer, a `load` of `slot_addr(slot)`: typed `int` for an `addr`
/// (which must not read as a root), `rawptr` for a `cell` (a GC cell, so the loaded pointer
/// is a root too).
pub fn homeAddr(b: *Builder, home: LocalHome) error{OutOfMemory}!Ir.ValueId {
    const int_ty = Typecheck.Type.int;
    switch (home) {
        .slot => |s| return try b.emit(.{ .slot_addr = s }, int_ty),
        .cell, .addr => |s| {
            const ptr_ty = if (home == .cell) Typecheck.Type.rawptr else int_ty;
            const sa = try b.emit(.{ .slot_addr = s }, int_ty);
            return try b.emit(.{ .load = .{ .addr = sa, .ty = ptr_ty } }, ptr_ty);
        },
    }
}

/// The ptr VALUE of a LOCAL-ROOTED place (`p`, `p.x`, `p.a.b`): a slot_addr at the
/// root + a field_addr per `.field` hop, resolving each field name → byte offset
/// from the layout. Returns `none_value` for a non-local-rooted place.
pub fn lowerPlaceAddr(b: *Builder, node_idx: Ast.Index) error{OutOfMemory}!Ir.ValueId {
    const n = b.in.tree.nodes[(node_idx).int()];
    switch (n.tag) {
        .identifier => {
            if (b.in.resolutions[(node_idx).int()] != .local) return Ir.none_value;
            return try homeAddr(b, try bindingHome(b, node_idx, b.in.nodes.typeOf(node_idx)));
        },
        .field_access, .tuple_field => {
            const base_addr = try lowerPlaceAddr(b, n.lhs);
            if (base_addr == Ir.none_value) return Ir.none_value;
            const base_ty = b.in.nodes.typeOf(n.lhs);
            if (base_ty.kind != .@"struct") return Ir.none_value;
            const layout = b.in.layouts[base_ty.struct_id];
            const fname = b.in.tokens[n.main_token].text(b.in.source);
            for (layout.field_names, 0..) |dn, j| {
                if (std.mem.eql(u8, dn, fname)) {
                    return try b.emit(.{ .field_addr = .{ .base = base_addr, .off = layout.offsets[j], .ty = layout.field_types[j] } }, Typecheck.Type.int);
                }
            }
            // A member that is no field (a bound method) has no address; the checker
            // rejects such a place (T0019), so this is unreachable on a checked program.
            return Ir.none_value;
        },
        // A `.field` off an index element (`ps[i].x`, `b.items[i].x`): the element is
        // an rvalue with no named place, so materialize it into a temp and return that
        // temp's ptr — the `.field_access` recursion above then computes `field_addr`
        // off it. A store through this ptr would write the copy, not the vector, so the
        // checker's `storeRoot` gate rejects assigning through it (T0070); only
        // the read base (`= b.items[i].x`, `b.items[i].m()`) reaches here.
        .index => {
            const op = try L.lowerExpr(b, node_idx);
            return try L.operandPtr(b, op);
        },
        .type_app => {
            if (!L.appHeadIsValue(b, node_idx)) return Ir.none_value;
            const op = try L.lowerExpr(b, node_idx);
            return try L.operandPtr(b, op);
        },
        // Any other base that produces an aggregate RVALUE with no named place — a
        // call/method result, a `match`/`if` expression — is materialized into a temp
        // once so the field_addr chain reads from the copy. Without this, `f().field`
        // (e.g. the idiomatic `v.get(i).unwrap().field`) resolves to no address and
        // silently reads a stale register, or crashes when used in arithmetic. The
        // checker rejects a store through such a base (T0075), so this read-only
        // materialization can never alias a real place.
        else => switch (L.passKind(b, b.in.nodes.typeOf(node_idx))) {
            .str, .@"struct", .@"enum" => {
                const op = try L.lowerExpr(b, node_idx);
                return try L.operandPtr(b, op);
            },
            else => return Ir.none_value,
        },
    }
}

/// Put each local a call in `body` passes as a `mut` argument or a `mut self` receiver into
/// `copy_on_read_roots`.
pub fn noteMutArgRoots(b: *Builder, body: Ast.Index) error{OutOfMemory}!void {
    const Visitor = struct {
        b: *Builder,
        pub fn on(v: *@This(), ev: Walk.Event) error{OutOfMemory}!void {
            const c = switch (ev) {
                .callee => |c| c,
                else => return,
            };
            const in = v.b.in;
            const call = in.tree.nodes[c.call.int()];
            if (Modes.receiverMut(in.bodyView(), in.programView(), c.call)) if (Callee.memberNode(in.tree, call)) |mn|
                try v.note(in.tree.nodes[mn.int()].lhs);
            const mask = Modes.argMask(in.bodyView(), in.programView(), c.call);
            if (mask == 0) return;
            const args = Ast.rangeSlice(in.tree, call.rhs.int());
            for (args, 0..) |a, i| if (Modes.isMut(mask, i)) try v.note(a);
        }

        fn note(v: *@This(), place: Ast.Index) error{OutOfMemory}!void {
            if (symbols_res.placeRoot(v.b.in.tree, v.b.in.resolutions, place)) |li|
                try v.b.copy_on_read_roots.put(v.b.gpa, li, {});
        }
    };
    var v: Visitor = .{ .b = b };
    try Walk.walk(.{ .tree = b.in.tree, .tokens = b.in.tokens, .source = b.in.source }, body, &v);
}

/// A local kept in the owner's frame is read by slot like any plain one, but a closure a
/// later argument calls may write it, as through its box at -O0, so a by-value read copies
/// it on the spot.
pub fn noteUnboxedRoots(b: *Builder) error{OutOfMemory}!void {
    for (b.in.closure.storage.frame_addrs) |li| try b.copy_on_read_roots.put(b.gpa, li, {});
}

/// Whether a by-value read of the local `node_idx` names must copy it now (see
/// `copy_on_read_roots`).
pub fn readsCopyOnReadRoot(b: *const Builder, node_idx: Ast.Index) bool {
    if (b.copy_on_read_roots.count() == 0) return false;
    return switch (b.in.resolutions[node_idx.int()]) {
        .local => |li| b.copy_on_read_roots.contains(li),
        else => false,
    };
}

/// Whether an aggregate store of `value` into the place rooted at `place` builds the value
/// aside first, because `value` may read that storage after part of it is written. Two
/// `addr` homes may name one caller storage (see `staged_places`). Otherwise only a value
/// that writes the destination piece by piece can see it half written, and only when it
/// reads the root: by name, or, for a root a closure or a caller may hold by reference,
/// through code it runs. Staging every aggregate store would change the code of programs
/// none of this reaches.
pub fn stagesStore(b: *Builder, place: Ast.Index, value: Ast.Index) bool {
    const li = symbols_res.placeRoot(b.in.tree, b.in.resolutions, place) orelse return false;
    const home = b.homes.get(li) orelse return false;
    const hold = b.in.closure.storage.hold(li);
    // A store to a local kept in the owner's frame is unstaged here, as through its box at -O0.
    if (b.staged_places and home == .addr and hold != .frame_addr) return true;
    if (!L.producesInPlace(b, value)) return false;
    const shared = home != .slot or hold == .frame_addr;
    const reads = readsOf(b, value, li);
    return reads.names_root or (shared and reads.runs_code);
}

const Reads = struct { names_root: bool = false, runs_code: bool = false };

/// Whether `expr` names local `li` (in a lambda inside it too), and whether it may run code
/// other than its own: a call, an index, a `for` (`iter`/`next`), or an operator a protocol
/// witness implements.
fn readsOf(b: *const Builder, expr: Ast.Index, li: u32) Reads {
    const Visitor = struct {
        b: *const Builder,
        li: u32,
        r: Reads = .{},
        pub fn on(v: *@This(), ev: Walk.Event) void {
            const e = switch (ev) {
                .enter => |e| e,
                else => return,
            };
            const in = v.b.in;
            switch (e.tag) {
                .identifier => switch (in.resolutions[e.idx.int()]) {
                    .local => |l| if (l == v.li) {
                        v.r.names_root = true;
                    },
                    else => {},
                },
                .call, .index, .type_app, .for_in_stmt, .for_in2_stmt => v.r.runs_code = true,
                .binary => switch (in.nodes.typeOf(in.tree.nodes[e.idx.int()].lhs).kind) {
                    .@"struct", .@"enum" => v.r.runs_code = true,
                    else => {},
                },
                else => {},
            }
        }
    };
    var v: Visitor = .{ .b = b, .li = li };
    Walk.walk(.{ .tree = b.in.tree, .tokens = b.in.tokens, .source = b.in.source }, expr, &v) catch unreachable;
    return v.r;
}

/// Produce the aggregate `value` into a fresh temp, then copy it to `dst`, so `value`
/// never reads that storage half-written.
pub fn storeStaged(b: *Builder, value: Ast.Index, dst: Ir.ValueId, ty: Typecheck.Type) error{OutOfMemory}!void {
    const tmp = try L.aggregateValue(b, value, ty);
    if (b.termSet()) return;
    const src = try b.emit(.{ .slot_addr = tmp.slot }, Typecheck.Type.int);
    _ = try b.emit(.{ .copy = .{ .dst = dst, .src = src, .ty = ty } }, null);
}

/// Get (or lazily bind) the home of the local that `node_idx` resolves to, at a site that
/// does not know about boxes: a boxed local reaching it would be read or written as a
/// plain slot, so that is a loud lower note rather than a miscompile.
pub fn localHome(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!LocalHome {
    const res = b.in.resolutions[(node_idx).int()];
    if (res == .local and b.isBoxedLocal(res.local))
        try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "boxed local reached an unconverted slot access in lower");
    return bindingHome(b, node_idx, ty);
}

/// The boxed local a binding node declares, or null when it is unboxed (always, in a body
/// without boxes).
pub fn boxedBinding(b: *const Builder, node_idx: Ast.Index) ?u32 {
    const res = b.in.resolutions[(node_idx).int()];
    if (res != .local or !b.isBoxedLocal(res.local)) return null;
    return res.local;
}

/// Bind boxed local `li` to a fresh zeroed GC cell sized for a `ty`, and return its `cell`
/// home: a `rawptr` slot (a precise root) holding the cell pointer. A local has one cell
/// slot, which each binding of it (a loop iteration, an or-pattern alternative) points at a
/// new cell, so every closure made so far keeps the cell it captured.
pub fn bindBox(b: *Builder, li: u32, ty: Typecheck.Type) error{OutOfMemory}!LocalHome {
    const sid = blk: {
        if (b.homes.get(li)) |h| if (h == .cell) break :blk h.cell;
        const s = try b.addSlot(Typecheck.Type.rawptr);
        try b.homes.put(b.gpa, li, .{ .cell = s });
        break :blk s;
    };
    const cell = try L.emitGcAlloc(b, @max(8, Abi.typeSize(ty, b.in.layouts, b.in.enum_layouts)));
    const sa = try b.emit(.{ .slot_addr = sid }, Typecheck.Type.int);
    _ = try b.emit(.{ .store = .{ .addr = sa, .val = cell, .ty = Typecheck.Type.rawptr } }, null);
    return .{ .cell = sid };
}

/// Move each boxed param (`lis[i]` bound to param slot `sids[i]`) into a cell of its own and
/// rebind it there; the param slot keeps only the incoming value.
pub fn boxParams(b: *Builder, lis: []const u32, sids: []const Ir.SlotId, tok: u32) error{OutOfMemory}!void {
    for (lis, sids) |li, sid| {
        if (!b.isBoxedLocal(li)) continue;
        std.debug.assert(b.homes.get(li).? == .slot);
        const ty = b.slots.items[sid].type;
        const dst = try homeAddr(b, try bindBox(b, li, ty));
        const src = try b.emit(.{ .slot_addr = sid }, Typecheck.Type.int);
        try closure.copyCapture(b, dst, src, ty, tok);
    }
}

/// Get (or lazily bind) the home of the local that `node_idx` resolves to. Params are
/// pre-bound in `lowerFn`; a `:=` local / for-var gets its own slot here the first time it
/// is touched. Allocating in touch order keeps slot ids deterministic for a given source.
/// A boxed local's home is its cell (bound by `bindBox`), which only a box-aware caller may
/// take.
pub fn bindingHome(b: *Builder, node_idx: Ast.Index, ty: Typecheck.Type) error{OutOfMemory}!LocalHome {
    const res = b.in.resolutions[(node_idx).int()];
    std.debug.assert(res == .local);
    const li = res.local;
    if (b.homes.get(li)) |h| return h;
    if (b.isBoxedLocal(li))
        try b.note(b.in.tree.nodes[(node_idx).int()].main_token, "boxed local used before its cell is bound in lower");
    const sid = try b.addSlot(ty);
    try b.homes.put(b.gpa, li, .{ .slot = sid });
    return .{ .slot = sid };
}
