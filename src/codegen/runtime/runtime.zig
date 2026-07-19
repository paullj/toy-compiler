//! The hand-emitted AArch64 runtime package root: the `HandBuiltin` table the link
//! tail scans + appends off, re-exporting the surface `link/emit.zig` needs.

const std = @import("std");
const Link = @import("../../link/Link.zig");
const emit = @import("emit.zig");
const str = @import("str.zig");
const panic = @import("panic.zig");
const gc = @import("gc.zig");
const trace = @import("trace.zig");

/// A hand-emitted AArch64 builtin body and its stable name. The link tail both
/// scans for referenced builtins and appends their bodies off this one list, so
/// the image stays a pure function of the fn set.
pub const HandBuiltin = struct {
    name: []const u8,
    lower: *const fn (std.mem.Allocator) error{OutOfMemory}!Link.FnCode,
    /// Other hand-emitted builtins this body statically calls from WITHIN its appended
    /// machine code (e.g. `gc_alloc` triggers `gc_collect`). Such an edge is invisible to
    /// the link tail's fn-only used-scan, so the linker closes the used set over it — a
    /// program that carries this builtin also carries its callees. Each name must be a
    /// `hand_builtins` row.
    static_call_deps: []const []const u8 = &.{},
};

/// The FIXED append order of the hand-emitted builtins. Load-bearing: it makes the
/// linked image order-independent of the fn set's discovery/thread order. Append,
/// never reorder.
pub const hand_builtins = [_]HandBuiltin{
    .{ .name = "panic", .lower = panic.lowerPanic },
    .{ .name = "gc_alloc", .lower = gc.lowerGcAlloc, .static_call_deps = &.{"gc_collect"} },
    .{ .name = "gc_span_count", .lower = gc.lowerGcSpanCount },
    .{ .name = "gc_collect", .lower = gc.lowerGcCollect, .static_call_deps = &.{ "ga_trace", "mp_trace" } },
    .{ .name = "gc_stats", .lower = gc.lowerGcStats },
    .{ .name = "gc_mark", .lower = gc.lowerGcMark },
    .{ .name = "gc_mark_leaf", .lower = gc.lowerGcMarkLeaf },
    .{ .name = "ga_trace", .lower = trace.lowerGaTrace, .static_call_deps = &.{"gc_mark_leaf"} },
    .{ .name = "mp_trace", .lower = trace.lowerMpTrace, .static_call_deps = &.{"gc_mark_leaf"} },
    .{ .name = "text_base", .lower = panic.lowerTextBase },
    .{ .name = "__int_to_str", .lower = str.lowerIntToStr, .static_call_deps = &.{"gc_alloc"} },
    .{ .name = "__str_concat", .lower = str.lowerStrConcat, .static_call_deps = &.{"gc_alloc"} },
};

/// The `hand_builtins` index of `name`, resolved at comptime — lets a caller name
/// a specific builtin (e.g. `panic`, which drives the backtrace symbol table)
/// without hardcoding a numeric slot.
pub fn handBuiltinIndex(comptime name: []const u8) usize {
    for (hand_builtins, 0..) |hb, i| {
        if (std.mem.eql(u8, hb.name, name)) return i;
    }
    @compileError("unknown hand builtin: " ++ name);
}

test "hand_builtins static_call_deps name real rows; gc_alloc pulls gc_collect" {
    for (hand_builtins) |hb| {
        for (hb.static_call_deps) |dep| {
            var found = false;
            for (hand_builtins) |other| {
                if (std.mem.eql(u8, other.name, dep)) found = true;
            }
            try std.testing.expect(found);
        }
    }
    const deps = hand_builtins[comptime handBuiltinIndex("gc_alloc")].static_call_deps;
    try std.testing.expectEqual(@as(usize, 1), deps.len);
    try std.testing.expectEqualStrings("gc_collect", deps[0]);

    const its = hand_builtins[comptime handBuiltinIndex("__int_to_str")].static_call_deps;
    try std.testing.expectEqual(@as(usize, 1), its.len);
    try std.testing.expectEqualStrings("gc_alloc", its[0]);

    const sc = hand_builtins[comptime handBuiltinIndex("__str_concat")].static_call_deps;
    try std.testing.expectEqual(@as(usize, 1), sc.len);
    try std.testing.expectEqualStrings("gc_alloc", sc[0]);

    // The collector's precise container traces: gc_collect may dispatch to ga_trace/mp_trace
    // on a container pop, and each leaf-marks its backings via gc_mark_leaf. These edges are
    // asm-level `bl`s invisible to the fn-only used-scan, so the linker closes over them.
    const gc_deps = hand_builtins[comptime handBuiltinIndex("gc_collect")].static_call_deps;
    try std.testing.expectEqual(@as(usize, 2), gc_deps.len);
    try std.testing.expectEqualStrings("ga_trace", gc_deps[0]);
    try std.testing.expectEqualStrings("mp_trace", gc_deps[1]);
    const gat = hand_builtins[comptime handBuiltinIndex("ga_trace")].static_call_deps;
    try std.testing.expectEqual(@as(usize, 1), gat.len);
    try std.testing.expectEqualStrings("gc_mark_leaf", gat[0]);
    const mpt = hand_builtins[comptime handBuiltinIndex("mp_trace")].static_call_deps;
    try std.testing.expectEqual(@as(usize, 1), mpt.len);
    try std.testing.expectEqualStrings("gc_mark_leaf", mpt[0]);
}

test {
    _ = emit;
    _ = str;
    _ = panic;
    _ = gc;
    _ = trace;
}
