//! The core-only heap / raw-pointer intrinsics — the single source of truth for
//! their names. The resolver seeds them, the checker types them, and the lowerer
//! folds or routes them; each keyed off `Kind` here instead of re-spelling the
//! name strings, so a new intrinsic is one enum row + one `name` arm and every
//! consumer's exhaustive `switch` fails to compile until it handles it.

const std = @import("std");

/// Declared in the order the resolver seeds them. Seed order fixes each intrinsic
/// fn's global id, which feeds the content fingerprint / -jN identity, so the
/// order is load-bearing — append, never reorder.
pub const Kind = enum { size_of, align_of, gc_alloc, gc_span_count, store, load, gc_array, offset, gc_collect, gc_stats };

pub fn name(k: Kind) []const u8 {
    return switch (k) {
        .size_of => "size_of",
        .align_of => "align_of",
        .gc_alloc => "gc_alloc",
        .gc_span_count => "gc_span_count",
        .store => "store",
        .load => "load",
        .gc_array => "gc_array",
        .offset => "offset",
        .gc_collect => "gc_collect",
        .gc_stats => "gc_stats",
    };
}

pub fn lookup(n: []const u8) ?Kind {
    inline for (@typeInfo(Kind).@"enum".fields) |f| {
        const k: Kind = @enumFromInt(f.value);
        if (std.mem.eql(u8, n, name(k))) return k;
    }
    return null;
}

test "name/lookup round-trip over every Kind, in seed order" {
    const order = std.enums.values(Kind);
    try std.testing.expectEqual(Kind.size_of, order[0]);
    try std.testing.expectEqual(Kind.gc_stats, order[order.len - 1]);
    for (order) |k| try std.testing.expectEqual(k, lookup(name(k)).?);
    try std.testing.expect(lookup("not_an_intrinsic") == null);
}
