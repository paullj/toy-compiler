//! The stdlib public symbols the front/middle-end reaches for by name — the single
//! source of truth for those strings. Surface sugar (for-in, list literals, indexing)
//! desugars against these methods/types, and the checker, discovery, and lowerer must
//! agree on the exact spelling. A rename in the corresponding std/*.toy otherwise
//! silently breaks that sugar with no compiler error; centralizing the strings makes
//! the rename a one-line edit here.

const std = @import("std");

/// The list/vector struct (`std/vec.toy`) list-literal and index sugar target.
pub const vec_struct = "Vec";

/// The protocol a `for-in` receiver's `iter()` must yield (`std/iter.toy`), matched by
/// name against `model.protocols`.
pub const iter_protocol = "Iterator";

/// Associated / instance method names the surface sugar desugars to.
pub const method_new = "new";
pub const method_push = "push";
pub const method_iter = "iter";
pub const method_next = "next";

/// The `core/mem` element-read primitive `xs[i]` desugars to, by qualified name.
pub const ga_at = "core/mem.ga_at";

test "method names are distinct, non-empty, and ga_at is qualified" {
    const methods = [_][]const u8{ method_new, method_push, method_iter, method_next };
    for (methods) |m| try std.testing.expect(m.len != 0);
    for (methods, 0..) |a, i| for (methods[i + 1 ..]) |b|
        try std.testing.expect(!std.mem.eql(u8, a, b));
    try std.testing.expect(std.mem.indexOfScalar(u8, ga_at, '.') != null);
    try std.testing.expect(vec_struct.len != 0 and iter_protocol.len != 0);
}
