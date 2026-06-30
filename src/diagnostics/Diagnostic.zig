//! The one diagnostic type shared by every stage. `byte_offset` points at the
//! offending token; the driver/CLI renders byte_offset -> line:col uniformly.
//! `scope` carries the owning module id in graph mode (`NO_SCOPE` single-file).

const std = @import("std");

/// The "untagged" scope: a single-file diagnostic carries no module id.
pub const NO_SCOPE: u32 = std.math.maxInt(u32);

/// The one diagnostic type shared by every stage. `byte_offset` points at the
/// offending token; the driver/CLI renders byte_offset -> line:col uniformly.
/// `scope` is the owning module id in graph mode, `NO_SCOPE` single-file. The
/// default keeps every existing `.{ .byte_offset = x, .message = m }` literal
/// compiling unchanged.
pub const Diagnostic = struct {
    byte_offset: u32,
    message: []const u8,
    scope: u32 = NO_SCOPE,
};
