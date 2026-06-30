//! Raw SGR framing primitives: the CSI introducer, the full reset, the
//! attribute codes, and a helper that wraps an already-assembled parameter
//! slice in `\x1b[` ... `m`. This is the lowest layer and has no notion of
//! Color or Style — it only frames bytes a caller has already decided on.

const std = @import("std");

pub const csi = "\x1b[";
pub const reset = "\x1b[0m";

pub const bold = "1";
pub const dim = "2";
pub const italic = "3";
pub const underline = "4";
pub const inverse = "7";

/// Frame caller-assembled SGR parameters (e.g. "1;38;5;12") as `\x1b[` + params + `m`.
pub fn frame(w: *std.Io.Writer, params: []const u8) !void {
    try w.writeAll(csi);
    try w.writeAll(params);
    try w.writeByte('m');
}

test "frame wraps params in CSI ... m" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    try frame(&fw, "1;38;5;12");
    try std.testing.expectEqualStrings("\x1b[1;38;5;12m", fw.buffered());
}

test "reset and csi constants" {
    try std.testing.expectEqualStrings("\x1b[0m", reset);
    try std.testing.expectEqualStrings("\x1b[", csi);
}
