//! Cursor positions for LSP tests, computed from the fixture text so no test hand-counts
//! a column. ASCII fixtures, so a byte offset equals a character offset.

const std = @import("std");
const protocol = @import("protocol.zig");

fn posAt(src: []const u8, idx: usize) protocol.Position {
    var line: u32 = 0;
    var col: u32 = 0;
    for (src[0..idx]) |ch| {
        if (ch == '\n') {
            line += 1;
            col = 0;
        } else col += 1;
    }
    return .{ .line = line, .character = col };
}

/// The position of the first occurrence of `needle`.
pub fn posOf(src: []const u8, needle: []const u8) protocol.Position {
    return posAt(src, std.mem.indexOf(u8, src, needle).?);
}

/// The position just past the last occurrence of `needle`.
pub fn posAfterLast(src: []const u8, needle: []const u8) protocol.Position {
    return posAt(src, std.mem.lastIndexOf(u8, src, needle).? + needle.len);
}
