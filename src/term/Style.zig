//! Composable text style: `Color` (ansi/indexed/rgb) with capability
//! `downgrade`, plus `Style` and `sgrInto`/`closeInto`/`styled`. Emission is
//! gated so a `.none` level (or an empty style) writes zero bytes — never a
//! bare `\x1b[m` that would read as an unintended reset.

const std = @import("std");
const ansi = @import("ansi.zig");

pub const ColorLevel = enum { none, ansi16, ansi256, truecolor };

// Standard 16-color ANSI palette RGB, used as the target set when downgrading
// to .ansi16 (nearest by Euclidean distance). Index matches the .ansi tag.
const ansi16_rgb = [16][3]u8{
    .{ 0, 0, 0 }, // 0 black
    .{ 205, 0, 0 }, // 1 red
    .{ 0, 205, 0 }, // 2 green
    .{ 205, 205, 0 }, // 3 yellow
    .{ 0, 0, 238 }, // 4 blue
    .{ 205, 0, 205 }, // 5 magenta
    .{ 0, 205, 205 }, // 6 cyan
    .{ 229, 229, 229 }, // 7 white
    .{ 127, 127, 127 }, // 8 bright black
    .{ 255, 0, 0 }, // 9 bright red
    .{ 0, 255, 0 }, // 10 bright green
    .{ 255, 255, 0 }, // 11 bright yellow
    .{ 92, 92, 255 }, // 12 bright blue
    .{ 255, 0, 255 }, // 13 bright magenta
    .{ 0, 255, 255 }, // 14 bright cyan
    .{ 255, 255, 255 }, // 15 bright white
};

pub const Color = union(enum) {
    ansi: u4,
    indexed: u8,
    rgb: struct { r: u8, g: u8, b: u8 },

    /// Reduce capability to fit `level`; never increases color fidelity.
    pub fn downgrade(self: Color, level: ColorLevel) Color {
        return switch (level) {
            .none, .truecolor => self,
            .ansi256 => switch (self) {
                .ansi, .indexed => self,
                .rgb => |c| .{ .indexed = rgbToCube(c.r, c.g, c.b) },
            },
            .ansi16 => switch (self) {
                .ansi => self,
                .indexed => |n| if (n < 16)
                    .{ .ansi = @intCast(n) }
                else blk: {
                    const c = cubeToRgb(n);
                    break :blk .{ .ansi = nearestAnsi16(c[0], c[1], c[2]) };
                },
                .rgb => |c| .{ .ansi = nearestAnsi16(c.r, c.g, c.b) },
            },
        };
    }
};

fn q(v: u8) u8 {
    return @intFromFloat(@round(@as(f32, @floatFromInt(v)) / 255.0 * 5.0));
}

fn rgbToCube(r: u8, g: u8, b: u8) u8 {
    if (r == g and g == b) {
        const gray: u8 = @intFromFloat(@round(@as(f32, @floatFromInt(r)) / 255.0 * 23.0));
        return 232 + gray;
    }
    return 16 + 36 * q(r) + 6 * q(g) + q(b);
}

// Inverse of rgbToCube, used only to re-derive an RGB approximation for an
// xterm-256 index when downgrading further to .ansi16.
fn cubeToRgb(n: u8) [3]u8 {
    if (n >= 232) {
        const gray: u8 = @intCast(@as(u16, n - 232) * 255 / 23);
        return .{ gray, gray, gray };
    }
    const steps = [_]u8{ 0, 95, 135, 175, 215, 255 };
    const i = n - 16;
    return .{ steps[i / 36], steps[(i / 6) % 6], steps[i % 6] };
}

fn nearestAnsi16(r: u8, g: u8, b: u8) u4 {
    var best: u4 = 0;
    var best_dist: u32 = std.math.maxInt(u32);
    for (ansi16_rgb, 0..) |p, i| {
        const dr = @as(i32, r) - @as(i32, p[0]);
        const dg = @as(i32, g) - @as(i32, p[1]);
        const db = @as(i32, b) - @as(i32, p[2]);
        const dist: u32 = @intCast(dr * dr + dg * dg + db * db);
        if (dist < best_dist) {
            best_dist = dist;
            best = @intCast(i);
        }
    }
    return best;
}

pub const Style = struct {
    fg: ?Color = null,
    bg: ?Color = null,
    bold: bool = false,
    dim: bool = false,
    italic: bool = false,
    underline: bool = false,
    inverse: bool = false,

    fn isEmpty(self: Style) bool {
        return self.fg == null and self.bg == null and
            !self.bold and !self.dim and !self.italic and
            !self.underline and !self.inverse;
    }

    pub fn sgrInto(self: Style, w: *std.Io.Writer, level: ColorLevel) !void {
        if (level == .none) return;
        if (self.isEmpty()) return;

        var pbuf: [64]u8 = undefined;
        var pw = std.Io.Writer.fixed(&pbuf);
        var wrote = false;

        const attrs = [_]struct { on: bool, code: []const u8 }{
            .{ .on = self.bold, .code = ansi.bold },
            .{ .on = self.dim, .code = ansi.dim },
            .{ .on = self.italic, .code = ansi.italic },
            .{ .on = self.underline, .code = ansi.underline },
            .{ .on = self.inverse, .code = ansi.inverse },
        };
        for (attrs) |a| {
            if (!a.on) continue;
            if (wrote) try pw.writeByte(';');
            try pw.writeAll(a.code);
            wrote = true;
        }
        if (self.fg) |c| {
            if (wrote) try pw.writeByte(';');
            try writeColor(&pw, c.downgrade(level), false);
            wrote = true;
        }
        if (self.bg) |c| {
            if (wrote) try pw.writeByte(';');
            try writeColor(&pw, c.downgrade(level), true);
            wrote = true;
        }

        try ansi.frame(w, pw.buffered());
    }

    pub fn closeInto(self: Style, w: *std.Io.Writer, level: ColorLevel) !void {
        if (level == .none) return;
        if (self.isEmpty()) return;
        try w.writeAll(ansi.reset);
    }

    pub fn styled(self: Style, w: *std.Io.Writer, level: ColorLevel, text: []const u8) !void {
        try self.sgrInto(w, level);
        try w.writeAll(text);
        try self.closeInto(w, level);
    }
};

fn writeColor(w: *std.Io.Writer, c: Color, is_bg: bool) !void {
    switch (c) {
        .ansi => |n| {
            const base: u16 = if (n < 8)
                (if (is_bg) @as(u16, 40) else 30) + n
            else
                (if (is_bg) @as(u16, 100) else 90) + (@as(u16, n) - 8);
            try w.print("{d}", .{base});
        },
        .indexed => |idx| try w.print("{s};5;{d}", .{ if (is_bg) "48" else "38", idx }),
        .rgb => |v| try w.print("{s};2;{d};{d};{d}", .{ if (is_bg) "48" else "38", v.r, v.g, v.b }),
    }
}

test "sgrInto ansi16 fg with bold exact bytes" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .fg = .{ .ansi = 1 }, .bold = true };
    try s.sgrInto(&fw, .ansi16);
    try std.testing.expectEqualStrings("\x1b[1;31m", fw.buffered());
}

test "sgrInto indexed fg ansi256 exact bytes" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .fg = .{ .indexed = 200 } };
    try s.sgrInto(&fw, .ansi256);
    try std.testing.expectEqualStrings("\x1b[38;5;200m", fw.buffered());
}

test "sgrInto rgb fg truecolor exact bytes" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .fg = .{ .rgb = .{ .r = 10, .g = 20, .b = 30 } } };
    try s.sgrInto(&fw, .truecolor);
    try std.testing.expectEqualStrings("\x1b[38;2;10;20;30m", fw.buffered());
}

test "sgrInto bright ansi fg and bg" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    // bright red (9) fg -> 91, blue (4) bg -> 44
    const s = Style{ .fg = .{ .ansi = 9 }, .bg = .{ .ansi = 4 } };
    try s.sgrInto(&fw, .ansi16);
    try std.testing.expectEqualStrings("\x1b[91;44m", fw.buffered());
}

test "sgrInto attribute order fixed" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .underline = true, .bold = true, .italic = true };
    try s.sgrInto(&fw, .truecolor);
    try std.testing.expectEqualStrings("\x1b[1;3;4m", fw.buffered());
}

test "sgrInto all attributes emit codes in fixed order" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .bold = true, .dim = true, .italic = true, .underline = true, .inverse = true };
    try s.sgrInto(&fw, .truecolor);
    try std.testing.expectEqualStrings("\x1b[1;2;3;4;7m", fw.buffered());
}

test "sgrInto none emits nothing" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .fg = .{ .ansi = 1 }, .bold = true };
    try s.sgrInto(&fw, .none);
    try std.testing.expectEqualStrings("", fw.buffered());
}

test "sgrInto empty style emits nothing at color level" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{};
    try s.sgrInto(&fw, .ansi16);
    try std.testing.expectEqualStrings("", fw.buffered());
    var buf2: [256]u8 = undefined;
    var fw2 = std.Io.Writer.fixed(&buf2);
    try s.sgrInto(&fw2, .truecolor);
    try std.testing.expectEqualStrings("", fw2.buffered());
}

test "closeInto non-empty at color level resets" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .bold = true };
    try s.closeInto(&fw, .ansi256);
    try std.testing.expectEqualStrings("\x1b[0m", fw.buffered());
}

test "closeInto at none is empty" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .bold = true };
    try s.closeInto(&fw, .none);
    try std.testing.expectEqualStrings("", fw.buffered());
}

test "closeInto empty style any level is empty" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{};
    try s.closeInto(&fw, .truecolor);
    try std.testing.expectEqualStrings("", fw.buffered());
}

test "styled at none writes only text" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .fg = .{ .ansi = 2 }, .bold = true };
    try s.styled(&fw, .none, "hello");
    try std.testing.expectEqualStrings("hello", fw.buffered());
}

test "styled at color level wraps text" {
    var buf: [256]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const s = Style{ .fg = .{ .ansi = 1 } };
    try s.styled(&fw, .ansi16, "hi");
    try std.testing.expectEqualStrings("\x1b[31mhi\x1b[0m", fw.buffered());
}

test "downgrade truecolor is identity" {
    const c = Color{ .rgb = .{ .r = 1, .g = 2, .b = 3 } };
    try std.testing.expectEqual(c, c.downgrade(.truecolor));
}

test "downgrade rgb pure red to ansi256 is 196" {
    const c = Color{ .rgb = .{ .r = 255, .g = 0, .b = 0 } };
    const d = c.downgrade(.ansi256);
    try std.testing.expectEqual(@as(u8, 196), d.indexed);
}

test "downgrade rgb pure red to ansi16 is bright red" {
    const c = Color{ .rgb = .{ .r = 255, .g = 0, .b = 0 } };
    const d = c.downgrade(.ansi16);
    try std.testing.expectEqual(@as(u4, 9), d.ansi);
}

test "downgrade indexed low to ansi16 maps straight" {
    const c = Color{ .indexed = 4 };
    const d = c.downgrade(.ansi16);
    try std.testing.expectEqual(@as(u4, 4), d.ansi);
}

test "downgrade ansi unchanged at lower levels" {
    const c = Color{ .ansi = 12 };
    try std.testing.expectEqual(c, c.downgrade(.ansi16));
    try std.testing.expectEqual(c, c.downgrade(.ansi256));
}
