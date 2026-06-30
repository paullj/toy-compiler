//! Color/tty policy. The resolution rule (`resolve`) is a pure function over an
//! explicit `ColorChoice`, a pre-computed `is_tty`, and an `EnvView` of already
//! extracted environment signals — it reads no environment and does no I/O, so
//! the whole precedence ladder is unit-testable without a real terminal. The
//! single impure step (asking the OS whether the output is a tty) lives in the
//! `detectTty` edge helper; the caller runs it once and feeds the boolean in.

const std = @import("std");
const Style = @import("Style.zig");

pub const ColorLevel = Style.ColorLevel;

/// Explicit user intent, e.g. from a future `--color` flag. `.auto` defers to
/// the environment + tty ladder; `.always`/`.never` short-circuit it.
pub const ColorChoice = enum { auto, always, never };

/// Already-extracted environment signals. Booleans are *presence/value*
/// decisions made by the builder at the edge so `resolve` stays pure and free
/// of string parsing. In particular `no_color` honors NO_COLOR regardless of
/// its value (even empty), per the no-color.org contract.
pub const EnvView = struct {
    /// NO_COLOR is set (any value, including empty).
    no_color: bool = false,
    /// CLICOLOR_FORCE is set to a non-empty, non-"0" value.
    clicolor_force: bool = false,
    /// CLICOLOR is explicitly "0" (disable color even on a tty).
    clicolor_zero: bool = false,
    /// TERM=dumb — terminal that cannot render SGR.
    term_dumb: bool = false,
    /// COLORTERM indicates truecolor ("truecolor" or "24bit").
    truecolor: bool = false,
    /// TERM names a 256-color terminal (contains "256color").
    ansi256: bool = false,

    /// Build an `EnvView` from an environment map. Pure: no I/O, no allocation.
    /// `map` is anything exposing `get([]const u8) ?[]const u8` and
    /// `contains([]const u8) bool` (e.g. `std.process.Environ.Map`), so tests
    /// can pass a lightweight stub. NO_COLOR is detected by presence alone.
    pub fn fromEnv(map: anytype) EnvView {
        const colorterm = map.get("COLORTERM") orelse "";
        const term = map.get("TERM") orelse "";
        return .{
            .no_color = map.contains("NO_COLOR"),
            .clicolor_force = isForce(map.get("CLICOLOR_FORCE")),
            .clicolor_zero = strEq(map.get("CLICOLOR"), "0"),
            .term_dumb = strEq(term, "dumb"),
            .truecolor = strEq(colorterm, "truecolor") or strEq(colorterm, "24bit"),
            .ansi256 = std.mem.indexOf(u8, term, "256color") != null,
        };
    }

    /// CLICOLOR_FORCE forces color when set to anything other than unset/empty/"0".
    fn isForce(v: ?[]const u8) bool {
        const s = v orelse return false;
        return s.len != 0 and !std.mem.eql(u8, s, "0");
    }

    fn strEq(a: ?[]const u8, b: []const u8) bool {
        return std.mem.eql(u8, a orelse return false, b);
    }
};

/// Pure resolution of the color level. Precedence (roadmap §5, with an added
/// `.ansi256` tier): explicit `.never` => `.none`; `.always` => tier; under
/// `.auto`, NO_COLOR wins (=> `.none`), then CLICOLOR_FORCE (=> tier), then any
/// suppressor — CLICOLOR=0, TERM=dumb, or not-a-tty — (=> `.none`), else tier.
/// `tier` is `.truecolor` if COLORTERM says so, else `.ansi256` if TERM is a
/// 256-color terminal, else `.ansi16`.
pub fn resolve(choice: ColorChoice, is_tty: bool, env: EnvView) ColorLevel {
    const t = tier(env);
    return switch (choice) {
        .never => .none,
        // .always overrides NO_COLOR and every suppressor, by design.
        .always => t,
        .auto => blk: {
            if (env.no_color) break :blk .none;
            if (env.clicolor_force) break :blk t;
            if (env.clicolor_zero or env.term_dumb or !is_tty) break :blk .none;
            break :blk t;
        },
    };
}

fn tier(env: EnvView) ColorLevel {
    if (env.truecolor) return .truecolor;
    if (env.ansi256) return .ansi256;
    return .ansi16;
}

/// Impure edge helper: ask the OS once whether `file` is a tty. A failed query
/// (`Io.Cancelable`) is treated as "not a tty" so the gate-safe default is plain
/// output. Run this at the edge and pass the result to `Terminal.init`.
pub fn detectTty(file: std.Io.File, io: std.Io) bool {
    return file.isTty(io) catch false;
}

/// A resolved output target: a color `level` paired with a writer. `init` is
/// fully pure (no `io`/`File`) — detect the tty with `detectTty` at the edge,
/// then hand the boolean here.
pub const Terminal = struct {
    level: ColorLevel,
    writer: *std.Io.Writer,

    pub fn init(choice: ColorChoice, is_tty: bool, env: EnvView, w: *std.Io.Writer) Terminal {
        return .{ .level = resolve(choice, is_tty, env), .writer = w };
    }

    /// Test/forcing constructor: pin a level directly, bypassing resolution.
    pub fn fromLevel(level: ColorLevel, w: *std.Io.Writer) Terminal {
        return .{ .level = level, .writer = w };
    }

    /// Plain (`.none`) terminal — convenience for non-tty/test sinks.
    pub fn plain(w: *std.Io.Writer) Terminal {
        return .{ .level = .none, .writer = w };
    }

    /// Emit `style`'s SGR open codes at this terminal's level.
    pub fn sgr(self: Terminal, style: Style.Style) !void {
        try style.sgrInto(self.writer, self.level);
    }

    /// Emit `text` wrapped in `style`'s open/close codes at this terminal's level.
    pub fn styled(self: Terminal, style: Style.Style, text: []const u8) !void {
        try style.styled(self.writer, self.level, text);
    }
};

// A minimal stub matching the EnvView.fromEnv duck type, so the builder is
// testable without constructing a real allocating process.Environ.Map.
const StubEnv = struct {
    pairs: []const [2][]const u8,
    fn get(self: StubEnv, key: []const u8) ?[]const u8 {
        for (self.pairs) |p| if (std.mem.eql(u8, p[0], key)) return p[1];
        return null;
    }
    fn contains(self: StubEnv, key: []const u8) bool {
        return self.get(key) != null;
    }
};

fn stub(pairs: []const [2][]const u8) StubEnv {
    return .{ .pairs = pairs };
}

test "resolve never always wins to none" {
    const env = EnvView{ .truecolor = true };
    try std.testing.expectEqual(ColorLevel.none, resolve(.never, true, env));
    try std.testing.expectEqual(ColorLevel.none, resolve(.never, false, env));
    // .never beats CLICOLOR_FORCE too
    try std.testing.expectEqual(ColorLevel.none, resolve(.never, true, .{ .clicolor_force = true }));
}

test "resolve always yields tier ignoring no_color and tty" {
    // .always overrides NO_COLOR, CLICOLOR=0, TERM=dumb, and not-a-tty.
    const env = EnvView{ .no_color = true, .clicolor_zero = true, .term_dumb = true };
    try std.testing.expectEqual(ColorLevel.ansi16, resolve(.always, false, env));
    try std.testing.expectEqual(ColorLevel.truecolor, resolve(.always, false, .{ .truecolor = true }));
    try std.testing.expectEqual(ColorLevel.ansi256, resolve(.always, false, .{ .ansi256 = true }));
}

test "resolve auto no_color beats clicolor_force" {
    const env = EnvView{ .no_color = true, .clicolor_force = true, .truecolor = true };
    try std.testing.expectEqual(ColorLevel.none, resolve(.auto, true, env));
}

test "resolve auto clicolor_force beats suppressors and non-tty" {
    const env = EnvView{ .clicolor_force = true, .clicolor_zero = true, .term_dumb = true };
    try std.testing.expectEqual(ColorLevel.ansi16, resolve(.auto, false, env));
}

test "resolve auto clicolor_force yields the actual tier (not just ansi16)" {
    // forced color on a non-tty must still honor the tier, not flatten to ansi16
    try std.testing.expectEqual(ColorLevel.truecolor, resolve(.auto, false, .{ .clicolor_force = true, .truecolor = true }));
    try std.testing.expectEqual(ColorLevel.ansi256, resolve(.auto, false, .{ .clicolor_force = true, .ansi256 = true }));
}

test "resolve auto not a tty is none" {
    try std.testing.expectEqual(ColorLevel.none, resolve(.auto, false, .{ .truecolor = true }));
}

test "resolve auto tty plain tier is ansi16" {
    try std.testing.expectEqual(ColorLevel.ansi16, resolve(.auto, true, .{}));
}

test "resolve auto clicolor_zero suppresses on tty" {
    try std.testing.expectEqual(ColorLevel.none, resolve(.auto, true, .{ .clicolor_zero = true }));
}

test "resolve auto term_dumb suppresses on tty" {
    try std.testing.expectEqual(ColorLevel.none, resolve(.auto, true, .{ .term_dumb = true, .truecolor = true }));
}

test "resolve auto tier truecolor on tty" {
    try std.testing.expectEqual(ColorLevel.truecolor, resolve(.auto, true, .{ .truecolor = true }));
}

test "resolve auto tier ansi256 on tty" {
    try std.testing.expectEqual(ColorLevel.ansi256, resolve(.auto, true, .{ .ansi256 = true }));
}

test "resolve auto truecolor outranks ansi256 in tier" {
    try std.testing.expectEqual(ColorLevel.truecolor, resolve(.auto, true, .{ .truecolor = true, .ansi256 = true }));
}

test "fromEnv no_color present empty value still disables" {
    const env = EnvView.fromEnv(stub(&.{.{ "NO_COLOR", "" }}));
    try std.testing.expect(env.no_color);
    try std.testing.expectEqual(ColorLevel.none, resolve(.auto, true, env));
}

test "fromEnv clicolor_force empty or zero is not forcing" {
    try std.testing.expect(!EnvView.fromEnv(stub(&.{.{ "CLICOLOR_FORCE", "" }})).clicolor_force);
    try std.testing.expect(!EnvView.fromEnv(stub(&.{.{ "CLICOLOR_FORCE", "0" }})).clicolor_force);
    try std.testing.expect(EnvView.fromEnv(stub(&.{.{ "CLICOLOR_FORCE", "1" }})).clicolor_force);
}

test "fromEnv clicolor zero vs unset" {
    try std.testing.expect(EnvView.fromEnv(stub(&.{.{ "CLICOLOR", "0" }})).clicolor_zero);
    try std.testing.expect(!EnvView.fromEnv(stub(&.{.{ "CLICOLOR", "1" }})).clicolor_zero);
    try std.testing.expect(!EnvView.fromEnv(stub(&.{})).clicolor_zero);
}

test "fromEnv truecolor from COLORTERM" {
    try std.testing.expect(EnvView.fromEnv(stub(&.{.{ "COLORTERM", "truecolor" }})).truecolor);
    try std.testing.expect(EnvView.fromEnv(stub(&.{.{ "COLORTERM", "24bit" }})).truecolor);
    try std.testing.expect(!EnvView.fromEnv(stub(&.{.{ "COLORTERM", "yes" }})).truecolor);
}

test "fromEnv ansi256 from TERM" {
    try std.testing.expect(EnvView.fromEnv(stub(&.{.{ "TERM", "xterm-256color" }})).ansi256);
    try std.testing.expect(!EnvView.fromEnv(stub(&.{.{ "TERM", "xterm" }})).ansi256);
    try std.testing.expect(EnvView.fromEnv(stub(&.{.{ "TERM", "dumb" }})).term_dumb);
}

test "fromEnv then resolve end to end on tty" {
    const env = EnvView.fromEnv(stub(&.{ .{ "TERM", "xterm-256color" }, .{ "COLORTERM", "truecolor" } }));
    try std.testing.expectEqual(ColorLevel.truecolor, resolve(.auto, true, env));
}

test "Terminal init resolves and stores level" {
    var buf: [64]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const t = Terminal.init(.auto, false, .{ .truecolor = true }, &fw);
    try std.testing.expectEqual(ColorLevel.none, t.level);
}

test "Terminal styled at none writes plain bytes" {
    var buf: [64]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const t = Terminal.plain(&fw);
    try t.styled(.{ .fg = .{ .ansi = 1 }, .bold = true }, "hi");
    try std.testing.expectEqualStrings("hi", fw.buffered());
}

test "Terminal styled at color level wraps" {
    var buf: [64]u8 = undefined;
    var fw = std.Io.Writer.fixed(&buf);
    const t = Terminal.fromLevel(.ansi16, &fw);
    try t.styled(.{ .fg = .{ .ansi = 1 } }, "hi");
    try std.testing.expectEqualStrings("\x1b[31mhi\x1b[0m", fw.buffered());
}
