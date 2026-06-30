//! Caller-driven spinners and progress bars with a pure frame renderer. The
//! clock lives at the edge: the caller reads `Io.Clock.Timestamp.now(io,.awake)
//! .raw.nanoseconds` (i96) and passes it in widened to i128 — Progress holds no
//! Io, so `renderFrame`/`throttle` are pure and testable over literal
//! timestamps with a fixed-buffer writer, no fake clock. A `.none` color level
//! makes every public method a zero-byte no-op so Progress (meant for stderr)
//! is structurally incapable of perturbing stdout byte gates. Cursor-hide on
//! start is paired with show via an errdefer on every draw and finish path, so
//! no return or error ever leaves the cursor hidden.

const std = @import("std");
const Style = @import("Style.zig");

const ColorLevel = Style.ColorLevel;

pub const Ns = i128;

pub const min_interval_ns: Ns = 80 * std.time.ns_per_ms;

pub const braille = [_][]const u8{ "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" };
pub const ascii = [_][]const u8{ "|", "/", "-", "\\" };

const hide_cursor = "\x1b[?25l";
const show_cursor = "\x1b[?25h";
const clear_to_eol = "\x1b[K";

// PURE: enough time elapsed since the last draw? Early-returns false (skip the
// draw) when within the interval. Never sleeps. A sentinel last_ns of 0 with
// started==false is handled by the callers, which force the first draw.
fn throttle(last_ns: Ns, now_ns: Ns, interval_ns: Ns) bool {
    return now_ns - last_ns >= interval_ns;
}

pub const Spinner = struct {
    level: ColorLevel,
    frames: []const []const u8 = &braille,
    label: []const u8 = "",
    i: usize = 0,
    last_ns: Ns = 0,
    started: bool = false,

    pub fn init(level: ColorLevel, label: []const u8) Spinner {
        return .{ .level = level, .label = label };
    }

    // PURE: current frame + label into w, framed by \r and clear-to-EOL so a
    // shorter later label leaves no residue. No clock read, no flush.
    pub fn renderFrame(self: Spinner, w: *std.Io.Writer) !void {
        if (self.level == .none) return;
        try w.writeAll("\r");
        try w.writeAll(self.frames[self.i % self.frames.len]);
        if (self.label.len != 0) {
            try w.writeAll(" ");
            try w.writeAll(self.label);
        }
        try w.writeAll(clear_to_eol);
    }

    pub fn start(self: *Spinner, w: *std.Io.Writer, now_ns: Ns) !void {
        if (self.level == .none) return;
        errdefer w.writeAll(show_cursor) catch {};
        try w.writeAll(hide_cursor);
        self.started = true;
        self.last_ns = now_ns;
        try self.renderFrame(w);
        try w.flush();
    }

    pub fn tick(self: *Spinner, w: *std.Io.Writer, now_ns: Ns) !void {
        if (self.level == .none) return;
        errdefer w.writeAll(show_cursor) catch {};
        if (self.started and !throttle(self.last_ns, now_ns, min_interval_ns)) return;
        self.started = true;
        self.i +%= 1;
        self.last_ns = now_ns;
        try self.renderFrame(w);
        try w.flush();
    }

    pub fn finish(self: *Spinner, w: *std.Io.Writer, done: []const u8) !void {
        if (self.level == .none) return;
        errdefer w.writeAll(show_cursor) catch {};
        try w.writeAll("\r");
        try w.writeAll(clear_to_eol);
        if (done.len != 0) {
            try w.writeAll(done);
            try w.writeAll("\n");
        }
        try w.writeAll(show_cursor);
        try w.flush();
    }
};

pub const ProgressBar = struct {
    level: ColorLevel,
    total: u64,
    width: u16 = 30,
    current: u64 = 0,
    last_ns: Ns = 0,
    start_ns: Ns = 0,
    started: bool = false,
    draws: u64 = 0,

    pub fn init(level: ColorLevel, total: u64) ProgressBar {
        return .{ .level = level, .total = total };
    }

    // PURE: total==0 yields 0 rather than a divide-by-zero trap.
    fn ratio(cur: u64, total: u64) f64 {
        if (total == 0) return 0;
        return @as(f64, @floatFromInt(cur)) / @as(f64, @floatFromInt(total));
    }

    // PURE: items per second. elapsed_ns<=0 yields 0.
    fn rate(cur: u64, elapsed_ns: Ns) f64 {
        if (elapsed_ns <= 0) return 0;
        return @as(f64, @floatFromInt(cur)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(elapsed_ns));
    }

    // PURE: nanoseconds remaining, projected from observed rate. cur==0,
    // elapsed_ns<=0, or already-complete yields 0.
    fn etaNs(cur: u64, total: u64, elapsed_ns: Ns) Ns {
        if (cur == 0 or elapsed_ns <= 0 or cur >= total) return 0;
        const remaining = total - cur;
        const per_item = @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(cur));
        return @intFromFloat(per_item * @as(f64, @floatFromInt(remaining)));
    }

    // PURE: \r + [bar] pct% cur/total + clear-to-EOL into w. now_ns drives the
    // ETA only; no clock read, no flush.
    pub fn renderFrame(self: ProgressBar, w: *std.Io.Writer, now_ns: Ns) !void {
        if (self.level == .none) return;
        const r = ratio(self.current, self.total);
        const filled: u16 = @intFromFloat(@as(f64, @floatFromInt(self.width)) * r);
        // filled cells truncate (a cell isn't "full" until reached); the percentage
        // rounds so 7/10 reads 70%, not the 69% an f64 truncation would show.
        const pct: u64 = @intFromFloat(@round(r * 100));

        try w.writeAll("\r[");
        var c: u16 = 0;
        while (c < self.width) : (c += 1) {
            try w.writeAll(if (c < filled) "#" else "-");
        }
        try w.print("] {d}% {d}/{d}", .{ pct, self.current, self.total });

        const elapsed: Ns = if (self.started) now_ns - self.start_ns else 0;
        const eta = etaNs(self.current, self.total, elapsed);
        if (eta > 0) {
            const eta_s: u64 = @intCast(@divTrunc(eta, std.time.ns_per_s));
            try w.print(" eta {d}s", .{eta_s});
        }
        try w.writeAll(clear_to_eol);
    }

    pub fn set(self: *ProgressBar, w: *std.Io.Writer, current: u64, now_ns: Ns) !void {
        if (self.level == .none) return;
        errdefer w.writeAll(show_cursor) catch {};
        if (!self.started) {
            self.start_ns = now_ns;
            self.started = true;
            try w.writeAll(hide_cursor);
        } else if (!throttle(self.last_ns, now_ns, min_interval_ns)) {
            self.current = current;
            return;
        }
        self.current = current;
        self.last_ns = now_ns;
        self.draws += 1;
        try self.renderFrame(w, now_ns);
        try w.flush();
    }

    pub fn finish(self: *ProgressBar, w: *std.Io.Writer, now_ns: Ns) !void {
        if (self.level == .none) return;
        errdefer w.writeAll(show_cursor) catch {};
        self.current = self.total;
        try self.renderFrame(w, now_ns);
        try w.writeAll("\n");
        try w.writeAll(show_cursor);
        try w.flush();
    }
};

test "throttle pure decision within and past interval" {
    try std.testing.expect(!throttle(1000, 1000 + min_interval_ns - 1, min_interval_ns));
    try std.testing.expect(throttle(1000, 1000 + min_interval_ns, min_interval_ns));
    try std.testing.expect(throttle(1000, 1000 + min_interval_ns + 1, min_interval_ns));
}

test "spinner braille frame bytes" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner.init(.ansi16, "load");
    s.i = 1;
    try s.renderFrame(&w);
    try std.testing.expectEqualStrings("\r⠙ load\x1b[K", w.buffered());
}

test "spinner braille frame zero index no label" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner.init(.ansi16, "");
    try s.renderFrame(&w);
    try std.testing.expectEqualStrings("\r⠋\x1b[K", w.buffered());
}

test "spinner ascii frame bytes" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner{ .level = .ansi16, .frames = &ascii, .label = "x" };
    s.i = 2;
    try s.renderFrame(&w);
    try std.testing.expectEqualStrings("\r- x\x1b[K", w.buffered());
}

test "spinner start hides cursor and draws first frame" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner.init(.ansi16, "");
    try s.start(&w, 1000);
    try std.testing.expectEqualStrings("\x1b[?25l\r⠋\x1b[K", w.buffered());
}

test "spinner two ticks within interval draw once" {
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner.init(.ansi16, "");
    try s.start(&w, 0);
    const after_start = w.buffered().len;
    try s.tick(&w, 10); // within interval -> no draw
    try std.testing.expectEqual(after_start, w.buffered().len);
    try std.testing.expectEqual(@as(usize, 0), s.i);
    try s.tick(&w, min_interval_ns); // past interval -> draw, advance
    try std.testing.expect(w.buffered().len > after_start);
    try std.testing.expectEqual(@as(usize, 1), s.i);
}

test "spinner finish shows cursor and clears line" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner.init(.ansi16, "");
    try s.finish(&w, "done");
    try std.testing.expectEqualStrings("\r\x1b[Kdone\n\x1b[?25h", w.buffered());
}

test "progressbar bytes at 0 50 100 percent" {
    var buf: [128]u8 = undefined;

    var w0 = std.Io.Writer.fixed(&buf);
    var p0 = ProgressBar{ .level = .ansi16, .total = 10, .width = 4, .current = 0 };
    try p0.renderFrame(&w0, 0);
    try std.testing.expectEqualStrings("\r[----] 0% 0/10\x1b[K", w0.buffered());

    var w5 = std.Io.Writer.fixed(&buf);
    var p5 = ProgressBar{ .level = .ansi16, .total = 10, .width = 4, .current = 5 };
    try p5.renderFrame(&w5, 0);
    try std.testing.expectEqualStrings("\r[##--] 50% 5/10\x1b[K", w5.buffered());

    var w10 = std.Io.Writer.fixed(&buf);
    var p10 = ProgressBar{ .level = .ansi16, .total = 10, .width = 4, .current = 10 };
    try p10.renderFrame(&w10, 0);
    try std.testing.expectEqualStrings("\r[####] 100% 10/10\x1b[K", w10.buffered());
}

test "progressbar fractional fill truncates, percent rounds" {
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var p = ProgressBar{ .level = .ansi16, .total = 10, .width = 4, .current = 7 };
    // 7/10: filled = trunc(4 * 0.7) = 2 ("##--"); percent = round(70.0) = 70%
    try p.renderFrame(&w, 0);
    try std.testing.expectEqualStrings("\r[##--] 70% 7/10\x1b[K", w.buffered());
}

test "progressbar total zero renders without divide-by-zero" {
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var p = ProgressBar{ .level = .ansi16, .total = 0, .width = 4, .current = 0 };
    try p.renderFrame(&w, 0);
    try std.testing.expectEqualStrings("\r[----] 0% 0/0\x1b[K", w.buffered());
}

test "progressbar eta rendered from literal elapsed" {
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var p = ProgressBar{ .level = .ansi16, .total = 10, .width = 4, .current = 5, .started = true, .start_ns = 0 };
    // 5 of 10 done in 5s -> ~5s remaining
    try p.renderFrame(&w, 5 * std.time.ns_per_s);
    try std.testing.expectEqualStrings("\r[##--] 50% 5/10 eta 5s\x1b[K", w.buffered());
}

test "progressbar ratio rate eta divide-by-zero guards" {
    try std.testing.expectEqual(@as(f64, 0), ProgressBar.ratio(5, 0));
    try std.testing.expectEqual(@as(f64, 0), ProgressBar.rate(5, 0));
    try std.testing.expectEqual(@as(f64, 0), ProgressBar.rate(5, -1));
    try std.testing.expectEqual(@as(Ns, 0), ProgressBar.etaNs(0, 10, 1000)); // cur==0
    try std.testing.expectEqual(@as(Ns, 0), ProgressBar.etaNs(5, 10, 0)); // elapsed==0
    try std.testing.expectEqual(@as(Ns, 0), ProgressBar.etaNs(10, 10, 1000)); // complete
}

test "progressbar rate items per second" {
    try std.testing.expectEqual(@as(f64, 2), ProgressBar.rate(10, 5 * std.time.ns_per_s));
}

test "progressbar two sets within interval draw once" {
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var p = ProgressBar.init(.ansi16, 10);
    try p.set(&w, 1, 0); // first -> draws, hides cursor
    const after_first = w.buffered().len;
    try std.testing.expect(after_first > 0);
    try std.testing.expectEqual(@as(u64, 1), p.draws);
    try p.set(&w, 2, 10); // within interval -> no draw, current still tracked
    try std.testing.expectEqual(after_first, w.buffered().len);
    try std.testing.expectEqual(@as(u64, 2), p.current);
    try std.testing.expectEqual(@as(u64, 1), p.draws);
    try p.set(&w, 3, min_interval_ns); // past interval -> draws
    try std.testing.expect(w.buffered().len > after_first);
    try std.testing.expectEqual(@as(u64, 2), p.draws);
}

test "progressbar finish forces 100 percent and shows cursor" {
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var p = ProgressBar{ .level = .ansi16, .total = 10, .width = 4, .current = 3 };
    try p.finish(&w, 0);
    try std.testing.expectEqualStrings("\r[####] 100% 10/10\x1b[K\n\x1b[?25h", w.buffered());
}

test "plain mode spinner emits zero bytes" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner.init(.none, "load");
    try s.start(&w, 0);
    try s.tick(&w, min_interval_ns);
    try s.tick(&w, 10 * min_interval_ns);
    try s.renderFrame(&w);
    try s.finish(&w, "done");
    try std.testing.expectEqual(@as(usize, 0), w.buffered().len);
}

test "plain mode progressbar emits zero bytes" {
    var buf: [64]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var p = ProgressBar.init(.none, 10);
    try p.set(&w, 1, 0);
    try p.set(&w, 5, min_interval_ns);
    try p.renderFrame(&w, min_interval_ns);
    try p.finish(&w, 2 * min_interval_ns);
    try std.testing.expectEqual(@as(usize, 0), w.buffered().len);
}

test "cursor hide on start is always paired with show on finish" {
    var buf: [128]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    var s = Spinner.init(.ansi16, "");
    try s.start(&w, 0);
    try s.finish(&w, "");
    const out = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, out, hide_cursor) != null);
    try std.testing.expect(std.mem.indexOf(u8, out, show_cursor) != null);
}
