//! A conservative "did you mean X?" name suggester for undeclared-name diagnostics.
//!
//! There is no note/help/secondary-label channel on the emit side yet, so the
//! resolver embeds any suggestion straight into the message string. This module
//! only computes the single best candidate; it emits nothing itself.
//!
//! Bounded OSA (Damerau-Levenshtein with adjacent transposition) edit distance +
//! a deliberately conservative selection rule: a STRICT unique winner within a
//! small absolute cap, gated so short names and length-disproportionate matches
//! never suggest. The gates keep the one hard-guarded exact-string test
//! (`resolve.zig`: `y := x` → plain "undeclared identifier 'x'") byte-identical,
//! and the strict-unique-winner rule makes the emitted string independent of
//! hashmap iteration order (so `-j1` == `-jN` determinism holds).

const std = @import("std");

/// The absolute maximum edit distance we will ever suggest across.
const cap: usize = 2;
/// Names shorter than this (target OR candidate) never participate — a 1-2 char
/// typo is as likely to be a different intended name as a slip.
const min_len: usize = 3;

/// Best in-scope candidate for a typo'd `target`, or `null` when nothing is close
/// enough or the winner is ambiguous. `candidates` is any value with a
/// `next() ?[]const u8` method (an iterator yielding candidate names). Conservative:
///   * skips `target`/candidates shorter than `min_len`,
///   * skips an exact match (not a typo),
///   * requires the distance to be a small fraction of the longer name
///     (`dist*3 <= max(len)`, the rustc-style "close relative to length" rule),
///   * demands a STRICT unique winner (a tie at the minimum distance → `null`).
/// Deterministic regardless of candidate iteration order.
pub fn suggest(target: []const u8, candidates: anytype) ?[]const u8 {
    if (target.len < min_len) return null;
    var best: ?[]const u8 = null;
    var best_d: usize = cap + 1;
    var tied = false;
    var it = candidates;
    while (it.next()) |cand| {
        if (cand.len < min_len) continue;
        if (std.mem.eql(u8, cand, target)) continue;
        const d = osaDistance(target, cand, cap) orelse continue;
        if (d * 3 > @max(target.len, cand.len)) continue; // distance ~1/3 of length
        if (d < best_d) {
            best_d = d;
            best = cand;
            tied = false;
        } else if (d == best_d) {
            // A same-distance candidate whose STRING equals the current best is a
            // re-observation of the SAME name (a shadowed local, or a local sharing
            // a fn's name), not a genuine tie — candidateIter yields names from every
            // lexical scope with no de-duplication. Only a *different* string at the
            // minimum distance is an ambiguous tie.
            if (best == null or !std.mem.eql(u8, cand, best.?)) tied = true;
        }
    }
    return if (tied) null else best;
}

/// Optimal String Alignment distance (Damerau-Levenshtein restricted to adjacent
/// transpositions) with three rolling rows and an early-out when a whole row's
/// minimum exceeds `max`. Returns `null` when the distance exceeds `max` (or a name
/// is implausibly long). Stack buffers sized for the 64-char name cap.
fn osaDistance(a: []const u8, b: []const u8, max: usize) ?usize {
    if (a.len > 64 or b.len > 64) return null;
    // Length prune: if the lengths differ by more than `max`, distance > max.
    if (a.len > b.len + max or b.len > a.len + max) return null;

    var prev2: [65]usize = undefined; // row i-2
    var prev: [65]usize = undefined; // row i-1
    var cur: [65]usize = undefined; // row i
    for (0..b.len + 1) |j| prev[j] = j;

    for (1..a.len + 1) |i| {
        cur[0] = i;
        var row_min: usize = cur[0];
        for (1..b.len + 1) |j| {
            const cost: usize = if (a[i - 1] == b[j - 1]) 0 else 1;
            var v = @min(@min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + cost);
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1])
                v = @min(v, prev2[j - 2] + 1);
            cur[j] = v;
            row_min = @min(row_min, v);
        }
        if (row_min > max) return null;
        prev2 = prev;
        prev = cur;
    }
    return if (prev[b.len] <= max) prev[b.len] else null;
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;

/// A trivial iterator over a fixed candidate slice for the unit tests.
const SliceIter = struct {
    items: []const []const u8,
    i: usize = 0,
    fn next(self: *SliceIter) ?[]const u8 {
        if (self.i >= self.items.len) return null;
        defer self.i += 1;
        return self.items[self.i];
    }
};

fn iter(items: []const []const u8) SliceIter {
    return .{ .items = items };
}

test "osaDistance: substitution/insertion/deletion/transposition within cap" {
    try testing.expectEqual(@as(?usize, 0), osaDistance("count", "count", 2));
    try testing.expectEqual(@as(?usize, 1), osaDistance("count", "cont", 2)); // deletion
    try testing.expectEqual(@as(?usize, 1), osaDistance("cont", "count", 2)); // insertion
    try testing.expectEqual(@as(?usize, 1), osaDistance("count", "couat", 2)); // substitution
    try testing.expectEqual(@as(?usize, 1), osaDistance("teh", "the", 2)); // transposition
}

test "osaDistance: beyond cap returns null (early-out)" {
    try testing.expectEqual(@as(?usize, null), osaDistance("count", "xyzzy", 2));
    try testing.expectEqual(@as(?usize, null), osaDistance("abc", "abcdef", 2)); // len prune
}

test "suggest: a close typo of a candidate is returned" {
    const cands = [_][]const u8{ "count", "total", "index" };
    var it = iter(&cands);
    try testing.expectEqualStrings("count", suggest("cont", it).?);
    it = iter(&cands);
    try testing.expectEqualStrings("count", suggest("coutn", it).?); // transposition
}

test "suggest: empty candidate set returns null" {
    const cands = [_][]const u8{};
    try testing.expectEqual(@as(?[]const u8, null), suggest("count", iter(&cands)));
}

test "suggest: short-name gate blocks single/two-char matches" {
    // The resolve.zig `y := x` landmine: target `x` with only `f`/`print` in scope.
    const cands = [_][]const u8{ "f", "print" };
    try testing.expectEqual(@as(?[]const u8, null), suggest("x", iter(&cands)));
    // A short candidate is also skipped even for a long target.
    const cands2 = [_][]const u8{"f"};
    try testing.expectEqual(@as(?[]const u8, null), suggest("foo", iter(&cands2)));
    // `i` vs `j`: both under min_len → no hint.
    const cands3 = [_][]const u8{"j"};
    try testing.expectEqual(@as(?[]const u8, null), suggest("i", iter(&cands3)));
}

test "suggest: a distant name yields no hint" {
    const cands = [_][]const u8{"count"};
    try testing.expectEqual(@as(?[]const u8, null), suggest("zzzzzz", iter(&cands)));
}

test "suggest: an ambiguous tie yields no hint" {
    // `bat` is distance 1 from both `cat` and `bar`; a tie → null.
    const cands = [_][]const u8{ "cat", "bar" };
    try testing.expectEqual(@as(?[]const u8, null), suggest("bat", iter(&cands)));
}

test "suggest: length-fraction gate rejects a big edit on a short name" {
    // distance 2 on a 4-char target: 2*3 = 6 > 4 → rejected even though within cap.
    const cands = [_][]const u8{"abcd"};
    try testing.expectEqual(@as(?[]const u8, null), suggest("axyd", iter(&cands)));
}

test "suggest: a duplicated winner is not a false tie (shadowed local / fn share)" {
    // candidateIter has no string de-dup: a shadowed local (or a local sharing a
    // fn name) yields the same string twice. The second yield is same-distance but
    // is the SAME name, so it must NOT trip the tie guard — the winner still wins.
    const cands = [_][]const u8{ "count", "count", "total" };
    try testing.expectEqualStrings("count", suggest("cont", iter(&cands)).?);
    // Order-independent: winner appearing after an unrelated candidate.
    const cands2 = [_][]const u8{ "total", "count", "count" };
    try testing.expectEqualStrings("count", suggest("cont", iter(&cands2)).?);
    // A genuine distinct tie at the same distance still yields null even with a dup.
    const cands3 = [_][]const u8{ "cat", "cat", "bar" };
    try testing.expectEqual(@as(?[]const u8, null), suggest("bat", iter(&cands3)));
}

test "suggest: the strictly-closer candidate wins over a farther one" {
    const cands = [_][]const u8{ "counter", "count" };
    // target `cont`: dist(count)=1, dist(counter)=3(>cap→null). `count` wins.
    try testing.expectEqualStrings("count", suggest("cont", iter(&cands)).?);
}
