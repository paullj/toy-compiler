//! Minimal `file://` <-> filesystem-path conversion for the LSP dispatch layer.
//!
//! Scope is deliberately narrow: `file://` + an absolute path only. There is no host
//! component (`file://host/…`), no `file://localhost/` special case, and no Windows
//! drive-letter handling — the editors this server targets emit `file:///abs/path`.
//!
//! Asymmetric percent handling by design: INBOUND (`uriToPath`) decodes `%XX` escapes so a
//! path with a space (`%20`) round-trips to real bytes; OUTBOUND (`pathToUri`) concatenates
//! the path verbatim and does NOT percent-encode. This is sound for the ASCII project paths
//! we emit Locations for; a path containing a space or `%` would produce a technically
//! non-encoded uri. Kept simple until a real need forces encoding.

const std = @import("std");

/// `file://` + `path` verbatim. Caller owns the returned bytes.
pub fn pathToUri(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa, "file://{s}", .{path});
}

/// The filesystem path of a `file://` uri (percent-decoded), or null if `uri` is not a
/// `file://` uri (e.g. `untitled:`, `http://`). Caller owns the returned bytes when non-null.
pub fn uriToPath(gpa: std.mem.Allocator, uri: []const u8) !?[]u8 {
    const prefix = "file://";
    if (!std.mem.startsWith(u8, uri, prefix)) return null;
    const enc = uri[prefix.len..];

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var i: usize = 0;
    while (i < enc.len) {
        if (enc[i] == '%' and i + 2 < enc.len) {
            if (hexNibble(enc[i + 1])) |hi| {
                if (hexNibble(enc[i + 2])) |lo| {
                    try out.append(gpa, hi << 4 | lo);
                    i += 3;
                    continue;
                }
            }
        }
        try out.append(gpa, enc[i]);
        i += 1;
    }
    return try out.toOwnedSlice(gpa);
}

/// The directory of a `file://` uri's path, or null if `uri` is not a file uri or has no
/// dirname. Caller owns the returned bytes when non-null.
pub fn dirOfUri(gpa: std.mem.Allocator, uri: []const u8) !?[]u8 {
    const p = (try uriToPath(gpa, uri)) orelse return null;
    defer gpa.free(p);
    const d = std.fs.path.dirname(p) orelse return null;
    return try gpa.dupe(u8, d);
}

fn hexNibble(c: u8) ?u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => null,
    };
}

const testing = std.testing;

test "pathToUri prefixes file:// verbatim" {
    const gpa = testing.allocator;
    const u = try pathToUri(gpa, "/a/b.toy");
    defer gpa.free(u);
    try testing.expectEqualStrings("file:///a/b.toy", u);
}

test "uriToPath decodes %XX escapes" {
    const gpa = testing.allocator;
    const p = (try uriToPath(gpa, "file:///a%20b/c.toy")).?;
    defer gpa.free(p);
    try testing.expectEqualStrings("/a b/c.toy", p);
}

test "uriToPath rejects non-file schemes" {
    const gpa = testing.allocator;
    try testing.expect((try uriToPath(gpa, "untitled:x")) == null);
    try testing.expect((try uriToPath(gpa, "http://x")) == null);
}

test "dirOfUri yields the parent dir of a file uri" {
    const gpa = testing.allocator;
    const d = (try dirOfUri(gpa, "file:///a/b.toy")).?;
    defer gpa.free(d);
    try testing.expectEqualStrings("/a", d);
}

test "dirOfUri rejects a non-file uri" {
    const gpa = testing.allocator;
    try testing.expect((try dirOfUri(gpa, "untitled:x")) == null);
}
