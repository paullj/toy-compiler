//! The open-document store: URI -> {full text, version}. `put` is a whole-document
//! replace; `spliceRange` applies one ranged delta in place. Both the key (uri) and value
//! (text) are OWNED copies: the incoming slices borrow the per-message JSON parse arena,
//! which is freed at the end of dispatch, so storing them directly would dangle on the
//! next check. `put` dupes both; replacing frees the old text but keeps the interned key.
//!
//! `put`/`spliceRange` are protocol-exact SYNC only: they make the stored buffer match the
//! client's exact bytes. They do NOT skip the recheck — the caller still re-lexes/re-parses
//! the edited file (its content fingerprint changed); only UNCHANGED modules reuse the
//! query cache.

const std = @import("std");

const Documents = @This();

pub const Doc = struct {
    text: []u8,
    version: i64,
};

map: std.StringHashMapUnmanaged(Doc) = .empty,

/// Insert or replace `uri`'s document. Dupes `text` unconditionally (freeing the prior
/// text on replace) and dupes `uri` only on first insert, so the key stays stable.
pub fn put(self: *Documents, gpa: std.mem.Allocator, uri: []const u8, text: []const u8, version: i64) !void {
    const text_copy = try gpa.dupe(u8, text);
    errdefer gpa.free(text_copy);
    const gop = try self.map.getOrPut(gpa, uri);
    if (gop.found_existing) {
        gpa.free(gop.value_ptr.text);
    } else {
        gop.key_ptr.* = gpa.dupe(u8, uri) catch |e| {
            // Undo the tentative slot so a failed key dupe leaves the map consistent.
            self.map.removeByPtr(gop.key_ptr);
            return e;
        };
    }
    gop.value_ptr.* = .{ .text = text_copy, .version = version };
}

/// Replace the half-open byte range [start,end) of `uri`'s text with `insert`, rebuilding
/// into one fresh allocation and freeing the old (INVALIDATES any prior *Doc.text pointer —
/// re-`get` after each call). No-op if the uri is unknown. Caller guarantees
/// 0 <= start <= end <= text.len (the server clamps). Version is untouched here.
///
/// Protocol-exact incremental SYNC only: it makes the stored buffer match the client's
/// exact bytes. It does NOT skip the recheck — the caller still re-lexes/re-parses this file.
pub fn spliceRange(self: *Documents, gpa: std.mem.Allocator, uri: []const u8, start: usize, end: usize, insert: []const u8) !void {
    const d = self.map.getPtr(uri) orelse return;
    std.debug.assert(start <= end and end <= d.text.len);
    const next = try gpa.alloc(u8, d.text.len - (end - start) + insert.len);
    errdefer gpa.free(next);
    @memcpy(next[0..start], d.text[0..start]);
    @memcpy(next[start..][0..insert.len], insert);
    @memcpy(next[start + insert.len ..], d.text[end..]);
    gpa.free(d.text);
    d.text = next;
}

/// Set the stored version for `uri` (applied once, after a whole didChange change array,
/// since LSP versions the resulting document, not each delta). No-op if absent.
pub fn setVersion(self: *Documents, uri: []const u8, version: i64) void {
    if (self.map.getPtr(uri)) |d| d.version = version;
}

pub fn get(self: *Documents, uri: []const u8) ?*Doc {
    return self.map.getPtr(uri);
}

pub fn remove(self: *Documents, gpa: std.mem.Allocator, uri: []const u8) void {
    if (self.map.fetchRemove(uri)) |kv| {
        gpa.free(kv.key);
        gpa.free(kv.value.text);
    }
}

pub fn deinit(self: *Documents, gpa: std.mem.Allocator) void {
    var it = self.map.iterator();
    while (it.next()) |e| {
        gpa.free(e.key_ptr.*);
        gpa.free(e.value_ptr.text);
    }
    self.map.deinit(gpa);
}

const testing = std.testing;

test "put then get returns the stored text and version" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.put(testing.allocator, "file:///a.toy", "hello", 1);
    const d = docs.get("file:///a.toy").?;
    try testing.expectEqualStrings("hello", d.text);
    try testing.expectEqual(@as(i64, 1), d.version);
}

test "put replaces text and frees the old copy" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.put(testing.allocator, "file:///a.toy", "v1", 1);
    try docs.put(testing.allocator, "file:///a.toy", "v2-longer", 2);
    const d = docs.get("file:///a.toy").?;
    try testing.expectEqualStrings("v2-longer", d.text);
    try testing.expectEqual(@as(i64, 2), d.version);
    try testing.expectEqual(@as(usize, 1), docs.map.count());
}

test "remove drops the entry and frees its memory" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.put(testing.allocator, "file:///a.toy", "x", 1);
    docs.remove(testing.allocator, "file:///a.toy");
    try testing.expect(docs.get("file:///a.toy") == null);
}

test "stored text does not alias the caller's buffer" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    var buf = [_]u8{ 'a', 'b', 'c' };
    try docs.put(testing.allocator, "file:///a.toy", &buf, 1);
    buf[0] = 'z'; // mutate the caller buffer after storing
    try testing.expectEqualStrings("abc", docs.get("file:///a.toy").?.text);
}

test "spliceRange: interior replace" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.put(testing.allocator, "file:///a.toy", "hello world", 1);
    try docs.spliceRange(testing.allocator, "file:///a.toy", 6, 11, "there");
    try testing.expectEqualStrings("hello there", docs.get("file:///a.toy").?.text);
}

test "spliceRange: pure insert, pure delete" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.put(testing.allocator, "file:///a.toy", "abc", 1);
    try docs.spliceRange(testing.allocator, "file:///a.toy", 1, 1, "XY"); // insert
    try testing.expectEqualStrings("aXYbc", docs.get("file:///a.toy").?.text);
    try docs.spliceRange(testing.allocator, "file:///a.toy", 0, 1, ""); // delete first byte
    try testing.expectEqualStrings("XYbc", docs.get("file:///a.toy").?.text);
}

test "spliceRange: two in-order splices, 2nd offset is against post-1st text" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.put(testing.allocator, "file:///a.toy", "abcdef", 1);
    try docs.spliceRange(testing.allocator, "file:///a.toy", 0, 0, "XX"); // -> "XXabcdef"
    // The '2' below only lands on the original 'a' if it is computed against the post-1st text.
    try docs.spliceRange(testing.allocator, "file:///a.toy", 2, 3, "Z"); // "XXabcdef" -> "XXZbcdef"
    try testing.expectEqualStrings("XXZbcdef", docs.get("file:///a.toy").?.text);
}

test "spliceRange: absent uri is a no-op" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.spliceRange(testing.allocator, "file:///missing.toy", 0, 0, "x");
    try testing.expect(docs.get("file:///missing.toy") == null);
}

test "setVersion changes version, not text" {
    var docs: Documents = .{};
    defer docs.deinit(testing.allocator);
    try docs.put(testing.allocator, "file:///a.toy", "body", 1);
    docs.setVersion("file:///a.toy", 7);
    const d = docs.get("file:///a.toy").?;
    try testing.expectEqual(@as(i64, 7), d.version);
    try testing.expectEqualStrings("body", d.text);
}
