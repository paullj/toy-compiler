//! The open-document store: URI -> {full text, version}. Full-text sync only, so a
//! `put` simply replaces the stored text. Both the key (uri) and value (text) are
//! OWNED copies: the incoming slices borrow the per-message JSON parse arena, which is
//! freed at the end of dispatch, so storing them directly would dangle on the next
//! check. `put` dupes both; replacing frees the old text but keeps the interned key.

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
