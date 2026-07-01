//! Exit-free CLI error accumulator: the parser feeds parse errors in as it walks
//! argv; rendering and process exit happen elsewhere, so this leaf pulls in only
//! `std`. Every `Error` string is BORROWED (a slice into argv or a static
//! literal), so the Sink allocates none and argv must outlive it.

const std = @import("std");

const Sink = @This();

/// Unmanaged accumulator: the backing list starts `.empty` and every method that
/// can allocate takes the gpa per call, so a caller can hold a Sink by value with
/// no init step (`var s: Sink = .{};`).
list: std.ArrayList(Error) = .empty,

pub const Kind = enum {
    unknown_flag,
    missing_value,
    bad_value,
    missing_required,
    unexpected_arg,
    conflict,
    unmet_requirement,
};

/// One accumulated CLI error. All string fields are borrowed (see the file
/// header): they point into argv or static literals owned by the caller, never
/// into Sink-allocated memory.
pub const Error = struct {
    kind: Kind,
    arg: []const u8 = "",
    got: []const u8 = "",
    expected: []const u8 = "",
    where: []const u8 = "",
};

/// Append preserving insertion order. Errors surface in the sequence the parser
/// encountered them, which is the order the caller will render.
pub fn add(self: *Sink, gpa: std.mem.Allocator, e: Error) !void {
    try self.list.append(gpa, e);
}

/// Free the backing list exactly once. There are no owned strings to free —
/// every `Error` field is borrowed.
pub fn deinit(self: *Sink, gpa: std.mem.Allocator) void {
    self.list.deinit(gpa);
    self.* = undefined;
}

/// Drop all accumulated errors while keeping the allocation, so a reused Sink
/// avoids a re-grow. No free — nothing here owns heap beyond the list itself.
pub fn reset(self: *Sink) void {
    self.list.clearRetainingCapacity();
}

pub fn count(self: Sink) usize {
    return self.list.items.len;
}

pub fn items(self: Sink) []const Error {
    return self.list.items;
}

pub fn isEmpty(self: Sink) bool {
    return self.list.items.len == 0;
}

// ---- tests -----------------------------------------------------------------

const testing = std.testing;

test "add preserves insertion order and count tracks items" {
    var s: Sink = .{};
    defer s.deinit(testing.allocator);

    try testing.expect(s.isEmpty());
    try testing.expectEqual(@as(usize, 0), s.count());

    try s.add(testing.allocator, .{ .kind = .unknown_flag, .arg = "--frob" });
    try s.add(testing.allocator, .{ .kind = .missing_value, .arg = "--out", .expected = "path" });
    try s.add(testing.allocator, .{ .kind = .bad_value, .arg = "--level", .got = "high", .expected = "int" });

    try testing.expect(!s.isEmpty());
    try testing.expectEqual(@as(usize, 3), s.count());

    const got = s.items();
    try testing.expectEqual(Kind.unknown_flag, got[0].kind);
    try testing.expectEqualStrings("--frob", got[0].arg);
    try testing.expectEqual(Kind.missing_value, got[1].kind);
    try testing.expectEqualStrings("--out", got[1].arg);
    try testing.expectEqualStrings("path", got[1].expected);
    try testing.expectEqual(Kind.bad_value, got[2].kind);
    try testing.expectEqualStrings("high", got[2].got);
    try testing.expectEqualStrings("int", got[2].expected);
}

test "reset drops count to zero without freeing" {
    var s: Sink = .{};
    defer s.deinit(testing.allocator);

    try s.add(testing.allocator, .{ .kind = .unexpected_arg, .arg = "extra" });
    try s.add(testing.allocator, .{ .kind = .conflict, .arg = "--a", .where = "--b" });
    try testing.expectEqual(@as(usize, 2), s.count());

    s.reset();
    try testing.expect(s.isEmpty());
    try testing.expectEqual(@as(usize, 0), s.count());

    // Reuse after reset still works and re-orders from empty.
    try s.add(testing.allocator, .{ .kind = .unmet_requirement, .arg = "--x", .where = "--y" });
    try testing.expectEqual(@as(usize, 1), s.count());
    try testing.expectEqual(Kind.unmet_requirement, s.items()[0].kind);
}

test "add then deinit leaks nothing under the testing allocator" {
    var s: Sink = .{};
    try s.add(testing.allocator, .{ .kind = .missing_required, .arg = "--input" });
    try s.add(testing.allocator, .{ .kind = .unknown_flag, .arg = "--nope" });
    s.deinit(testing.allocator); // testing.allocator fails the test on any leak.
}

test "failing allocator add returns OutOfMemory without corrupting state" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var s: Sink = .{};
    defer s.deinit(failing.allocator());

    try testing.expectError(error.OutOfMemory, s.add(failing.allocator(), .{ .kind = .bad_value, .arg = "--z" }));
    try testing.expect(s.isEmpty());
    try testing.expectEqual(@as(usize, 0), s.count());
}
