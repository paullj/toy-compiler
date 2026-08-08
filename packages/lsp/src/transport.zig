//! LSP base-protocol framing: `Content-Length: N\r\n\r\n<body>` over an INJECTABLE
//! `std.Io.Reader` / `std.Io.Writer`. Keeping the reader/writer injectable is what lets
//! an in-process test drive the whole loop over in-memory buffers with no subprocess.

const std = @import("std");
const Reader = std.Io.Reader;
const Writer = std.Io.Writer;

pub const Error = error{
    /// The stream ended after a message had already begun (headers consumed but no body).
    UnexpectedEof,
    /// Headers ended without a `Content-Length`.
    MissingContentLength,
    /// A header line was malformed (a non-numeric `Content-Length`).
    InvalidHeader,
    /// A header line exceeded the reader's buffer with no `\n` in sight.
    StreamTooLong,
};

pub const ReadError = Error || error{ EndOfStream, ReadFailed, OutOfMemory };

/// Read one framed message body (heap-owned; caller frees). Returns `error.EndOfStream`
/// ONLY at a clean message boundary (no header bytes consumed yet) so the run loop can
/// stop cleanly on a pipe close; a stream that dies mid-message is `error.UnexpectedEof`.
pub fn readMessage(gpa: std.mem.Allocator, r: *Reader) ReadError![]u8 {
    var content_len: ?usize = null;
    var saw_any = false;
    while (true) {
        const line = r.takeDelimiterInclusive('\n') catch |e| switch (e) {
            // No delimiter + stream end: a clean boundary iff we have not started a message.
            error.EndOfStream => return if (saw_any) error.UnexpectedEof else error.EndOfStream,
            error.StreamTooLong => return error.StreamTooLong,
            error.ReadFailed => return error.ReadFailed,
        };
        saw_any = true;
        const trimmed = std.mem.trimEnd(u8, line, "\r\n");
        if (trimmed.len == 0) break; // the blank line terminates the header block
        // Parse the length NOW into a usize: `line` is a slice INTO the reader buffer,
        // invalidated by the next read, so nothing may hold it across the next iteration.
        if (headerNameIs(trimmed, "content-length")) {
            const v = std.mem.trim(u8, trimmed["content-length".len + 1 ..], " \t");
            content_len = std.fmt.parseInt(usize, v, 10) catch return error.InvalidHeader;
        }
        // Any other header (Content-Type, ...) is ignored.
    }
    const len = content_len orelse return error.MissingContentLength;
    const body = try gpa.alloc(u8, len);
    errdefer gpa.free(body);
    // Fill a heap buffer with `readSliceAll`, never `take(n)`: `take` caps at the reader's
    // internal buffer and would `StreamTooLong` on a body larger than it, so a big document
    // would break. A short read here means the final frame was truncated.
    r.readSliceAll(body) catch |e| switch (e) {
        error.EndOfStream => return error.UnexpectedEof,
        error.ReadFailed => return error.ReadFailed,
    };
    return body;
}

/// Write one framed message. Flushes per message so an in-process `Writer.Allocating`
/// (and a real client) observes each response immediately rather than at loop end.
pub fn writeMessage(w: *Writer, body: []const u8) Writer.Error!void {
    try w.print("Content-Length: {d}\r\n\r\n", .{body.len});
    try w.writeAll(body);
    try w.flush();
}

/// Case-insensitive `<name>:` header-line match. `name` is expected lowercase.
fn headerNameIs(line: []const u8, comptime name: []const u8) bool {
    if (line.len < name.len + 1) return false;
    if (!std.ascii.eqlIgnoreCase(line[0..name.len], name)) return false;
    return line[name.len] == ':';
}

const testing = std.testing;

test "readMessage: two concatenated frames decode to two bodies" {
    const raw = "Content-Length: 2\r\n\r\nhiContent-Length: 5\r\n\r\nworld";
    var r = Reader.fixed(raw);
    const a = try readMessage(testing.allocator, &r);
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("hi", a);
    const b = try readMessage(testing.allocator, &r);
    defer testing.allocator.free(b);
    try testing.expectEqualStrings("world", b);
    // A third read at the clean boundary is EndOfStream.
    try testing.expectError(error.EndOfStream, readMessage(testing.allocator, &r));
}

test "readMessage: Content-Type and unknown headers are ignored; case-insensitive length" {
    const raw = "content-length: 3\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\nabc";
    var r = Reader.fixed(raw);
    const a = try readMessage(testing.allocator, &r);
    defer testing.allocator.free(a);
    try testing.expectEqualStrings("abc", a);
}

test "readMessage: empty input is a clean EndOfStream" {
    var r = Reader.fixed("");
    try testing.expectError(error.EndOfStream, readMessage(testing.allocator, &r));
}

test "readMessage: headers then a truncated body is UnexpectedEof" {
    const raw = "Content-Length: 10\r\n\r\nshort";
    var r = Reader.fixed(raw);
    try testing.expectError(error.UnexpectedEof, readMessage(testing.allocator, &r));
}

test "readMessage: headers with no Content-Length is an error" {
    const raw = "X-Foo: bar\r\n\r\n";
    var r = Reader.fixed(raw);
    try testing.expectError(error.MissingContentLength, readMessage(testing.allocator, &r));
}

test "readMessage: non-numeric Content-Length is an InvalidHeader" {
    const raw = "Content-Length: abc\r\n\r\n";
    var r = Reader.fixed(raw);
    try testing.expectError(error.InvalidHeader, readMessage(testing.allocator, &r));
}

test "round-trip: writeMessage then readMessage yields identical bytes" {
    var aw: Writer.Allocating = .init(testing.allocator);
    defer aw.deinit();
    try writeMessage(&aw.writer, "{\"jsonrpc\":\"2.0\"}");
    var r = Reader.fixed(aw.written());
    const body = try readMessage(testing.allocator, &r);
    defer testing.allocator.free(body);
    try testing.expectEqualStrings("{\"jsonrpc\":\"2.0\"}", body);
}
