//! The language server as a wasm32-wasi reactor for a browser worker. The host pushes
//! framed JSON-RPC bytes in and reads the framed replies back; there is no stdio loop and
//! no filesystem (imports resolve only against open documents).
//!
//! Host protocol: `toy_lsp_alloc(n)` for an input buffer, write the bytes, then
//! `toy_lsp_feed(ptr, n)` (which frees the buffer) returns the reply length or -1 on a
//! fatal fault; the reply is at `toy_lsp_output()` until the next feed.

const std = @import("std");
const Server = @import("Server.zig").Server;

const gpa = std.heap.wasm_allocator;

/// Safety checks stay on in this build, so a latent bug traps instead of running on in
/// undefined behaviour; the host catches the trap and restarts the server. A trap is all
/// the host can observe, so the message and stack-trace machinery would be dead weight.
pub const panic = std.debug.FullPanic(struct {
    fn call(_: []const u8, _: ?usize) noreturn {
        @trap();
    }
}.call);

var server: Server = undefined;
var started = false;
var out: std.Io.Writer.Allocating = .init(gpa);

export fn toy_lsp_alloc(len: usize) ?[*]u8 {
    const buf = gpa.alloc(u8, len) catch return null;
    return buf.ptr;
}

export fn toy_lsp_feed(ptr: [*]u8, len: usize) i32 {
    defer gpa.free(ptr[0..len]);
    if (!started) {
        server = Server.init(gpa);
        started = true;
    }
    out.clearRetainingCapacity();
    _ = server.feed(ptr[0..len], &out.writer) catch return -1;
    return @intCast(out.written().len);
}

export fn toy_lsp_output() [*]const u8 {
    return out.written().ptr;
}
