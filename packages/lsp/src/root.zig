//! Language-server surface: a stdio JSON-RPC server that checks toy documents and
//! publishes diagnostics. The server core takes an INJECTABLE reader + writer, so the
//! whole request/response loop is driven in-memory by a test with no subprocess; `toy
//! lsp` wires the same core to real stdin/stdout.

const Server = @import("Server.zig");

pub const name = "toy-lsp";
pub const serve = Server.serve;
pub const Server_ = Server.Server;
pub const ExitCode = Server.ExitCode;

test {
    _ = @import("transport.zig");
    _ = @import("protocol.zig");
    _ = @import("Documents.zig");
    _ = @import("diagnostics.zig");
    _ = @import("hover.zig");
    _ = @import("completion.zig");
    _ = @import("definition.zig");
    _ = @import("signature.zig");
    _ = @import("uri.zig");
    _ = @import("Workspace.zig");
    _ = Server;
}
