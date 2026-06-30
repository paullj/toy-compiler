//! Facade for the CLI builder: re-exports the schema, parser, typed result, errors, and help.

pub const Spec = @import("Spec.zig");
pub const Parsed = @import("Parsed.zig");
pub const Parser = @import("Parser.zig");
pub const Sink = @import("Sink.zig");
pub const Help = @import("Help.zig");
