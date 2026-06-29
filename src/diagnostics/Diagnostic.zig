//! The one diagnostic type shared by every stage, re-exported from the sink that
//! owns it. `byte_offset` points at the offending token; the driver/CLI renders
//! byte_offset -> line:col uniformly. `scope` carries the owning module id in
//! graph mode (`NO_SCOPE` single-file).
pub const Diagnostic = @import("Sink.zig").Diagnostic;
pub const NO_SCOPE = @import("Sink.zig").NO_SCOPE;
