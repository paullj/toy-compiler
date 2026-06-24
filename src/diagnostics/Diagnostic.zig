//! The one diagnostic type shared by every stage. `byte_offset` points at the
//! offending token; the driver/CLI renders byte_offset -> line:col uniformly.
pub const Diagnostic = struct {
    byte_offset: u32,
    message: []const u8,
};
