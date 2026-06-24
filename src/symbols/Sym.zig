//! Stable symbol identity shared across resolve/types/codegen/link.
//!
//! WHY this lives in `symbols/` (a peer data module) and not in `link/`: a
//! `SymName` is the source-position-INDEPENDENT identity of a definition — what
//! a cached reloc names, what the fingerprint folds, what the linker interns. It
//! is consumed by several stages; making it a peer data module lets each stage
//! depend on the DATA rather than sideways on the stage that happened to define
//! it first.

const std = @import("std");

/// The kind of a stable symbol identity. The tag makes collision-freedom
/// structural: a user fn "f", a builtin "f", and an import "f" are distinct
/// without sigils.
pub const SymKind = enum(u8) { user_fn, builtin, import };

/// Stable, source-position-INDEPENDENT identity of a definition. This is what a
/// cached reloc names — never a source index (indices shift under reorder/insert,
/// so an index in a cache hit would call the wrong code). PERSISTED on disk; the
/// `name` is the sigil-free internal key ("main"/"print"/"write"). dyld-facing
/// linkage names (_main, _write) are DERIVED from this at emit. In-session the
/// linker assigns a dense u32 handle via `SymInterner` for the hot path.
///
/// OWNERSHIP: borrowed from the source while a `FnCode` is freshly lowered in the
/// same session; OWNED (heap-duped) on the unpacked-from-disk path. The
/// "always own" rule (every fresh `FnCode` dupes its names too) makes `deinit`
/// uniform — see `FnCode.deinit`.
pub const SymName = struct {
    kind: SymKind,
    name: []const u8,

    pub fn eql(a: SymName, b: SymName) bool {
        return a.kind == b.kind and std.mem.eql(u8, a.name, b.name);
    }
};
