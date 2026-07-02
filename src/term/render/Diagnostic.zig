//! Render-side view of the pretty-diagnostic data model. The types now LIVE on the
//! emit side (`diagnostics/model.zig`) so the sink's coded builder can construct
//! `Label`/`Note` slices without a render->diagnostics import inversion; this file is
//! a thin RE-EXPORT so every existing `Rr.Diagnostic.*` call site is unchanged.
//!
//! The old `fromSink` lossy adapter is GONE: the render seam (`DiagRender`) now builds
//! the rich `Diagnostic` inline from the sink POD, threading its `code`/`severity`.

const model = @import("../../diagnostics/model.zig");

pub const Severity = model.Severity;
pub const Span = model.Span;
pub const LabelKind = model.LabelKind;
pub const Label = model.Label;
pub const NoteKind = model.NoteKind;
pub const Note = model.Note;
pub const NO_SOURCE = model.NO_SOURCE;
pub const Diagnostic = model.Diagnostic;
pub const richFromPod = model.richFromPod;

// The model's own unit tests (Span.isZeroWidth, rich literal, defaults, sentinel
// pinning) live in `diagnostics/model.zig` and run there; nothing to re-test here.
