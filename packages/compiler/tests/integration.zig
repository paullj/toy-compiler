//! Integration test root — the repo-root `tests/` suite.
//!
//! These tests consume the compiler as a BLACK BOX through the published
//! `toy_compiler` module (src/root.zig's `pub const` exports), via
//! `@import("toy_compiler")` — never `../` into src/ (a Zig module cannot import
//! above its root source file). build.zig gives this artifact the compiler as a
//! module dependency (`toy-integration-test`), so these run in their OWN binary,
//! separate from the inline unit tests in `toy-test`. The idiomatic split: src/
//! holds the library + its inline unit tests; tests/ holds integration tests that
//! exercise the published surface.
//!
//! Run under `zig build test`, or `zig build test-bin` then
//! `./zig-out/bin/toy-integration-test` to run the installed binary directly.

test {
    _ = @import("query_engine.zig");
    _ = @import("driver.zig");
    _ = @import("check.zig");
    _ = @import("differential.zig");
    _ = @import("corpus.zig");
    _ = @import("heap.zig");
    _ = @import("ref.zig");
    _ = @import("vec.zig");
    _ = @import("desc.zig");
    _ = @import("map.zig");
    _ = @import("set.zig");
    _ = @import("string.zig");
    _ = @import("math.zig");
    _ = @import("unit.zig");
}
