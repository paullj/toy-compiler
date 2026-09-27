//! `--timings` per-stage accumulator, split into the three costs a cache-backed stage
//! divides into: COMPUTE (the miss path — `lowerOne` for codegen, file-read+tokenize
//! for lex, the AST build for parse), cache GET I/O (the hit-or-miss read), and cache
//! PUT I/O (the atomic-rename temp-file write on a miss). Atomic because a stage's
//! fan-out (codegen, and discovery's per-module queries) runs jobs in parallel and
//! each adds its own deltas. BORROWED: one probe per stage lives on the
//! driver frame and a `*StageProbe` is threaded into every job. `null` (the default)
//! is zero-overhead — no clock is read.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

const StageProbe = @This();

compute_ns: std.atomic.Value(u64) = .init(0),
get_ns: std.atomic.Value(u64) = .init(0),
put_ns: std.atomic.Value(u64) = .init(0),

fn add(field: *std.atomic.Value(u64), dt: u64) void {
    // wasm32 has no 64-bit atomics, and a single-threaded build needs none.
    if (builtin.single_threaded) {
        field.raw +%= dt;
        return;
    }
    _ = field.fetchAdd(dt, .monotonic);
}

/// Charge `now - start` to the COMPUTE bucket from OUTSIDE the query path — for a
/// stage's miss-side work that is not itself a cached query (e.g. discovery's
/// `readFileAlloc`, which is always paid and is the file-read part of the
/// "file-read+lex+parse" discover compute). Reads the clock once; callers gate the
/// call on the probe being present so a plain build pays nothing.
pub fn lapCompute(self: *StageProbe, io: Io, start: i128) void {
    lap(io, &self.compute_ns, start);
}

/// Charge `now - start` to the cache-GET bucket from OUTSIDE the query path — for a
/// cached serve that doesn't go through `Engine.query`/`lex`/`parse` (the warm-discover
/// fast path's direct `cache.get`s of the source/lex/parse blobs). Mirrors `lapCompute`;
/// callers gate on the probe being present so a plain build reads no clock.
pub fn lapGet(self: *StageProbe, io: Io, start: i128) void {
    lap(io, &self.get_ns, start);
}

/// The monotonic clock the probe charges against (`Io.Clock`, since Zig 0.16 has no
/// `std.time.Timer`). Exposed so an out-of-query caller can snapshot a start stamp.
pub fn now(io: Io) i128 {
    return Io.Clock.Timestamp.now(io, .awake).raw.nanoseconds;
}

/// Charge `dt = now - start` to `field`, but only when a probe is present. Reading
/// the clock per get/put/compute is cheap relative to the syscalls they bracket, and
/// it is only paid under `--timings` (probe != null), so a plain build is unaffected.
pub fn lap(io: Io, field: *std.atomic.Value(u64), start: i128) void {
    const dt = now(io) - start;
    if (dt > 0) add(field, @intCast(dt));
}
