//! In-process front-end fuzzer (`zig build fuzz`).
//!
//! Drives many random inputs through the SAME front-end `build`/`check` run —
//! lex → `Parser.parse`, and on a clean parse also resolve + typecheck the
//! single-file way (`Graph.single`) — and enforces the robustness contract:
//! the front-end NEVER panics, NEVER hangs (the fuel guards bound every
//! recovery loop), and ALWAYS returns a tree. The parser's own invariants (span
//! totality, structural pairing, forward progress) fire automatically on every
//! `Parser.parse` under `std.debug.runtime_safety` — this driver's job is only to
//! DRIVE inputs and let those asserts (and any panic/hang) surface, NOT to
//! re-implement the invariants. The build wires this exe in Debug so the asserts
//! are ON (a ReleaseFast build would silently disable them and make the run a
//! no-op, so the build forces Debug regardless of `-Doptimize`).
//!
//! Inputs come from two sources, both driven by a SEEDED PRNG so a run is exactly
//! reproducible: (a) MUTATION — load the seed corpus (`tests/ui/**/*.toy` +
//! `examples/**/*.toy`) and apply random byte edits (bit/byte flip, insert,
//! delete, truncate, splice); (b) a small GRAMMAR-AWARE generator that emits
//! random-but-plausible toy programs (fn decls, precedence-climbing exprs over the
//! real operators, stmts, if/loop/match, struct/enum, imports).
//!
//! Determinism: the seed comes from `FUZZ_SEED` (else a fixed default), NEVER from
//! wall-clock. Iterations come from `FUZZ_ITERS` (else a bounded default of 5000 so
//! `zig build fuzz` finishes in a few seconds and is CI-runnable). On any failure
//! (assert/panic) the installed panic handler prints the offending input bytes +
//! the seed so the exact failure reproduces with `FUZZ_SEED=<seed>`.

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");

const Lexer = toyc.Lexer;
const Parser = toyc.Parser;
const Ast = toyc.Ast;
const Graph = toyc.Graph;
const ResolveGraph = toyc.ResolveGraph;
const TypecheckGraph = toyc.TypecheckGraph;

/// A fixed, arbitrary default seed (NOT wall-clock) so an unconfigured
/// `zig build fuzz` is reproducible run-to-run.
const default_seed: u64 = 0x0123_4567_89ab_cdef;
/// Bounded default iteration count: small enough that `zig build fuzz` finishes in
/// a few seconds (CI-runnable), overridable via `FUZZ_ITERS` for long soak runs.
const default_iters: usize = 5000;

// A parser-invariant assert (or any other panic) aborts the process; before it
// does, print the exact bytes + seed that were in flight so the failure
// reproduces deterministically. The globals are set right before each front-end
// call and cleared after, so the handler only prints when a call was live.

var cur_seed: u64 = default_seed;
var cur_input: ?[]const u8 = null;
var cur_kind: []const u8 = "none";

fn fuzzPanic(msg: []const u8, ra: ?usize) noreturn {
    std.debug.print(
        \\
        \\==== FUZZ FAILURE ====
        \\reproduce with: FUZZ_SEED={d} zig build fuzz
        \\input kind: {s}
        \\
    , .{ cur_seed, cur_kind });
    if (cur_input) |input| {
        // Print the raw bytes (they may be non-UTF-8 garbage) then a hex dump so the
        // exact input is recoverable regardless of encoding.
        std.debug.print("input bytes ({d}):\n{s}\n---- hex ----\n", .{ input.len, input });
        for (input, 0..) |b, i| {
            std.debug.print("{x:0>2} ", .{b});
            if (i % 16 == 15) std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }
    // Delegate to the default panic so we still get the assert message + trace and
    // a nonzero exit.
    std.debug.defaultPanic(msg, ra);
}

/// Override the root panic handler so an invariant assert dumps the reproducer.
pub const panic = std.debug.FullPanic(fuzzPanic);

pub fn main(init: std.process.Init) !void {
    // The lock-free per-CPU allocator the driver uses; the resolve/typecheck paths
    // may touch the pool, and this avoids the DebugAllocator lock contention.
    const gpa = std.heap.smp_allocator;
    const io = init.io;
    const env = init.environ_map;

    const seed = envU64(env, "FUZZ_SEED") orelse default_seed;
    const iters = envUsize(env, "FUZZ_ITERS") orelse default_iters;
    cur_seed = seed;

    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    // Load the seed corpus (all tracked .toy files under tests/ui + examples). A
    // missing corpus directory is tolerated — the generator alone still exercises
    // the front-end — but we surface a note so a mis-run cwd is visible.
    var corpus = try loadCorpus(gpa, io);
    defer {
        for (corpus.items) |c| gpa.free(c);
        corpus.deinit(gpa);
    }

    std.debug.print("fuzz: seed={d} iters={d} corpus={d} files\n", .{ seed, iters, corpus.items.len });

    var mutated: usize = 0;
    var generated: usize = 0;
    var clean_parses: usize = 0;

    var scratch: std.ArrayList(u8) = .empty;
    defer scratch.deinit(gpa);

    var i: usize = 0;
    while (i < iters) : (i += 1) {
        scratch.clearRetainingCapacity();
        // Split roughly half mutation / half generation. When the corpus is empty,
        // fall back to generation so we always feed SOMETHING.
        const use_mutation = corpus.items.len > 0 and rand.boolean();
        if (use_mutation) {
            const base = corpus.items[rand.uintLessThan(usize, corpus.items.len)];
            try mutate(gpa, &scratch, base, corpus.items, rand);
            cur_kind = "mutation";
            mutated += 1;
        } else {
            try generate(gpa, &scratch, rand, 0);
            cur_kind = "generation";
            generated += 1;
        }

        if (try drive(gpa, scratch.items)) clean_parses += 1;
    }

    cur_input = null;
    cur_kind = "none";

    std.debug.print(
        "fuzz: ran {d} inputs ({d} mutated, {d} generated); {d} parsed clean; no panic/assert/hang\n",
        .{ iters, mutated, generated, clean_parses },
    );
}

/// Run ONE input through the front-end. Returns true if the parse was clean (so we
/// also ran resolve+typecheck). A malformed PROGRAM is never an error — the front-
/// end must always return a tree/diagnostics. The only errors that propagate are
/// genuine host-resource failures (`OutOfMemory`) or an unexpected resolve/typecheck
/// error, both of which are real failures the fuzzer should surface (nonzero exit).
/// Records the input for the panic handler.
fn drive(gpa: std.mem.Allocator, input: []const u8) !bool {
    cur_input = input;
    defer cur_input = null;

    // lex: enforces span totality by construction under runtime_safety.
    const tokens = try Lexer.tokenize(gpa, input);
    defer gpa.free(tokens);

    // parse: ALWAYS returns a tree; under runtime_safety `checkInvariants` fires
    // here (span totality, bracket pairing, forward progress). The fuel guards keep
    // every recovery loop bounded, so this call terminates — a hang here would be a
    // real bug (a missing fuel guard), surfaced as a timeout by the bounded loop.
    const res = try Parser.parse(gpa, tokens, input);
    defer gpa.free(@constCast(res.diags));
    defer gpa.free(res.tree.nodes);
    defer gpa.free(res.tree.extra);
    defer if (res.tree.pub_bits.len != 0) gpa.free(@constCast(res.tree.pub_bits));

    // A tainted parse (any diagnostic) stops at parse — exactly as the driver
    // pipeline does (a poisoned tree must not reach resolve/typecheck).
    if (res.diags.len > 0) return false;

    // Clean parse: drive the single-file resolve+typecheck path the same way
    // `Driver.pipeline(.check)` does — wrap the borrowed tree in a graph-of-one and
    // run the whole-graph front-end. This must also never panic/hang on any input
    // the parser accepted.
    var graph = try Graph.single(gpa, "main", "fuzz", input, tokens, res.tree.nodes, res.tree.extra, res.tree.pub_bits);
    defer graph.deinit(gpa);

    var resolve = try ResolveGraph.resolveGraph(gpa, &graph);
    defer resolve.deinit(gpa);
    if (resolve.diags.len > 0) return true; // a name error stops before typecheck

    var tc = try TypecheckGraph.checkGraph(gpa, &graph, &resolve, null, 0);
    tc.deinit(gpa);
    return true;
}

/// Walk `tests/ui` and `examples` for `*.toy` files, returning their contents.
/// Each entry is owned (freed by the caller). Empty files are kept (a valid, if
/// degenerate, seed). A missing directory is skipped silently.
fn loadCorpus(gpa: std.mem.Allocator, io: Io) !std.ArrayList([]const u8) {
    var out: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (out.items) |c| gpa.free(c);
        out.deinit(gpa);
    }
    for ([_][]const u8{ "tests/ui", "examples" }) |root| {
        var dir = Io.Dir.cwd().openDir(io, root, .{ .iterate = true }) catch continue;
        defer dir.close(io);
        var walker = try dir.walk(gpa);
        defer walker.deinit();
        while (try walker.next(io)) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.basename, ".toy")) continue;
            const bytes = dir.readFileAlloc(io, entry.path, gpa, .unlimited) catch continue;
            try out.append(gpa, bytes);
        }
    }
    return out;
}

/// The maximum size a mutated input may reach; keeps a pathological insert/splice
/// chain from ballooning the working buffer (and legitimately OOMing the host).
const max_mutation_len: usize = 64 * 1024;

/// Copy `base` into `out` and apply 1..8 random byte edits. Edits: bit flip, byte
/// flip, byte insert, byte delete, truncate, splice-in a chunk of another seed.
fn mutate(
    gpa: std.mem.Allocator,
    out: *std.ArrayList(u8),
    base: []const u8,
    corpus: []const []const u8,
    rand: std.Random,
) !void {
    try out.appendSlice(gpa, base);
    const rounds = rand.intRangeAtMost(usize, 1, 8);
    var r: usize = 0;
    while (r < rounds) : (r += 1) {
        // With an empty buffer only insert/splice can make progress.
        const strategy = if (out.items.len == 0)
            rand.intRangeAtMost(u8, 2, 5)
        else
            rand.intRangeAtMost(u8, 0, 5);
        switch (strategy) {
            0 => { // bit flip
                const idx = rand.uintLessThan(usize, out.items.len);
                out.items[idx] ^= @as(u8, 1) << rand.intRangeAtMost(u3, 0, 7);
            },
            1 => { // byte flip (replace with a random byte)
                const idx = rand.uintLessThan(usize, out.items.len);
                out.items[idx] = rand.int(u8);
            },
            2 => { // byte insert
                if (out.items.len < max_mutation_len) {
                    const idx = if (out.items.len == 0) 0 else rand.uintLessThan(usize, out.items.len);
                    try out.insert(gpa, idx, randByte(rand));
                }
            },
            3 => { // byte delete
                if (out.items.len > 0) {
                    const idx = rand.uintLessThan(usize, out.items.len);
                    _ = out.orderedRemove(idx);
                }
            },
            4 => { // truncate
                if (out.items.len > 0) {
                    const keep = rand.uintLessThan(usize, out.items.len);
                    out.shrinkRetainingCapacity(keep);
                }
            },
            5 => { // splice: insert a chunk of another seed
                if (corpus.len > 0 and out.items.len < max_mutation_len) {
                    const donor = corpus[rand.uintLessThan(usize, corpus.len)];
                    if (donor.len > 0) {
                        const start = rand.uintLessThan(usize, donor.len);
                        const end = rand.intRangeAtMost(usize, start, donor.len);
                        const chunk = donor[start..end];
                        const at = if (out.items.len == 0) 0 else rand.uintLessThan(usize, out.items.len);
                        try out.insertSlice(gpa, at, chunk);
                        if (out.items.len > max_mutation_len) out.shrinkRetainingCapacity(max_mutation_len);
                    }
                }
            },
            else => unreachable,
        }
    }
}

/// A byte biased toward source-relevant characters (so inserts land near real
/// tokens more often than pure noise would), with a tail of fully random bytes.
fn randByte(rand: std.Random) u8 {
    const interesting = "(){}[]<>+-*/=!,.:;\n \t\"abcdefgh0123_";
    if (rand.boolean()) return interesting[rand.uintLessThan(usize, interesting.len)];
    return rand.int(u8);
}

const max_gen_depth: usize = 4;
const idents = [_][]const u8{ "a", "b", "c", "x", "y", "foo", "bar", "n", "acc" };
const types = [_][]const u8{ "int", "bool", "str", "()" };
const bin_ops = [_][]const u8{ "+", "-", "*", "/", "==", "!=", "<", ">", "<=", ">=", "and", "or" };

/// Emit a random-but-plausible program: 1..4 top-level items (fn/struct/enum/
/// import). At least one `fn main`-like function so the whole-program check has an
/// entry to reason about. Not guaranteed to be VALID — the point is to feed the
/// front-end plausible shapes that stress recovery + resolution.
fn generate(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random, depth: usize) error{OutOfMemory}!void {
    const items = rand.intRangeAtMost(usize, 1, 4);
    var i: usize = 0;
    while (i < items) : (i += 1) {
        switch (rand.uintLessThan(u8, 5)) {
            0 => try genImport(gpa, out, rand),
            1 => try genStruct(gpa, out, rand),
            2 => try genEnum(gpa, out, rand),
            else => try genFn(gpa, out, rand),
        }
        try out.append(gpa, '\n');
    }
    _ = depth;
}

fn genImport(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random) error{OutOfMemory}!void {
    try out.appendSlice(gpa, "import ");
    try out.appendSlice(gpa, pick(idents, rand));
    if (rand.boolean()) {
        try out.appendSlice(gpa, " as ");
        try out.appendSlice(gpa, pick(idents, rand));
    }
}

fn genStruct(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random) error{OutOfMemory}!void {
    if (rand.boolean()) try out.appendSlice(gpa, "pub ");
    try out.appendSlice(gpa, "struct ");
    try out.appendSlice(gpa, pick(idents, rand));
    try out.appendSlice(gpa, " { ");
    const fields = rand.intRangeAtMost(usize, 0, 3);
    var f: usize = 0;
    while (f < fields) : (f += 1) {
        if (f > 0) try out.appendSlice(gpa, ", ");
        try out.appendSlice(gpa, pick(idents, rand));
        try out.appendSlice(gpa, ": ");
        try out.appendSlice(gpa, pick(types, rand));
    }
    try out.appendSlice(gpa, " }");
}

fn genEnum(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random) error{OutOfMemory}!void {
    if (rand.boolean()) try out.appendSlice(gpa, "pub ");
    try out.appendSlice(gpa, "enum ");
    try out.appendSlice(gpa, pick(idents, rand));
    try out.appendSlice(gpa, " { ");
    const variants = rand.intRangeAtMost(usize, 1, 3);
    var v: usize = 0;
    while (v < variants) : (v += 1) {
        if (v > 0) try out.appendSlice(gpa, ", ");
        // Variant names are capitalized-ish; reuse idents, it only needs to lex.
        try out.appendSlice(gpa, pick(idents, rand));
        if (rand.boolean()) {
            try out.appendSlice(gpa, "(");
            try out.appendSlice(gpa, pick(types, rand));
            try out.appendSlice(gpa, ")");
        }
    }
    try out.appendSlice(gpa, " }");
}

fn genFn(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random) error{OutOfMemory}!void {
    if (rand.boolean()) try out.appendSlice(gpa, "pub ");
    try out.appendSlice(gpa, "fn ");
    try out.appendSlice(gpa, pick(idents, rand));
    try out.appendSlice(gpa, "(");
    const params = rand.intRangeAtMost(usize, 0, 3);
    var p: usize = 0;
    while (p < params) : (p += 1) {
        if (p > 0) try out.appendSlice(gpa, ", ");
        try out.appendSlice(gpa, pick(idents, rand));
        try out.appendSlice(gpa, ": ");
        try out.appendSlice(gpa, pick(types, rand));
    }
    try out.appendSlice(gpa, ") ");
    if (rand.boolean()) {
        try out.appendSlice(gpa, "-> ");
        try out.appendSlice(gpa, pick(types, rand));
        try out.append(gpa, ' ');
    }
    try genBlock(gpa, out, rand, 1);
}

fn genBlock(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random, depth: usize) error{OutOfMemory}!void {
    try out.appendSlice(gpa, "{\n");
    const stmts = rand.intRangeAtMost(usize, 0, 4);
    var s: usize = 0;
    while (s < stmts) : (s += 1) {
        try genStmt(gpa, out, rand, depth);
        try out.append(gpa, '\n');
    }
    try out.appendSlice(gpa, "}");
}

fn genStmt(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random, depth: usize) error{OutOfMemory}!void {
    // Cap nesting so control-flow bodies stay bounded (the parser has its own depth
    // guard, but keeping the generator shallow avoids gigantic buffers).
    const kind = if (depth >= max_gen_depth) rand.uintLessThan(u8, 3) else rand.uintLessThan(u8, 6);
    switch (kind) {
        0 => { // binding
            try out.appendSlice(gpa, pick(idents, rand));
            try out.appendSlice(gpa, " := ");
            try genExpr(gpa, out, rand, 0);
        },
        1 => { // assignment
            try out.appendSlice(gpa, pick(idents, rand));
            try out.appendSlice(gpa, " = ");
            try genExpr(gpa, out, rand, 0);
        },
        2 => { // return
            try out.appendSlice(gpa, "return");
            if (rand.boolean()) {
                try out.append(gpa, ' ');
                try genExpr(gpa, out, rand, 0);
            }
        },
        3 => { // if / if-else
            try out.appendSlice(gpa, "if ");
            try genExpr(gpa, out, rand, 0);
            try out.append(gpa, ' ');
            try genBlock(gpa, out, rand, depth + 1);
            if (rand.boolean()) {
                try out.appendSlice(gpa, " else ");
                try genBlock(gpa, out, rand, depth + 1);
            }
        },
        4 => { // loop / while
            if (rand.boolean()) {
                try out.appendSlice(gpa, "loop ");
            } else {
                try out.appendSlice(gpa, "while ");
                try genExpr(gpa, out, rand, 0);
                try out.append(gpa, ' ');
            }
            try genBlock(gpa, out, rand, depth + 1);
        },
        5 => { // match
            try out.appendSlice(gpa, "match ");
            try genExpr(gpa, out, rand, 0);
            try out.appendSlice(gpa, " {\n");
            const arms = rand.intRangeAtMost(usize, 1, 3);
            var a: usize = 0;
            while (a < arms) : (a += 1) {
                try out.appendSlice(gpa, if (rand.boolean()) "_" else pick(idents, rand));
                try out.appendSlice(gpa, " => ");
                try genExpr(gpa, out, rand, 0);
                try out.append(gpa, '\n');
            }
            try out.append(gpa, '}');
        },
        else => unreachable,
    }
}

fn genExpr(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random, depth: usize) error{OutOfMemory}!void {
    // Bounded recursion: past a shallow cap only emit an atom, so precedence
    // climbing stays a plausible but finite tree.
    if (depth >= max_gen_depth or rand.uintLessThan(u8, 3) == 0) {
        try genAtom(gpa, out, rand);
        return;
    }
    switch (rand.uintLessThan(u8, 4)) {
        0 => { // binary op (feeds the real precedence table)
            try genExpr(gpa, out, rand, depth + 1);
            try out.append(gpa, ' ');
            try out.appendSlice(gpa, pick(bin_ops, rand));
            try out.append(gpa, ' ');
            try genExpr(gpa, out, rand, depth + 1);
        },
        1 => { // unary
            try out.appendSlice(gpa, if (rand.boolean()) "-" else "!");
            try genExpr(gpa, out, rand, depth + 1);
        },
        2 => { // parenthesized
            try out.append(gpa, '(');
            try genExpr(gpa, out, rand, depth + 1);
            try out.append(gpa, ')');
        },
        3 => { // call
            try out.appendSlice(gpa, pick(idents, rand));
            try out.append(gpa, '(');
            const args = rand.intRangeAtMost(usize, 0, 3);
            var a: usize = 0;
            while (a < args) : (a += 1) {
                if (a > 0) try out.appendSlice(gpa, ", ");
                try genExpr(gpa, out, rand, depth + 1);
            }
            try out.append(gpa, ')');
        },
        else => unreachable,
    }
}

fn genAtom(gpa: std.mem.Allocator, out: *std.ArrayList(u8), rand: std.Random) error{OutOfMemory}!void {
    switch (rand.uintLessThan(u8, 5)) {
        0 => try out.appendSlice(gpa, pick(idents, rand)),
        1 => {
            var buf: [8]u8 = undefined;
            const n = std.fmt.bufPrint(&buf, "{d}", .{rand.uintLessThan(u32, 1000)}) catch unreachable;
            try out.appendSlice(gpa, n);
        },
        2 => try out.appendSlice(gpa, "true"),
        3 => try out.appendSlice(gpa, "false"),
        4 => try out.appendSlice(gpa, "\"s\""),
        else => unreachable,
    }
}

fn pick(comptime set: anytype, rand: std.Random) []const u8 {
    return set[rand.uintLessThan(usize, set.len)];
}

fn envU64(env: *std.process.Environ.Map, name: []const u8) ?u64 {
    const v = env.get(name) orelse return null;
    return std.fmt.parseInt(u64, std.mem.trim(u8, v, " \t\r\n"), 10) catch null;
}

fn envUsize(env: *std.process.Environ.Map, name: []const u8) ?usize {
    const v = env.get(name) orelse return null;
    return std.fmt.parseInt(usize, std.mem.trim(u8, v, " \t\r\n"), 10) catch null;
}
