# Performance Roadmap

Compile-time performance work, opened after the query-native pipeline capstone (Item 7 in
`ROADMAP_TECH.md`) shipped. The capstone unified the architecture but — as its design predicted —
did **not** move build time. This roadmap targets the measured time sinks.

## How to read this (for a future reviewer)

Same discipline as `ROADMAP_TECH.md`: each item states **Why** (the measured friction), **Requirements**
(testable done-conditions, incl. a perf-win bar), **Premises to re-check** (facts to re-measure before
implementing — perf numbers drift fast), **Invalidate if**, and **Success criteria**. Performance items
must prove a *measured* win, not just "looks faster" — every item names the benchmark that demonstrates it.

### Measured baseline (re-validate before starting — release, 1240 fns, post-7c, 2026-06-30)

| Stage | cold -j1 | note |
|-------|----------|------|
| discover (lex+parse) | **56.8%** (11.9ms) | serial; does NOT shrink warm (re-reads+re-lex+re-parse) |
| resolve | 8.6% | |
| typecheck | 2.8% | per-fn query nodes (Pass-C deleted) |
| lower (codegen+link) | 28.5% (compute 46% of it) | codegen region fans out per-fn |
| image+sign | 3.4% | |
| **total** | ~20.9ms | |

`-j` scaling (best of 3): **-j1 19.2ms → -j2 20.0 → -j4 22.2 → -j8 26.1ms** — more threads = *slower*.

Benchmarks: `examples/bench/profile.sh <entry> [jobs…]` (per-stage + -j sweep, needs a ReleaseFast
`toy`), `examples/bench/gen.sh <dir> <mods> <fns>` (corpus generator). Determinism/correctness gates
that EVERY perf change must keep green: `examples/diff.sh` ([C11] byte-identity), `examples/incremental.sh`
(stale-cache soundness + cutoff), `__text` hash identical across `-jN` (codegen determinism).

---

## P1 — Chunked parallel dispatch (make `-jN` actually scale)

> **Status: SHIPPED (uncommitted on `main`, 2026-06-30).** New `Engine.chunkedFanOut` primitive: splits
> `[0,n)` into ~`ncpu` contiguous ranges, one task per range, each loops its units serially; serial
> below a per-stage threshold; ncpu from the runtime knob (`-j1` → inline serial). Per-stage `Chunk`
> heuristics: `body`/`codegen` {threshold 64, 8 ranges/cpu}, `fn_link` {128, 4}, `cstr` {256, 4},
> `small_count` {256, 2}. Call sites switched across typecheck bodies / codegen / link / cstr.
> **Independently verified (1240-fn corpus, release):** -jN now BEATS -j1 — pre-P1 j1 19.2 → j8 26.1ms
> (slower); post-P1 j1 19.7 → **j4 16.9** → j8 17.3ms (~14% faster at j4, slowdown eliminated). Byte-
> identity preserved: `__text` j1==j8 identical; 465 unit + 42 integration + diff.sh 69/0 + incremental
> 5/5. (P1 resolves the "parallelism is inert/counterproductive" finding; the dead Pass-C was already
> deleted by Item 7c.)

**Cluster:** `src/query/Engine.zig` (`fanOut`), the `StageGraph` per-fn fan-out regions
(`src/query/StageGraph.zig`: the codegen region; the resolve+check region), `src/types.zig` /
`src/codegen` per-unit job bodies.

### Why
`-jN` is *slower* than `-j1` at every size. Root cause is dispatch **granularity**, not the
per-stage logic: `Engine.fanOut` submits **one task per function** (e.g. 10,201 tasks for 10,201
~21-line fns), each µs of work, so per-task overhead **≥** the work. Three compounding costs:
1. **Serial dispatch tax** — the submit loop runs N-tasks× on the main thread regardless of thread count.
2. **Hot shared-atomic contention** — µs workers finish instantly and pound the shared `Io.Group`
   completion counter + `Io.Threaded` queue; those cache lines ping-pong across cores, and contention
   *rises* with thread count (a slowdown, not just no-speedup, is the cache-line-bouncing signature).
3. **Allocator churn ×N-tasks across N CPUs** — each unit builds a fresh `BodyChecker` (ArrayLists from
   empty), a `DiagnosticSink`, per-construct field-seen bitsets; per-CPU allocator caches churn + a
   cross-thread free when a sink init'd on main is deinit'd on a worker.

Typecheck shows it worst (smallest unit → most lopsided ratio); codegen and the front-end use the same
per-unit pattern with more work to hide it (and warm codegen = cache-hits, so nearly as bad).

### The fix (chunk, don't de-parallelize)
Split units into **~`ncpu` contiguous ranges** and dispatch **one task per range** (each task loops its
range serially). N-tasks dispatches → ~ncpu; per-task overhead amortized; shared-atomic traffic cut
~N-tasks/ncpu → contention stops and it can actually scale on cold/`--force` builds where there's real
work. **Below a small unit-count threshold, stay serial.** Use **per-stage heuristics** (different
work-per-unit ⇒ different threshold + chunk size): typecheck bodies (tiniest), codegen (bigger; warm =
cache-hits), the per-file front-end (P2) each get their own tuning.

### Requirements
- R1: Per-unit fan-out (codegen region, resolve+check region) dispatches ~`ncpu` range-tasks, not one
  per fn; each range-task processes its slice serially. A per-stage threshold keeps small builds serial.
- R2: **Measured win** — on a large cold/`--force` build, `-j4`/`-j8` is now *faster* than `-j1` (or at
  minimum no longer slower), demonstrated via `profile.sh`. This is the bar; "no regression" is not enough.
- R3: Determinism preserved — `diff.sh` [C11] 69/0, `incremental.sh` 5/5, and `__text` byte-identical
  across `-j1`/`-j4`/`-j8` (range-chunking must not reorder observable results; per-worker outputs still
  merge in deterministic unit order).
- R4: Heuristics are explicit + documented (thresholds/chunk sizes per stage), not magic numbers buried.

### Premises to re-check
- Re-measure `profile.sh` first — confirm `-jN` still hurts and which stages fan out per-fn (post-7c the
  codegen region and the resolve+check region do; `grep Engine.fanOut`).
- `Engine.fanOut`'s shape is still per-unit submit-loop + `Io.Group` await (the contention source).
- The per-worker result merge is still deterministic-by-unit-order (range-chunking must preserve it).

### Invalidate if
- A re-measure shows `-jN` already scales (then this is done — verify).
- The std `Io` runtime gained a chunked/scheduled fan-out primitive (use it instead of hand-chunking).

### Success criteria
- `profile.sh` shows `-j4`/`-j8` beating `-j1` on a large cold corpus (the measured win).
- All determinism/correctness gates green; `__text` identical across thread counts.

### Risk / blast radius
**Medium.** Touches the one parallelism mechanism + the per-stage call sites. Determinism is the
hazard (chunk boundaries must not change merge order) — guarded by diff.sh + incremental.sh + the
`__text`-across-`-jN` check.

---

## P2 — Eliminate redundant `discover` syscalls (NOT parallelize — research overturned that)

> **Status: SHIPPED (uncommitted on `main`, 2026-06-30).** `Graph.zig` only. Stage 1: deleted the
> standalone `fileExists` open+close; `canonicalize` now returns `?[]u8` (null = missing), so each import
> gets existence + physical-identity from ONE `realPathFileAlloc` instead of openFile/close THEN realpath.
> Stage 2: added `realpath_cache` (memoize realpath by raw resolved path) so a file reached by N importers
> realpaths once, not N×. (A fix round resolved a latent OOM double-free via dup-before-`put` ordering.)
> **Independently verified — controlled best-of-5 before/after, same machine, back-to-back:** discover
> **linear 17.2→9.9ms (1.73×)**, **fan-in (100 modules share one util) 29.4→13.7ms (2.14×)**.
> Behaviour-identical: diff.sh [C11] 69/0, incremental 5/5, **missing-import diagnostics unchanged**
> (modules/check.sh 10/0, same message + import-token offset), 465 unit + 42 integration; only `Graph.zig`
> changed; no module-ordering/id change (no [C11] re-baseline). Stage 3 (warm manifest read-skip) NOT done
> — optional future. (Single `profile.sh` runs are noisy ±2-3ms from machine load — see P4.)

**Cluster:** `src/driver/Graph.zig` (`load` + `collectImports`: `fileExists`, `canonicalize`/`realPathFileAlloc`, `readFileAlloc`).

> **RESEARCH OUTCOME (2026-06-30, read-only, INSTRUMENTED):** P2 began as "parallelize discover." A
> research pass *measured* it (per-phase instrumentation + stub experiments, medium corpus 101 modules
> ~1400 fns, ReleaseFast) and **overturned the premise.** Cold discover 33.9ms breaks down as:
> `readFileAlloc` 12.3ms · lex 1.6 · parse 1.5 · **`collectImports` 17.4ms** (of which **`fileExists`
> open+close 10.1ms**, **`canonicalize`/realpath 7.2ms**, AST walk 0.13ms). So **lex+parse+astwalk =
> ~3.2ms cold (<10%)** — parallelizing it (the original P2 plan + all 4 briefed strategies) chases a
> ~3ms sliver. The real cost is **~3 redundant syscalls per module on the serial DFS critical path**
> (read + existence open/close + realpath). Validation by stubbing: `fileExists`→true dropped discover
> ~34→~24ms; stubbing **both** `fileExists` + `canonicalize` collapsed `collectImports` to 0.16ms and
> **total build 46.5ms → 19.4ms (~58% off)** — zero parallelism, zero determinism risk. (Bonus: the
> `gen.sh` bench corpus is a strict linear chain, so BFS/worklist parallel discovery is a no-op on it.)

### Why
Discover is the build-time dominator (~57% cold), but it is **syscall-bound, not CPU-bound**: redundant
per-import `fileExists` (a separate open+close just to test existence) and per-import `realpath`
canonicalization dominate. These are eliminable without touching parallelism or module ordering.

### The fix (staged, syscall elimination — determinism-free)
- **Stage 0 (validate, ~30min):** reproduce the per-phase split on the *real target* corpus (throwaway
  timer in `Graph.zig` `load`+`collectImports`: read/lex/parse/fileExists/canonicalize/astwalk). Confirm
  the breakdown holds before changing code (the bench corpus is a linear chain — re-measure on a realistic
  graph).
- **Stage 1 (big, safe win — kill `fileExists`):** fold existence-detection INTO `canonicalize` (return
  `?[]u8`, null = missing) and delete the separate `fileExists` open+close. ~10ms off cold discover.
  Wire the null case to emit the missing-import diagnostic at the same import-token offset (preserve
  diagnostic quality — see risk).
- **Stage 2 (attack realpath — new dominator):** eliminate/defer the per-import `realPathFileAlloc` (the
  ~7ms remaining). Options: skip realpath when the path is already canonical; cache realpath results;
  dedup by a cheaper key (mind physical-identity dedup for symlinks).
- **Stage 3 (OPTIONAL, warm-only):** if warm discover is still dominated by `readFileAlloc`, add a
  per-program manifest {path→(mtime,size)} to skip re-read of unchanged files. (Warm discover does NOT
  shrink today because `readFileAlloc` is unconditional even on cache hits.)
- **Stage 4 (LAST, probably never — parallel reads):** only if a post-Stage-1-3 profile shows
  `readFileAlloc` still serialized AND wide enough to parallelize (uses P1's `chunkedFanOut`).

### Requirements
- R1: Delete the redundant `fileExists` open+close (Stage 1) and eliminate/defer per-import realpath
  (Stage 2). Measured win via `profile.sh`/`--timings` on a realistic corpus.
- R2: Determinism untouched — module ordering (serial sorted DFS → intern order → global ids) is NOT
  changed, so [C11] holds with ZERO re-baseline (the big advantage over the rejected parallel strategies,
  which permute intern order DFS→BFS and need a one-time golden re-bless).
- R3: **Missing-import diagnostic quality preserved** — the `canonicalize`-returns-null path must emit at
  the exact import-token offset, same message, as today's `fileExists` check (the main correctness risk).
- R4: gates green: `diff.sh` [C11] 69/0, `incremental.sh` 5/5, `__text` across `-jN`, and the
  `examples/modules/errors/*` corpus (missing/cycle/etc. diagnostics) unchanged.

### Premises to re-check
- Re-measure the per-phase split on the real corpus (Stage 0) — the bench corpus is a linear chain; a wide
  import graph may shift proportions (but per-module syscall overhead is topology-independent).
- Confirm `fileExists` + `realPathFileAlloc` are still separate per-import syscalls in `collectImports`.

### Invalidate if
- A re-measure shows lex/parse actually dominates (then parallelization is back on the table — but the
  instrumented evidence says it doesn't).

### Success criteria
- Measured discover/total-build speedup (the stub ceiling suggests ~58% off total build is reachable);
  all gates green; missing-import diagnostics unchanged; ZERO determinism re-baseline.

### Risk / blast radius
**Low-medium.** No parallelism, no module-ordering change → no determinism risk. The real risk is
diagnostic regression (missing-import message/offset) and symlink physical-identity dedup when realpath is
deferred — both guarded by the `examples/modules/errors/*` corpus + `diff.sh`.

---

## P3 — Warm-rebuild overhead — RESOLVED ✅ (2026-06-30)

Was ~36ms unattributed on a warm 1240-fn rebuild (GraphFrozen construction + a serial O(N²) red-green
reuse pre-pass). Resolved by the reuse-reversal (content-fp sole driver, DAG debug-only) + the nodeFp
O(N²)→binary-search fix. Re-measured warm: `lower` compute = 0, cache-get = 0.44ms. Recorded for history;
no action.

---

## P4 — `--timings` observability: per-stage compute/cache split + reconcile to total

> **Status: SHIPPED (uncommitted on `main`, 2026-06-30).** `LowerProbe` generalized to a stage-agnostic
> `Engine.StageProbe` (3 atomics; `LowerProbe` kept as an alias = no call-site churn), wired into the
> shared query get/put path + a discover probe threaded through `Graph.discoverDag`/`load` (file-read+lex+
> parse → compute, lex/parse cache → get). `printTimings` now uses build wall-clock as `total`, adds
> `setup` (cache open + `pack.load` + dag/realpath init) and `post` (DAG-persist window) laps, and always
> prints an explicit `other/ovh = total − Σ(named)` residual. **Independently verified:** top-level rows
> SUM TO total (ReleaseFast force: setup+discover+resolve+typecheck+lower+post+image+sign+ovh = 50.944 vs
> total 50.943 — `other/ovh 0.000`); `discover` shows compute/cache-get/cache-put. Observability-only:
> diff.sh 69/0, incremental 5/5, 465 unit, 42 integration; probes are zero-cost when `--timings` off.
> CAVEAT: sub-rows are per-worker times SUMMED across threads, so at `-jN` they can exceed their parent
> wall-clock (a parallel-overlap signal, labeled "% of parent") — they are NOT part of the total
> reconciliation (the total uses each stage's wall-clock). Lower-INTERNAL attribution (GraphFrozen +
> per-fn Walks/fingerprint outside the codegen cache laps) is a deeper future split, out of P4's scope.
>
> **NEW finding (now visible because timings reconcile):** `setup` is a real cost — ~0.4ms cold but
> **~8ms (25%) on a WARM build** (the `pack.load` first-FS-open + cache/dag init). That's the next warm
> lever, previously hidden in the unattributed gap. Tracked under "Future candidates" below.

**Cluster:** `src/driver/main.zig` (the `--timings` printer), `src/query/Engine.zig` (`LowerProbe` and the
query get/put path), the `StageGraph` interpreter.

### Why
Two gaps undermine trust in every perf number this roadmap relies on:
1. **No per-stage compute/cache split.** Only `lower` reports `compute` / `cache-get` / `cache-put` /
   `link-tail` (via `Engine.LowerProbe`). The other stages — `discover` (lex+parse, which have an on-disk
   content cache), `resolve`, `typecheck` (memo-tier in-build) — report a single wall-clock with no
   visibility into whether time went to compute vs a cache hit/miss/write. You can't tell a slow stage
   from a cold-cache stage.
2. **Timings don't sum to total.** The per-stage lines + `image+sign` do NOT add up to the reported
   `total` — there's unattributed time (GraphFrozen construction, the StageGraph interpreter + barrier
   joins, allocator/setup/teardown, the `-jN` dispatch tail). The gap means the breakdown can mislead
   (a real cost hides in "the gap").

### Requirements
- R1: Every stage reports a compute / cache-get / cache-put sub-breakdown where it has a cache tier
  (generalize the `LowerProbe` pattern to a per-stage probe driven by the engine's get/put path, so any
  query-backed stage gets it uniformly). Stages with no cache (e.g. the barrier joins) report compute only.
- R2: The printed lines RECONCILE to total — either every contributor is a named line, or an explicit
  `other/overhead` line carries `total − Σ(named)` so the columns always sum to `total` (and `--timings`
  prints the residual rather than hiding it). The residual should be small and labeled, not a silent gap.
- R3: Observability-only — NO behaviour/byte-identity change; the probes are behind `--timings` and add
  ~zero cost when off (matches the existing `LowerProbe` discipline). Still run the gates after.
- R4: Determinism of the NUMBERS is not required (wall-clock varies), but the STRUCTURE (which lines
  exist, summing to total) must be stable.

### Premises to re-check
- After Item 7, timing is collected via the StageGraph interpreter + `Engine.LowerProbe`; confirm where
  each stage's wall-clock is lapped and where get/put happen, so the per-stage probe hooks the right spots.
- Confirm the current gap empirically first (sum the printed lines vs `total` on a real build) so the
  `other/overhead` line is attributing a real residual, then drill into the largest contributor.

### Success criteria
- `--timings` shows compute/cache-get/cache-put per cache-backed stage, and the lines sum to `total`
  (residual surfaced as a labeled line). Gates green (observability-only).

### Risk / blast radius
**Low.** Instrumentation behind `--timings`; no output/behaviour change. The value is trustworthy perf
numbers for the rest of this roadmap. Do AFTER P2 lands (it touches `discover`, which this then instruments).

---

## Future perf candidates (unscoped)

- **P5 (new, surfaced by P4) — warm `setup` cost (~8ms / ~25% of a warm build):** the `pack.load`
  first-FS-open + cache/dag init. Now visible because timings reconcile. Likely the top *warm* lever
  (cold it's ~0.4ms). Investigate lazy/deferred pack load, mmap, or overlapping it with discover.
- **Lower-internal attribution:** split `lower`'s wall-clock further (GraphFrozen build + per-fn
  Walks/fingerprint/unpack outside the codegen cache laps) — a deeper `--timings` breakdown if wanted.
- Warm discover re-lex/parse skip (folded into P2's research; partly Stage 3 there) — note P2 already cut
  warm discover via the realpath cache; the remaining warm floor is `readFileAlloc` + `setup` (above).
- `link-tail` / image+sign (currently small; revisit if it grows).
- Allocator strategy for the per-unit job bodies (arena-per-range once P1 chunks dispatch).
