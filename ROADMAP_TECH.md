# Technical Architecture Roadmap

Module-deepening refactors to reduce coupling and make the compiler testable at
boundaries instead of inside. Source: architectural review of the lex→parse→
resolve→typecheck→lower→opt→codegen→link pipeline plus the query/incremental layer
(2026-06-29).

## How to read this document (for a future reviewer)

This roadmap will be implemented incrementally and may sit unstarted for a while.
Premises drift. Each item below is written so you can **re-validate it before
implementing**, not just follow it blindly.

Every item has:

- **Why** — the friction it removes and what it buys. If this no longer hurts, drop the item.
- **Requirements** — concrete, testable conditions for "done".
- **Premises to re-check** — facts the item assumes are still true. **Verify these first.**
- **Invalidate if** — observable conditions that mean this item should be dropped or redesigned.
- **Success criteria** — how we know it worked.
- **Risk / blast radius** — what could break, how widely.

A guiding principle throughout (Ousterhout): a **deep module** has a small interface
hiding a large implementation. We are trading shallow modules + convention-enforced
seams for deeper modules + boundary tests. Testing strategy is **replace, don't layer**:
when a boundary test exists, the old shallow-module unit tests it subsumes get deleted.

### Project facts assumed by the whole roadmap (re-check at start)

| Fact | As of 2026-06-29 | Why it matters |
|------|------------------|----------------|
| All deepened modules are pure / in-memory (no I/O, no network) | true | Dependency category is **in-process** everywhere → "merge and test directly", no ports/mocks needed |
| Target is AArch64 Mach-O only | true | Single-backend assumptions in codegen/link items |
| No generics; pattern-matching parsed-but-unchecked | true | Type-algebra & typecheck items scoped to current language |
| Single-file (`check`) path still exists alongside graph (`checkGraph`) path | true | Item 1 (unify) presupposes both exist and duplicate logic |
| Cache correctness relies on hand-mirrored AST walks ([C7]) | true | Item 2 |
| `zig build test` deadlocks; tests run via `zig build test-bin` → `./zig-out/bin/toyc-test`; integration via `toyc-integration-test` | true | All "validation" steps below |
| VERIFY byte-identity re-lower ([C11]) is the cross-check for codegen determinism | true | Regression guard for items 1, 3, 4, 6 |

If the first four flip (e.g. single-file path deleted, second backend added, generics
landed), **stop and re-plan** — several items below assume the current shape.

---

## North star

**Every compilation stage should run through the query engine** so that dependency
tracking, content-addressed caching, and red-green invalidation work *end-to-end* — and so
the whole pipeline parallelises against one scheduler. Today only `lex`, `parse`, and
`codegen` are query nodes; **resolve and typecheck bypass the engine entirely** (direct
calls, recomputed every build, no cache), and `lower`/`opt` are fused inside the codegen
query rather than being cacheable nodes of their own. **Item 7** is the goal that closes
this gap; several other items are prerequisites or facets of it.

## Sequencing

```
   ┌────────────── Phase 0 (isolated, low risk) ──────────────┐
   │  Item 2: Fingerprint/Walk → one walk   Item 5: Diag sink  │
   └───────────────┬───────────────────────────────┬──────────┘
                   │                                │
                   │                                ▼
                   │                      ┌──── Phase 1 ────┐
                   │                      │ Item 3: extract │
                   │                      │ Type algebra/   │
                   │                      │ layout module   │
                   │                      └────────┬────────┘
                   │                               ▼
                   │                      ┌──── Phase 2 ────┐
                   │                      │ Item 1: unify   │
                   │                      │ single-file +   │
                   │                      │ graph paths     │
                   │                      └────────┬────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                  ┌──────────────────── Phase 3 ───────────────────────┐
                  │ Item 7: query-native pipeline end-to-end (CAPSTONE) │
                  │   ├─ 7a generalise engine: barriers / fan-in        │
                  │   ├─ 7b resolve + typecheck → query nodes           │
                  │   └─ 7c lower + opt → query nodes                   │
                  │   absorbs Item 4 (facade) + Item 6 (typed view)     │
                  └────────────────────────────────────────────────────┘
```

| Phase | Items | Rationale for ordering |
|-------|-------|------------------------|
| 0 | **2 ✅**, **5 ✅ (partial)** | Isolated, high value-per-effort, no deps. Build confidence + leave clean diagnostics for later items to inherit. SHIPPED on branch `phase-0-architecture`; Item 5 front-end only — back-end carried over to **Item 8**. |
| 0+ | **8** | Extend the Item 5 sink to the back-end (lower/codegen/parse). No dep on 1/3; can run anytime or fold into 7c. |
| 1 | **3 ✅** | Shrinks the 4.5k-line `types.zig` monolith; prerequisite that makes Items 1 and 7 tractable. SHIPPED (`src/layout/Engine.zig`; types.zig 4384→3967). |
| 2 | **1 ✅** | Consolidation: one resolve + one typecheck. **Prerequisite for Item 7** — make it query-native *once*, not twice. SHIPPED — single-file deleted; graph-of-one is the one path (net −588 lines). |
| 3 | **7 ✅** (capstone) | Everything through the engine. **Subsumes Item 4** (the generalised engine API *is* the facade) and **Item 6** (per-query typed inputs *are* the typed view). SHIPPED via 7a/7b/7c — `StageGraph` + `Engine.barrier`, dead Pass-C deleted, enum collapsed; Items 8/9 absorbed. |

Phases are gated, not item-strict: 2 and 5 can run concurrently. **Items 4 and 6 are no
longer standalone** — they are folded into Item 7 (see each item's note); they remain
listed for traceability.

---

## Item 2 — Collapse Fingerprint/Walk into one AST semantic walk

**Phase 0. Cluster:** `src/query/Fingerprint.zig` (663), `src/query/Walks.zig` (342).

### Why
`Walks.walkCalls` "MUST mirror `Fingerprint.walk` positionally" so the order-sensitive
[C7] fold lines up (see `Walks.zig:4-20`). Two hand-written traversals must stay
byte-for-byte order-identical; a one-line drift silently corrupts the incremental
cache → **miscompile with no error**. `Walks.zig` currently has **zero tests**. This is
the highest-severity silent-failure seam in the repo, and the cheapest to close.

### Requirements
- R1: A single traversal defines AST visit order once; both the fingerprint fold and the
  edge/dependency collection are driven from it (visitor callback, shared iterator, or
  one walk emitting both). No second hand-mirrored walk remains.
- R2: Public behaviour of fingerprinting and edge-collection is unchanged (same hashes,
  same edges) for all existing inputs.
- R3: A boundary test asserts the two consumers observe the **same nodes in the same
  order** — i.e. drift is made impossible or caught, not merely discouraged by a comment.

### Premises to re-check
- The "MUST mirror" comments still exist in `Walks.zig` / `Fingerprint.zig` (grep `[C7]`,
  `mirror`). If the code was already unified, this item is **done** — verify and close.
- The fold is still order-sensitive (test `swapped operands hash differently … [C7]`
  at `Fingerprint.zig:376` still present).

### Invalidate if
- The two walks were already merged.
- Fingerprinting moved to a content-hash of serialized AST bytes (making the structural
  walk irrelevant).

### Success criteria
- Deleting/altering one consumer's visit order makes the new boundary test fail.
- All existing fingerprint tests (27) and the cache/firewall integration tests pass
  unchanged.
- `Walks.zig` no longer carries the "hand-mirrored / silently breaks" warning.

### Risk / blast radius
**Low, but high-stakes.** Contained to `query/`. Any regression shows up as a cache
miscompile, so guard with: full integration suite + a `--force --verify` build of
`examples/bench` (byte-identity [C11]) before and after.

---

## Item 5 — Diagnostics sink (own memory, ordering, module-tagging)

**Phase 0. Cluster:** `src/diagnostics/Diagnostic.zig` (6 lines) + scattered `owned_msgs`,
`diag_mods`, `sortDiagsStable`, `GraphDiagnostic` across `resolve.zig` / `types.zig` /
`resolve_graph.zig` / `types_graph.zig`.

> **Status: SHIPPED (PARTIAL) on branch `phase-0-architecture`.** `src/diagnostics/Sink.zig`
> now owns emission + message lifetime + `(scope, byte_offset)` stable ordering; the
> `diag_mods` length-sync hazard and `sortDiagsStable` are gone. The **front-end
> resolve+typecheck paths (single-file and graph) are fully migrated** and verified
> (446 unit + 42 integration + corpus/[C11] + incremental-soundness all green).
>
> **CAVEAT — call sites NOT migrated (deliberately out of Item 5's stated scope, which was
> resolve+typecheck).** The back-end and parse stages still hand-build
> `std.ArrayList(Diagnostic)` / hand-manage `owned_msgs` and do NOT flow through the Sink:
> - `src/lower.zig` — lowering diagnostics via `out_diags: *std.ArrayList(Diagnostic)`.
> - `src/codegen/CodegenIr.zig` — codegen diagnostics via `ArrayList(Diagnostic)` + manual append.
> - `src/driver/Driver.zig` (~233–265, 507, 594, 1073) — the lower/codegen result struct still
>   carries an `owned_msgs` field freed by hand.
> - `src/query/Engine.zig:372` — codegen-query diagnostics `ArrayList(CodegenIr.Diagnostic)`.
> - `src/parse.zig` — single `*?Diagnostic` out-param (a different shape; lower priority).
>
> These are tracked as **Item 8**. Note the front-end `Result` structs still *expose* an
> `owned_msgs` field, but it is the Sink's `toOwned()` handoff (the Result owns + frees it via
> `deinit`; callers use `res.deinit`/`tc.deinit`, not manual frees) — that is the intended
> boundary contract, not a missed site.

### Why
`Diagnostic` is a 6-line struct; all real behaviour lives smeared across callers:
heap-vs-static message ownership, manual `deinit`, module tagging, stable re-sort by
byte offset. Ownership is by convention → leaks are silent. Every stage's `Result` carries
its own `owned_msgs` bookkeeping. Doing this **before** Items 1 and 3 means those rewrites
inherit a clean sink instead of re-propagating the manual pattern.

### Requirements
- R1: One module owns diagnostic storage and message lifetime. Emitting a formatted
  (heap) or static message goes through it; it frees everything on `deinit`. Callers stop
  hand-managing `owned_msgs`.
- R2: Supports optional module tagging and produces a stable, deterministic ordering
  (subsumes `sortDiagsStable`). Single-file mode is "no module tag"; graph mode tags.
- R3: Ordering is deterministic and independent of emission thread/order (relied on by
  parallel typecheck).

### Premises to re-check
- Diagnostic message ownership is still a mix of static + heap with manual `deinit`
  (grep `owned_msgs`, `sortDiagsStable`, `GraphDiagnostic`).
- Diagnostics are still emitted from multiple stages with the same struct shape.

### Invalidate if
- A diagnostics/error-reporting refactor already happened (e.g. an arena owns all
  messages) — then ownership is solved; only the ordering/tagging consolidation may remain.

### Success criteria
- No caller outside the sink module references `owned_msgs` / does manual message frees.
- A leak test (or `GeneralPurposeAllocator` leak check in the test binary) is clean across
  resolve/typecheck error paths.
- Diagnostic ordering tests pass; cross-module diagnostic order is stable under `-j4`.

### Risk / blast radius
**Low.** Touches stage `Result` shapes (resolve/types and graph variants), so it ripples
to the driver's consumption of diagnostics. Mechanical; caught at compile time by Zig.

---

## Item 3 — Extract the Type algebra + layout deep module from `types.zig`

> **Status: SHIPPED on branch `phase-0-architecture` (uncommitted working tree, on top of the
> Phase 0 commit).** New `src/layout/Engine.zig` (~972 lines) owns the algebra + the layout
> /cycle-detection state machine behind a `computeAll`-style entry with a narrowed callback
> `Env` (instead of `*Typecheck`). `types.zig` re-exports the value types
> (`Kind`/`Type`/`Layout`/`VariantForm`/`VariantLayout`/`EnumLayout`) so all 6 downstream
> importers needed **zero** edits; `types.zig` shrank ~4384→3967. The laid-table types
> (`StructSym`/`EnumSym`/`LayoutState`) are deliberately **public** in the engine because the
> Pass-C freeze aliases them into `Model` and `BodyChecker` reads `.state/.poisoned/...` —
> the engine hides the recursion/state-machine, not the laid data. Behaviour-preserving:
> 14 new layout boundary tests (cycle / indirect-cycle / recursive-enum sentinel / algebra /
> layout-math, all check()-free). Independently verified: 457 unit + 42 integration + corpus
> [C11] 69/0 + incremental soundness 5/5 (incl the **layout-edit** scenario → identical ids
> /fingerprints, cutoff intact).

**Phase 1. Cluster:** `src/types.zig` (4526) — the `Type`/`Kind`/`eql`/`assignable` algebra
and the struct/enum **layout + cycle detection** (`LayoutState` + `poisoned`), currently
fused with checker orchestration and parallel job dispatch.

### Why
`types.zig` is the monolith: type algebra, layout resolution, single-file checking, graph
orchestration, and parallel Pass-C dispatch in one 4.5k-line file. Codegen, lower, Abi,
and FrameLayout all `@import("types.zig")` **just to read `Type.Kind` / `Layout`** — so the
algebra leaks out while the layout-cycle state machine hides inside the checker. Pulling the
algebra + layout into a small deep module: (a) shrinks the monolith's interface, (b) lets
layout/assignability/cycle-detection be tested at a small boundary instead of via full
`check()`, (c) is the enabling step for Item 1 (unifying the two checkers is far easier when
layout is not entangled with orchestration).

### Requirements
- R1: A module owns the type algebra (`Type`, `Kind`, `eql`, `assignable`, constants) and
  layout computation (struct/enum `Layout`, size/align, **cycle detection**, poisoning).
  Its interface is the type/layout queries; the recursion + state machine are hidden.
- R2: Downstream readers (codegen, lower, Abi, FrameLayout) import the algebra from this
  module, not from the checker monolith.
- R3: Layout cycle detection and poisoning are covered by direct boundary tests
  (direct cycle, indirect cycle, recursive enum sentinel) without constructing a full
  typecheck pass.
- R4: No behaviour change to emitted diagnostics or inferred types.

### Premises to re-check
- `types.zig` is still a single large file mixing algebra + layout + orchestration
  (re-check line count and that `LayoutState`/`poisoned` live alongside `check`/`checkGraph`).
- Downstream modules still import `types.zig` only for `Type.Kind`/`Layout` (grep imports
  in `codegen/`, `lower.zig`, `abi/Abi.zig`, `frame/FrameLayout.zig`).
- No second numeric type / generic has been added that would change the algebra's shape.

### Invalidate if
- `types.zig` was already decomposed.
- The type system gained generics/inference complexity that means "algebra" can't be a
  small leaf module — re-plan the boundary.

### Success criteria
- Monolith interface surface shrinks; layout logic no longer requires `check()` to test.
- New layout/algebra boundary tests pass; the relevant subset of the 105 `types.zig` unit
  tests that were really layout/algebra tests **move** to the new module (replace, don't
  layer).
- Full suite + VERIFY build unchanged.

### Risk / blast radius
**Medium.** Wide import churn (many modules read `Type`). Pure code move + re-point, so
compile-time-checked; behaviour risk is low if no logic changes. Do it as a mechanical
extract first, behaviour-preserving, before any cleanup.

---

## Item 1 — Unify single-file and whole-graph paths

> **Status: SHIPPED (uncommitted working tree on `main`).** Decision (confirmed by analysis):
> single-file was pre-M14 **legacy**, not a perf path, and graph-of-one is a faithful superset
> (zero behavioural deltas) — so the single-file forks were **deleted**, not unified-in-place.
> Implementation: new in-memory `Graph.single` constructor (BORROWS the caller's
> source/tokens/AST slices — load-bearing: typecheck `EnumLayout` variant names are borrowed
> into `module.source`; `deinitSingle` frees only the 1-module spine) → `resolveGraph`/`checkGraph`
> → `projectResolve`/`projectTypecheck` adapters fold the one-module result back into the
> `Resolve.Result`/`Typecheck.Result` carriers so every downstream reader + inline test is
> unchanged. `Driver.pipeline`'s `--emit check` block + the test helpers (`checkSource`,
> `expectLowered`, resolve tests) route through it; `checkGraph` gained `io: ?Io` (single-source
> passes `null` → serial Pass-C, so `Driver.run`'s per-file fan-out is the only parallelism — no
> nested blowup). **Deleted:** `resolve.zig`'s entire `Resolve` walker + `resolve()` (926→405),
> `types.zig::check` + single-file `run()`, dead `renderProgramIr`. **Net −588 lines.** Files kept
> their names (minimal-churn; a cosmetic `resolve_graph→resolve` rename is a deferred
> behaviour-preserving step). Independently verified: 458 unit + 42 integration + corpus [C11]
> 69/0 + incremental soundness 5/5; `--emit check/ir` byte-identical at -j1 vs -j4. The workflow
> caught + fixed a real use-after-free (a first cut duped the source → dangling variant names).

**Phase 2. Cluster:** `src/resolve.zig` (926) + `src/resolve_graph.zig` (1189);
`types.zig::check` + `types.zig::checkGraph` + `src/types_graph.zig` (592).

### Why
The graph variants are copy-paste retrofits of the single-file logic, and `types.zig`
branches throughout on `graph: ?*GraphCtx`. **The same semantics are implemented twice and
must be kept in sync by hand** — every resolve/typecheck bug fix has to land in two places,
and the two paths can silently diverge. Treating single-file as "graph-of-one" collapses
this to one implementation with one set of boundary tests. Biggest coupling/DRY win in the
codebase.

### Why now (ordering)
Land **after** Item 3 (so layout/algebra is already factored out of the merge) and
**after** Item 5 (so the unified path emits diagnostics through the clean sink, not two
copies of the manual pattern).

### Requirements
- R1: One resolve implementation and one typecheck implementation. The single-file path is
  expressed as the trivial graph (one module, no imports), not a parallel codebase.
- R2: No duplicated traversal/checking logic between a "_graph" file and its base file.
  `resolve_graph.zig` / `types_graph.zig` either disappear or become thin adapters with no
  re-implemented logic.
- R3: Output for a single-file program is identical (types, diagnostics, sigs, layouts) to
  the pre-refactor single-file path.
- R4: Output for multi-file programs is identical to the pre-refactor graph path.
- R5: Parallel Pass-C and determinism ([C11] byte-identity) are preserved.

### Premises to re-check
- **Both paths still exist and still duplicate logic** (grep `checkGraph`, `resolveGraph`;
  confirm `resolve_graph.zig` mirrors `resolve.zig`). *If the single-file path was already
  deleted or already routes through graph-of-one, this item is partly/fully done.*
- The single-file path still has real callers (the `--emit check`/`-o` flow). If nothing
  calls it, the cheaper move is **delete single-file**, not unify — re-decide here.
- Graph mode still assigns program-global struct/enum/fn ids that single-file gets as an
  identity map (the thing that makes graph-of-one behave like single-file).

### Invalidate if
- One of the two paths is already gone.
- A decision is made to drop multi-file support (then keep single-file, delete graph).
- Performance measurement shows graph-of-one materially regresses single-file compile
  time (the single-file path may have existed for speed — **measure before committing**).

### Success criteria
- `resolve_graph.zig` + `types_graph.zig` contain no re-implemented resolve/typecheck logic.
- The union of resolve (27+11) and typecheck (105+10) unit tests is expressible against one
  module; redundant duplicates deleted (replace, don't layer).
- Integration suite (`query_engine.zig`, `dag_firewall.zig`) and a VERIFY build pass
  unchanged; single-file vs multi-file outputs match the pre-refactor baselines.

### Risk / blast radius
**High.** This is the largest change and touches the driver's orchestration of both paths.
Mitigations: do it strictly behaviour-preserving; snapshot pre-refactor outputs (types,
diagnostics, emitted binaries) for a corpus and diff after; keep `--force --verify` green
throughout.

---

## Item 4 — Query facade: Cache + Engine + Dag + reuse as one boundary

**Phase 3. Cluster:** `src/query/Engine.zig` (830), `src/query/Cache.zig` (949),
`src/query/Dag.zig` (1031), `src/query/Key.zig` (135).

> **Status: folded into Item 7.** Once every stage is a query node (7), the generalised
> engine entry point *is* the facade. Kept here for traceability; implement as part of 7a.

### Why
Callers thread `Cache` + optional `Dag` + `LowerProbe` + reuse decisions separately, and
several derived facts must agree by convention: `Engine.nodeKeyFor` ↔ `Cache.Phase`
(add a phase, forget the mapping → DAG nodes silently skipped), and `Dag.sigFingerprint`
↔ `Fingerprint` fold (independent folds that can drift). `Engine.codegen` also takes
`anytype frozen` (duck-typed, no compile-time contract). A facade that owns the
co-threading and the key/phase mapping removes whole classes of silent drift, and turns the
firewall behaviours currently only covered by integration tests into unit tests on the
facade.

### Why now (ordering)
After Item 1, there is **one** compilation path to wrap (not single-file + graph), and
after Item 2 the fingerprint/walk drift is already closed — so the facade wraps a stable
surface.

### Requirements
- R1: A single entry point co-threads Cache + Dag + reuse + probe, so callers pass one
  thing, not four. Forgetting a piece becomes impossible (it's not exposed).
- R2: The `Cache.Phase` ↔ DAG node-key mapping is centralized and total — adding a phase
  forces handling the mapping (compile-time, e.g. exhaustive switch).
- R3: `Engine.codegen`'s `anytype frozen` is replaced by (or constrained to) a typed
  contract (depends on / coordinates with Item 6).
- R4: No change to cache hit/miss behaviour, fingerprints, or reuse verdicts.

### Premises to re-check
- The piecemeal threading still exists (driver passes Cache + Dag + LowerProbe separately;
  `Engine.codegen` still `anytype`).
- `nodeKeyFor` ↔ `Cache.Phase` is still a hand-maintained mapping (grep `nodeKeyFor`,
  `Phase`).
- `Dag.sigFingerprint` and `Fingerprint` still fold sig identity independently.

### Invalidate if
- A facade/gate already exists.
- The DAG/incremental layer is being replaced (e.g. moving to per-query persisted results,
  M18 latch work landed) — re-scope against the new design.

### Success criteria
- Driver threads one query object; no call site assembles Cache+Dag+reuse by hand.
- Adding a `Cache.Phase` without updating the node-key mapping fails to compile.
- `dag_firewall.zig` behaviours have unit-level coverage on the facade; integration tests
  still pass.

### Risk / blast radius
**Medium-high.** Central to incremental correctness; a mistake is a cache miscompile.
Guard with the full firewall integration suite + `--force --verify`. Coordinate with
Item 6 (shared `frozen` contract).

---

## Item 6 — Formalize the codegen `Frozen` view (kill the `anytype` duck-typing)

**Phase 3. Cluster:** `Driver.Frozen` / `GraphFrozen` / `frozenFor` + `Engine.codegen(anytype
frozen)` + `lower.Inputs`.

> **Status: folded into Item 7.** When `lower`/`opt`/`codegen` become query nodes (7c),
> each needs a typed per-query input — that typed contract *is* this item. Kept here for
> traceability; implement as part of 7c.

### Why
The per-function read-only view passed to codegen is an undocumented struct contract:
`node_types[i]` for a struct must match `layouts[struct_id]`; the tree/resolutions/node_types
**index spaces must align**; nothing validates any of it (`lower.zig` is "PURE of its frozen
inputs" with no cross-field invariant checks). A typed boundary makes the contract explicit,
enables view-construction tests, and protects `lower`'s purity contract from silent data
corruption.

### Why now (ordering)
After Item 1 there is **one** Frozen shape to formalize (single-file `Frozen` and
`GraphFrozen` converge), and Item 3 gives the typed `Layout`/`Type` the view references.
Naturally paired with Item 4 (which also wants a typed `frozen`).

### Requirements
- R1: One typed view type (no `anytype`) is the codegen/lower input contract.
- R2: The cross-field invariants that are currently comments (index-space alignment;
  `node_types`↔`layouts` agreement) are encoded as constructor-time assertions or made
  unrepresentable.
- R3: `Driver.Frozen` and `GraphFrozen` converge to one shape (or one + a thin
  single-fn projection via `frozenFor`).
- R4: No behaviour change; determinism / [C11] preserved.

### Premises to re-check
- `Engine.codegen` still takes `anytype frozen`; `Frozen` and `GraphFrozen` are still
  distinct shapes (grep `anytype frozen`, `GraphFrozen`, `frozenFor`).
- `lower.zig` still documents an unvalidated purity contract over its `Inputs`.
- Item 1 has landed (otherwise there are still two shapes — do Item 1 first).

### Invalidate if
- The view was already typed/unified.
- Item 1 was abandoned (two paths remain) — then this item must handle both shapes; reassess
  whether it's still worth it.

### Success criteria
- No `anytype frozen` in the codegen path.
- A constructed-but-inconsistent view (misaligned indices) is caught by an assertion/test
  rather than miscompiling.
- Full suite + VERIFY build unchanged.

### Risk / blast radius
**Medium.** Touches the codegen entry signature and the driver's view construction.
Compile-time-checked; behaviour risk low if purely a typing change.

---

## Item 7 — Query-native pipeline end-to-end (capstone)

> **Status: SHIPPED (staged on `main`, 2026-06-30) via 7a + 7b + 7c.** 7a unified the result shape
> (Item 9). 7b added `Engine.barrier()` (sorted-multiset aggregate key, lookup/OUTPUT-fp split) and
> collapsed `Cache.Phase` ≡ `Dag.Kind` into one enum (`nodeKeyFor` switch deleted). 7c moved every
> stage onto the engine via a declarative `src/query/StageGraph.zig` interpreter (two fan-out regions:
> resolve+check, then codegen behind a hard body barrier) and **deleted the dead Pass-C**
> (`runPassCParallel`/`bodyJob`) + the bespoke hand-sequenced orchestration. Content-fp remains the
> SOLE correctness/cutoff driver; the DAG is scheduling + observability only. Independently verified:
> 462 unit + 42 integration + diff.sh [C11] 69/0 + incremental 5/5 sound+cutoff + `-jN` codegen
> byte-identity (`__text` j1=j4=j8) + no-leftover grep (0 hits). **Note (perf):** the rewrite did NOT
> change the build-time picture — re-measured -jN still *hurts* (per-fn fan-out in the codegen region),
> discover still ~57%. See the **Performance Roadmap** (`PERF_ROADMAP.md`) for P1 (chunking) + P2.

**Phase 3. Cluster:** `src/query/Engine.zig`, `Cache.zig`, `Dag.zig`, `Key.zig` +
`resolve.zig` / `types.zig` / `lower.zig` / `opt/` as they become query producers + the
driver's orchestration.

> **DESIGN OUTCOME (measured, adversarially red-teamed — 2026-06-29).** A design workflow
> produced 4 architectures; chosen = **"pipeline-as-typed-data + one `barrier()` primitive +
> TWO fan-out regions"** (a static `StageGraph` table; `Engine.barrier()` with a sorted-multiset
> aggregate key; collapse `Cache.Phase`≡`Dag.Kind` into one compiler-enforced enum). The red-team
> found the naive *fused* per-fn pipeline FATAL (fusing `checkBody`→`codegen` breaks the codegen
> firewall, which reads the body fp at record time) — so the design was corrected **structurally**:
> codegen is a SEPARATE fan-out region behind a hard body-fp barrier, never fused into resolve→check.
>
> **Open questions answered by measurement (ReleaseFast, examples/bench):**
> - **Q1 — is the caching/parallelism win real? NO.** Cold -j1: discover (lex+parse) **56–64%**,
>   resolve 2–3%, typecheck 5–7%, lower ~29% (codegen compute ~12%), image+sign 4–19%. **`-jN`
>   never beats `-j1` at any size** — because red-green is always on, `types.zig:2246`
>   (`if dag != null or io == null` → serial) makes the "parallel Pass-C" fan-out **dead on every
>   production build**. Caching resolve/typecheck on disk is the WRONG trade (string-heavy tables
>   cost more to serialize than recompute). **Honest motivation for Item 7 is NOT speed.** The real
>   payoffs: (1) ONE mechanism — delete the dead Pass-C fork, the dual init/initDag intent, and the
>   Item 1 projection adapters; (2) **dissolve Item 9 structurally** (the dangling `sigs[].name`
>   becomes impossible); (3) a **red-green cutoff foundation** via barrier nodes — the only payoff
>   that GROWS with project size; (4) kill the silent `nodeKeyFor` phase-drift footgun.
> - **Q2 — [C8] one-tier: RE-AFFIRM.** lower+opt stay fused inside codegen. Cache tiers: disk =
>   {lex, parse, codegen}; memo (in-build, NOT serialized) = {collect, layout, sig, resolve_name,
>   body}. (Red-team flagged a memo-vs-serialized contradiction to resolve before 7c — see risk.)
> - **Q3 — aggregate key:** sort upstream fps by contributor identity, then length-prefixed Wyhash
>   fold (set-order-invariant + flips iff any member fp flips); separate the LOOKUP key from the
>   recorded OUTPUT fp.
>
> **DE-RISKED STAGED PLAN (replaces 7a/7b/7c sketch above):**
> - **7a — result unification + Item 9 dissolution (LOW risk, no engine change):** delete
>   `projectResolve`/`projectTypecheck` + legacy single-source carriers + `Graph.deinitSingle` +
>   the dangling-pointer comment; `GraphResult` is the ONE shape for 1 file and N files. Verifiable
>   by diff.sh byte-identity + incremental.sh. **Standalone; this IS Item 9 as the safe first step.**
> - **7b — engine primitive + enum collapse (MEDIUM):** add `Engine.barrier()` + `aggKey` + the
>   lookup/OUTPUT-fp split; collapse `Cache.Phase`/`Dag.Kind`, delete `nodeKeyFor`'s switch. No
>   stage moved yet. Unit-testable (set-order invariance + flip-on-change).
> - **7c — stages as nodes + StageGraph interpreter (HIGH):** collect/layout/sig as `barrier()`
>   demands; REGION 1 = resolveName+checkBody per-fn nodes; REGION 2 = codegen (separate, behind the
>   body-fp barrier — the red-team constraint). Deletes the dead Pass-C fan-out (R7). Resolve the
>   memo-tier contradiction here.
> - **Item 8 — link as aggregate node (LOW):** closes REGION 2.
> - **7d — cross-stage overlap parallelism (GATED, LIKELY DROP):** only if a measured win appears
>   AND serial discover is parallelized first. Per the measurements, **expected to be dropped.**
>
> **Newly discovered, HIGHER-value perf targets — OUTSIDE Item 7 (tracked below as P1–P3).**

### Why
The point of the query engine is that **dependency tracking + caching + red-green
invalidation + parallel scheduling** are uniform across the compiler. Today they only
apply to `lex`, `parse`, `codegen`. Resolve and typecheck run as **direct, uncached calls
recomputed on every build** (`Cache.Phase.check` exists but stores nothing — "run
in-memory, not stored, M0"); `lower`/`opt` are fused inside the codegen query under the
one-tier rule [C8]. Consequences:

- A one-character body edit re-runs **all** of resolve + typecheck for the whole program,
  every build, even though the engine could skip the unchanged 99%.
- Resolve/typecheck parallelism is a **hand-rolled fan-out inside `types.zig`** (Pass-A
  whole-program barrier, then `Engine.fanOut` for Pass-C bodies) rather than the engine's
  scheduler — a second, parallel mechanism to maintain.
- Red-green stops at the typecheck→codegen firewall: resolve/typecheck *record* DAG edges
  but can't be *skipped*, only re-run.

We want one mechanism, all the way through, so caching/red-green/parallelism are a property
of the engine and not re-implemented per stage.

### The generalisation needed (safe stopping points)
Not every stage transition is the same shape. Some are **global barriers** (must wait for
all units), others are **per-unit pipelines** (a unit proceeds independently). The engine
must express both:

```
  lex(file_i) ─▶ parse(file_i)            per-file, no barrier (already query nodes)
        │
        ▼  BARRIER: every file parsed
  collect-global-symbols                  fan-in: depends on ALL parse(file_*)
        │
        ▼  BARRIER: all top-level decls known
  assign-global-ids + layouts             fan-in: depends on collect + Type-algebra layout
        │
        ▼  global tables ready  ── from here, PER-FUNCTION PIPELINE (no global barrier) ──
  resolve-body(fn) ─▶ check-body(fn) ─▶ lower(fn) ─▶ opt(fn) ─▶ codegen(fn)
        function F may reach codegen while function G is still resolving
```

This is exactly the user requirement: *"every file needs to be parsed before we resolve,
but then … continue to the next compilation stage for a function without waiting for all
the other parts of the program."* The engine today has nested per-unit `query()` and a
flat `fanOut`, but **no first-class notion of a fan-in/aggregate (barrier) node** keyed by
a *set* of inputs, and no scheduler that mixes barrier nodes with per-unit pipelines.

### Requirements
- R1 (engine — barriers/fan-in): The engine supports an **aggregate query node** whose key
  is derived deterministically from a *set* of upstream results (e.g. collect-globals keyed
  by the multiset of module parse fingerprints), cached and red-green'd like any node.
- R2 (engine — scheduling): A stage transition is declared as either a **barrier** (await
  all units) or a **per-unit edge** (proceed independently). The scheduler honours both;
  per-function stages after the global-tables barrier pipeline without a global join.
- R3 (resolve → query): Resolve runs as query node(s) — a global collect/id-assignment
  barrier node + per-function name-resolution nodes — cached and red-green'd. The direct
  `Resolve.resolve` / `ResolveGraph.resolveGraph` call path is removed.
- R4 (typecheck → query): Pass-A (registration + layout) becomes barrier node(s); per-fn
  body checks become per-function query nodes (replacing the bespoke `fanOut` Pass-C).
  Results cached; `Cache.Phase.check` becomes a real stored tier (or splits into finer
  phases — see open question on granularity).
- R4b (Item 9 — one result shape): the canonical cacheable resolve/typecheck result becomes
  *the* front-end result shape; the Item 1 `projectResolve`/`projectTypecheck` adapters +
  legacy single-source `Result` carriers + the `Graph.single` projection are removed, so there
  is no translation layer and no dangling-but-safe ownership reasoning. (See Item 9.)
- R5 (lower/opt → query): `lower` and `opt` become query nodes so red-green covers them,
  resolving or explicitly reaffirming the one-tier [C8] decision (see open question).
- R6 (determinism): Byte-identity [C11] and stable parallel output (`-jN`) preserved
  throughout; the scheduler must not introduce order-dependent results.
- R7 (single mechanism): The hand-rolled parallel fan-out inside `types.zig` is gone;
  parallelism comes from the engine scheduler. (Dead-code sweep D1–D4 applies.)

### Premises to re-check
- Resolve/typecheck still bypass the engine (grep: `Resolve.resolve(`/`checkGraph(` called
  directly in `driver/`; `resolve.zig` has **0** `Cache`/`Engine` refs).
- `Cache.Phase.check` still exists-but-stores-nothing ("run in-memory, not stored").
- Pass-A is still a whole-program barrier and Pass-C still uses `Engine.fanOut`, not query
  nodes (grep `fanOut` in `types.zig`).
- `lower`/`opt` are still fused into the codegen query under one-tier [C8].
- Resolve/typecheck results are still **serialisable** (Resolution is a 32-byte union;
  Type is byte-foldable; sigs/layouts are flat) — required to cache them. *If a future
  result type holds pointers/arenas, caching needs a serialisation story first.*

### Invalidate if
- The engine was already generalised with barrier/fan-in nodes (then 7a is done — verify).
- Resolve/typecheck already route through the engine (then 7b is done).
- A measurement shows resolve+typecheck are a negligible fraction of build time **and**
  per-function parallelism past the global barrier yields no real speedup — i.e. the cost
  of more cache tiers + invalidation edges outweighs the reuse. **Measure before 7b/7c**
  (the existing `--timings` flag): if typecheck is already <1% as memory notes suggest,
  the *caching* win is small and the case for 7 rests mainly on **uniform parallelism +
  one mechanism**, not on skipping recompute. Re-justify on that basis or descope.

### Success criteria
- A body-only edit to one function recompiles only that function's per-fn chain; the
  unchanged functions' resolve/typecheck/lower/codegen are served from cache (observable
  via `--query-stats` / `--dump-dag`).
- No direct resolve/typecheck call path remains; no bespoke `fanOut` in `types.zig`.
- Function-level pipelining is observable: under `-jN`, a function reaches codegen before
  siblings finish resolving (e.g. via a scheduling trace/test).
- Full unit + integration suites pass; `--force --verify` byte-identity holds at `-j1`
  and `-j4`.

### Risk / blast radius
**Highest in the roadmap.** Touches the engine core, the scheduler, every front-end stage,
and the driver's orchestration; regressions here are cache miscompiles or nondeterminism.
Mitigations: land it **after** Item 1 (one path) and Item 3 (layout factored out); stage
it 7a → 7b → 7c with the full validation suite (incl. `--force --verify` at multiple `-j`)
green between sub-steps; keep the old direct path behind a flag until the query path proves
byte-identical on a corpus, then delete it (dead-code sweep).

### Relationship to other items
- **Requires Item 1** (unify) — do not make two paths query-native.
- **Requires Item 3** (Type algebra) — the layout barrier node depends on a layout module
  separable from the checker.
- **Absorbs Item 4** — the generalised engine API is the facade (R1/R2).
- **Absorbs Item 6** — per-query typed inputs are the typed view (R5).
- **Absorbs Item 9** — making resolve/typecheck query-native (R4/R4b) dissolves the Item 1
  `Graph.single` projection adapters into one canonical result shape.
- **Benefits from Item 5** — query nodes that emit diagnostics use the clean sink.
- **Benefits from Item 8** — once the back-end emits through the sink too, new query nodes
  (lower/codegen) inherit one diagnostic mechanism instead of the `ArrayList(Diagnostic)` pattern.

---

## Item 8 — Extend the diagnostics sink to the back-end (and parse)

**Phase 0 follow-up (or fold into 7c). Cluster:** `src/lower.zig`, `src/codegen/CodegenIr.zig`,
`src/driver/Driver.zig` (the lower/codegen result struct), `src/query/Engine.zig`,
`src/parse.zig`.

### Why
Item 5 unified the **front-end** (resolve+typecheck) diagnostics onto `DiagnosticSink`, but
its stated scope stopped there. The **back-end and parse stages still hand-build
`std.ArrayList(Diagnostic)` and hand-manage `owned_msgs`** — so the repo currently has *two*
diagnostic mechanisms. That is the exact "smeared ownership / convention-not-structure"
problem Item 5 set out to remove, just relocated to lower/codegen. Closing it gives **one**
diagnostic mechanism repo-wide (consistent ownership, ordering, tagging) and is a natural
prerequisite for Item 7: when `lower`/`codegen` become query nodes (7c), their diagnostics
should already flow through the sink rather than the ad-hoc list.

### Requirements
- R1: `lower` and `codegen` emit through `DiagnosticSink` (static via `emit`, formatted via
  `emitFmt`). No stage-local `ArrayList(Diagnostic)` remains for emission.
- R2: The Driver's lower/codegen result drops its hand-managed `owned_msgs` field in favour
  of the sink's `Owned` handoff (or carries the sink directly), so no caller does manual
  message frees on that path.
- R3: Reconcile `parse.zig`'s single `*?Diagnostic` out-param — either keep its one-shot
  shape (documented as an intentional exception) or route it through the sink. Decide
  explicitly; do not leave it ambiguous.
- R4: Behaviour-preserving — identical diagnostics, identical ordering, [C11] byte-identity
  and `-jN` determinism preserved (codegen diagnostics participate in cached query results).
- R5: Dead-code sweep (D1–D4): remove the old `owned_msgs` field + manual free loops and any
  now-unused `ArrayList(Diagnostic)` plumbing once migrated.

### Premises to re-check
- The back-end still hand-builds diagnostics (grep `ArrayList(Diagnostic)`, `out_diags`,
  `owned_msgs` outside `diagnostics/` — see the Item 5 caveat list for the current sites).
- `DiagnosticSink` is still the canonical sink (Item 5 shipped, not reverted).
- Whether back-end diagnostics are mostly **static** messages (e.g. "frame too large", "too
  many parameters") — if so the *ownership* win is small and the case rests on **consistency
  + readiness for Item 7**, not on fixing leaks. Re-justify on that basis if so.

### Invalidate if
- Item 7 (7c) already migrated lower/codegen onto query nodes *and* moved their diagnostics
  to the sink — then this is done as a side effect; verify and close.
- A decision is made that the back-end's one-shot/static diagnostics are fine as-is and the
  consistency win does not justify the churn — then descope to just R2 (drop the Driver's
  manual `owned_msgs`).

### Success criteria
- No `ArrayList(Diagnostic)` / `owned_msgs` / manual message frees outside `diagnostics/`.
- One diagnostic mechanism across lex→…→codegen; full suite + corpus/[C11] + incremental
  soundness green.

### Risk / blast radius
**Low–medium.** Mostly mechanical; codegen diagnostics are largely static (low ownership
risk). The one subtlety is that codegen diagnostics ride inside cached query results, so the
migration must not perturb byte-identity — guard with `examples/diff.sh` and
`examples/incremental.sh`.

### Relationship to other items
- **Follows Item 5** (extends the same sink).
- **Feeds Item 7** (7c query nodes emit through the sink) — may be folded into 7c, or done
  earlier as standalone cleanup since it has no dependency on Items 1/3.

---

## Item 9 — Complete the front-end unification (remove the graph-of-one adapters)

> **Status: SHIPPED via 7a (uncommitted, staged on `main`).** Both `GraphResult`s are stored
> WHOLE on `FileResult` (sibling fields, one shared lifetime); `projectResolve`/`projectTypecheck`
> + both legacy single-source `Result` carriers DELETED. Ownership is now STRUCTURAL — the resolve
> result solely owns the fn-name strings, typecheck `sigs[].name` borrows them, and the owner
> outlives every borrower by construction (no free-boundary, the dangling-but-safe alias is
> impossible, not just harmless; `Sig.name`/`sigs` doc'd as borrowed). `Graph.deinitSingle` kept
> (deleting it would double-free the borrowed 1-element spine — a deliberate, justified deviation
> from the original acceptance line). Net −105 source lines; NO engine/cache/dag/codegen change.
> Independently verified: 458 unit + 42 integration + corpus [C11] 69/0 + incremental soundness 5/5;
> byte-identical (the single-module codegen name stays bare via `buildGraphNames`, untouched).

**Folded into Item 7 / 7b. Cluster:** `src/driver/Driver.zig` (`projectResolve` / `projectTypecheck`
+ the `Resolve.Result` / `Typecheck.Result` legacy carriers) + `Graph.single` + the single-source
readers (`printFailure` / `dumpCheck` / `lowerProgram`) + the inline test helpers.

### Why
Item 1 deleted the single-file *forks* but, on the minimal-churn lens, kept the single-source
`Resolve.Result` / `Typecheck.Result` structs as **carriers** and added `projectResolve` /
`projectTypecheck` to fold the one-module `GraphResult` back into them. That leaves residual
friction:
- **Two result shapes** for the same data (`GraphResult` per-module arrays ↔ the flat
  single-source `Result`) plus a shallow adapter translating between them.
- **Fragile ownership-by-convention**: `projectResolve` frees the fn-name table while a projected
  `sigs[].name` still aliases it — a deliberate "dangling-but-safe because single-source codegen
  never reads it" pointer, held together by a comment (`Driver.zig` ~249–256). This is exactly the
  ownership-by-convention hazard the deep-module work removes.

This is real, low-grade friction (not a bug — verified leak-clean), worth removing so the front-end
has **one** result shape end-to-end.

### Why now (folded into 7b)
Item 7's 7b makes resolve+typecheck **query nodes**, which forces a single canonical, cacheable
result shape flowing through the engine/cache. That canonical shape should be *the* shape — so the
projection adapters + legacy carriers dissolve naturally as part of 7b rather than as a separate
pre-step. Doing it standalone first would just be re-plumbing the same readers twice.

### Requirements
- R1: One front-end result shape consumed by all readers (`printFailure`/`dumpCheck`/`lowerProgram`)
  and tests. `projectResolve`/`projectTypecheck` and the legacy single-source `Result` carriers are
  removed (or become the canonical query-node result, with no translation layer).
- R2: No "dangling-but-safe" aliasing reasoning survives — ownership is structural.
- R3: Behaviour-preserving — identical output for single-file and multi-file; [C11]/-jN determinism.

### Premises to re-check
- The adapters still exist (grep `projectResolve`/`projectTypecheck`/`Graph.single`).
- 7b is the active vehicle. If Item 7 is deferred, this can be done standalone (re-plumb the ~3
  readers + test helpers to the `GraphResult` shape directly) but the cost/benefit is lower.

### Invalidate if
- 7b already removed the adapters as part of making resolve/typecheck query-native (then done).
- A decision is made that the single-source `Result` carrier is the desired *canonical* shape and
  the graph path should project *into* it everywhere — then keep the carriers and instead delete the
  duplication on the other side. (Pick one canonical shape; do not keep both + an adapter.)

### Success criteria
- No projection adapter / dual result shape in the front-end; one shape, structural ownership.
- Full suite + corpus/[C11] + incremental soundness green.

### Risk / blast radius
**Low–medium.** Touches the front-end result type + ~3 readers + test helpers; behaviour-preserving,
compile-checked. The subtlety is the freed-vs-borrowed sig-name ownership — the unified shape must
make that unambiguous.

---

## Cross-cutting validation (every item)

Run before and after each item; an item is not "done" until all are green:

| Check | Command |
|-------|---------|
| Unit tests | `zig build test-bin` then `./zig-out/bin/toyc-test` (`zig build test` deadlocks) |
| Integration tests | run `./zig-out/bin/toyc-integration-test` (query engine + DAG firewall) |
| Determinism / no miscompile | `--force --verify` build of `examples/bench` is byte-identical ([C11]) |
| Incremental soundness | edit-propagation: cached rebuild matches clean rebuild |

For behaviour-preserving items (1, 3, 6): snapshot a corpus of compiled outputs
(types/diagnostics/binaries) before, diff after — zero diff is the bar.

### Dead-code sweep (required after every item)

Deepening **removes** code: Item 1 strands the old single-file functions and may delete
whole `_graph` files; Item 3 leaves the moved-from sites; Item 5 removes `owned_msgs`
plumbing; Items 4/6 drop the `anytype` duck-typing helpers. **Zig will not catch most of
this** — unused *locals* and *parameters* are compile errors, but unreferenced top-level
`pub`/private declarations and no-longer-imported files compile clean. An item is not done
until its now-dead code is gone, not just unreferenced.

- D1: No orphaned files — every `src/**/*.zig` is reachable from a root import (`root.zig`,
  the exe, or a test artifact). After Item 1 specifically, confirm `resolve_graph.zig` /
  `types_graph.zig` are either deleted or contain only live adapter code.
- D2: No unreferenced top-level decls left by the refactor. Sweep: for each `pub fn`/`fn`/
  `const` touched, grep the tree for references; zero hits outside its own file (and not a
  test/entry export) ⇒ delete. Pay special attention to the *old* path's entry points
  (e.g. the single-file `check`/`resolve` once graph-of-one subsumes them).
- D3: No dead struct fields — fields that the deepened module no longer reads (e.g. the
  per-`Result` `owned_msgs`/`diag_mods` after Item 5).
- D4: No stale comments/invariant tags referring to removed machinery — grep the `[C7]`
  "mirror" notes after Item 2, the "PURE of its frozen inputs" note after Item 6, etc.

Sweep at the **end of each item** (not just at the roadmap's end) so dead code never
accumulates across phases and a reviewer of one item's diff sees the removals inline with
the additions.

## Performance backlog (discovered during the Item 7 design — 2026-06-29; RE-BASELINED 2026-06-30)

The Item 7 design measured the build and found the real time sinks are NOT what Item 7 touches.
**Re-baselined 2026-06-30** (release, 1240 fns) after the red-green reuse-reversal (content-fp is now
the sole correctness driver; the DAG is debug-only) + the nodeFp O(N²)→binary-search fix:

- **P1 — `-jN` does not help; it HURTS. STILL OPEN (root-caused 2026-06-30).** Re-measured: -j1 20.5ms
  → -j2 21.5 → -j4 23.5 → -j8 27.8ms (more threads = slower).
  - **Root cause = dispatch GRANULARITY, not the typecheck logic.** `Engine.fanOut` submits **one task
    per function** (e.g. 10,201 tasks for 10,201 ~21-line fns); each task is µs of work, so per-task
    overhead (serial `group.concurrent` enqueue on the main thread + the `Io.Group` completion atomic +
    worker dequeue + completion signal) **≳ the work**. Three compounding costs: (1) **serial dispatch
    tax** — the submit loop runs 10,201× on the main thread regardless of N; (2) **hot shared-atomic
    contention** — µs workers finish instantly and pound the shared `Io.Group` counter + `Io.Threaded`
    queue; those cache lines ping-pong across cores, and contention RISES with N (a *slowdown*, not just
    "no speedup", is the signature of cache-line bouncing); (3) **allocator churn ×N-tasks** — each unit
    builds a fresh `BodyChecker` (slot_types/loop_stack ArrayLists from empty), a `DiagnosticSink`, and
    per-construct field-seen bitsets, churning per-CPU allocator caches across N threads (plus a
    cross-thread free when a sink init'd on main is deinit'd on a worker).
  - **Fix = chunk, don't de-parallelize.** Split the units into ~`ncpu` contiguous ranges and dispatch
    **one task per range** (each task loops its range serially): ~10,201 dispatches → ~8, per-task
    overhead amortized, shared-atomic traffic cut ~N-tasks/ncpu → contention stops and it actually
    scales on cold/`--force` builds where there's real work. Below a small unit-count threshold, stay
    serial. **Per-stage heuristics** (different work-per-unit ⇒ different chunking): typecheck bodies
    (tiniest unit, worst ratio), codegen (more work/unit; warm = cache-hits so nearly as bad), the
    per-file front-end (P2) — each needs its own threshold + chunk size. NOTE: 7c restructures the
    per-fn `fanOut` into the StageGraph two-region scheduler, so this fix lands on the *post-7c* dispatch
    shape; the principle (chunk per ~ncpu range) is unchanged. Re-locate exact call sites post-7c.
- **P2 — serial `discover` (lex+parse) is THE dominator. STILL OPEN (now the #1 lever).** Re-measured:
  cold **47.8%**, warm **59–64%** of build, single-threaded — and it does NOT shrink on warm rebuilds
  (it re-reads+re-lexes+re-parses every file). Parallelizing/​caching discovery is the highest-leverage
  speed win and is independent of Item 7. **Needs RESEARCH** (not a known fix like P1): how to
  parallelize module discovery+lex+parse given the import-graph dependency (each file's imports gate
  which files to discover next) — e.g. parallel-BFS over the import frontier, speculative parse of
  already-read files, or a worklist of discovered-but-unparsed files fed to a chunked fan-out; plus
  whether warm discovery can skip re-lex/parse via the content cache (it currently doesn't shrink warm).
  Scope a research+design pass before implementing.
- **P3 — warm-rebuild overhead: RESOLVED ✅** (by the reuse-reversal + binary-search fix). Re-measured
  warm (hot cache, no edit): `lower` compute = 0, cache-get = **0.44ms** (was ~36ms of GraphFrozen +
  serial O(N²) red-green pre-pass). Warm total ~21–26ms, now bounded by P2 (discover), not the
  pre-pass. No further action.

> **Item 7 reassessment (2026-06-30):** with reuse REVERSED (content-fp sole driver, DAG debug-only),
> the byte-level **cutoff already works via the content-fp cache** (incremental.sh: 5/5 sound + cutoff).
> So Item 7's "red-green cutoff foundation" payoff is now largely **already delivered** — weakening the
> case for 7b/7c further. Item 7's remaining honest value is mechanism-unification + footgun removal
> (P1's dead Pass-C, the phase-drift `nodeKeyFor`). Worth revisiting whether 7b/7c earn their risk, or
> whether the better next move is **P2 (parallelize discover)** — the one measured, growing lever.

## Out of scope (explicitly deferred)

- Generics, pattern-match typechecking — language features, not architecture.
- Second backend / register allocation (slot-coloring) — performance, tracked elsewhere.
- Golden-binary tests for `emit.zig` / `CodeSign.zig` — desirable but independent of these
  deepenings; can follow Item 6.

## Open questions

- Item 1: does single-file path exist for **speed**? Measure graph-of-one single-file compile
  time before committing — if it regresses, keep both or invert (delete single-file).
- Item 1 vs 4: confirm unify (1) lands before facade (4); if 1 is deferred, re-scope 4 to
  wrap both paths.
- Items 4/6: settle the shared typed `frozen` contract once, used by both — decide owner
  module before starting either.
- Item 7 (cache granularity vs [C8]): one-tier cache deliberately builds IR *inside* the
  codegen query. Making `lower`/`opt` separate cached nodes (R5) adds tiers + invalidation
  edges. Decide: finer tiers (more reuse, more bookkeeping) vs keep one-tier and accept
  lower/opt re-run whenever codegen does. Re-affirm or retire [C8] explicitly.
- Item 7 (is the caching win real?): if `--timings` confirms resolve+typecheck are <1% of
  build time, the justification shifts from "skip recompute" to "uniform parallelism + one
  mechanism." Confirm the motivation holds before committing 7b/7c.
- Item 7 (aggregate-node key): how is a fan-in/barrier node keyed stably over a *set* of
  inputs so it's order-independent and red-green-correct (e.g. sorted multiset of upstream
  fps)? Settle before 7a.
