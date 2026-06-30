# Contract ledger

The single authoritative home for the `[C1]..[C11]` contract IDs referenced from
source comments. These tags guard determinism, warm-cache soundness, and
miscompile-class invariants; a bare `[Cn]` in the code is a cross-reference to the
definition below. The tags double as a cross-file grep affordance: `grep -rn '\[C8\]'`
finds every enforcing site **and** this entry.

Most sites restate their invariant in adjacent prose, so the tag is a redundant
cross-ref there. A handful of sites are **sole carriers** — the prose does *not*
restate the invariant, so the tag (or its inlined meaning) is load-bearing and must
survive comment debridement. Those are called out per entry.

## `[C1]..[C11]`

| ID | Invariant | Enforced in | Sole-carrier (do not strip) |
|---|---|---|---|
| **C1** | A callee **body** change must NOT invalidate its callers: their transitive fingerprint is unchanged, so the cutoff fires and only the edited fn recompiles. The complement of `[C2]`. Breaking it wastes recompiles. | `query/Fingerprint.zig`, `symbols/Sig.zig` | — |
| **C2** | A callee **signature** (param + return types) change MUST flip every caller's transitive fingerprint, so the caller re-lowers. The body/sig split is the point of M5 incremental codegen. Breaking it serves a caller lowered against a stale callee signature — a silent miscompile. | `query/Fingerprint.zig`, `symbols/Sig.zig` | — |
| **C3** | The structural body fold sees only node **tags** + **leaf token text** (length-prefixed) + **index-free** type-layout descriptors — never an absolute node index or source offset — so a fn's fingerprint is position-independent: editing or reordering a sibling cannot shift it. Breaking it gives spurious invalidation or, worse, a stale hit. | `query/AstWalk.zig`, `query/Fingerprint.zig` | `query/AstWalk.zig:61` — the only statement that leaf text is *length-prefixed* (the precondition that keeps adjacent leaves from colliding). |
| **C4** | The linker resolves every fn handle and `call26` target **by name, single-threaded, before** the parallel fan-out (the symbol interner's `get` is not thread-safe), and a call's `bl` delta is computed from the target's name-resolved offset, not its layout slot. So reorder / `-jN` never miscompiles a call, and a stale target surfaces as `UnresolvedSymbol` at exactly the point the old serial loop raised it. | `link/Link.zig` | — |
| **C5** | The body walk emits a `count: u32` arity/length sentinel for **every** variable-length child group (block stmts, fn params, struct fields, enum variants, tuple arity, match arms, pattern alternatives, call args), so a changed count flips the fingerprint (`f()` ≠ `f(0)`, and likewise for each other group). Generic — not limited to call arity. | `query/AstWalk.zig`, `query/Fingerprint.zig` | — |
| **C6** | The emitted `SymName{kind,name}` is part of the lowered bytes, so it must enter cache identity: a **callee's** `{kind,name}` folds into its caller's transitive fingerprint, and a fn's **own** emitted (possibly module-qualified) name folds into its codegen cache key via `symMix`. Folding only the bare body/sig would let a shadow/unshadow rebind (builtin `print` vs a user `fn print` of identical sig) or a single-file-vs-graph name (`add` vs `m.add`) alias to one slot → stale-cache miscompile / `UnresolvedSymbol` at link. *(Was the unnumbered `[Cx]` placeholder; `C6` was otherwise unused.)* | `query/Fingerprint.zig`, `query/Key.zig`, `symbols/Sig.zig` | — |
| **C7** | Children fold in canonical **LHS→RHS wiring order, never a commutative XOR**, so `a - b` ≠ `b - a`; **and** a presence/shape `flag` sentinel is folded (bare `return` vs `return v`, has-else, inferred-vs-qualified, guard, rename-vs-pun, has-subpattern, has-ret-type) so structurally-distinct-but-same-token constructs cannot collide. Breaking either lets a semantically different edit collide on the cache key. (Arity is `[C5]`, not `[C7]`.) | `query/AstWalk.zig`, `query/Fingerprint.zig` | `query/AstWalk.zig:68` — the only link of the optionality/shape `flag` field to this invariant. |
| **C8** | The codegen cache is **one-tier**: the M12 IR (and the in-place M13 opt) is built and freed entirely inside the per-fn `lowerOne` query and never escapes or is cached separately; only the packed aarch64 `FnCode` blob is keyed by content fingerprint and cached (then freed in the uncached relink tail). Breaking it double-caches/leaks IR or serves stale bytes. | `query/Cache.zig`, `query/Engine.zig` | `driver/Driver.zig:360` — the relink/free tail's prose says nothing about one-tier; the bare `[C8]` is the only tie from "`FnCode` is the sole cached/owned artifact" back to this invariant. |
| **C9** | Every `FnCode` is **"always own"**: all of its names/bytes (own `sym.name`, each `.func`/`.import` reloc target name, the code, the literals) are unconditionally heap-owned regardless of provenance — freshly lowered (Codegen dupes at emit) or unpacked from a disk cache hit — so one uniform `deinit` frees it with no owned-flag and no double-free ambiguity. Breaking it double-frees borrowed memory or leaks. | `link/Link.zig` | — |
| **C10** | The compilation target is folded into the on-disk cache key for, and only for, `Phase.targetSensitive()` phases (only codegen), so aarch64 machine-code blobs never alias across targets while target-independent phases (lex/parse) share one entry. Breaking it serves a blob built for one target as another's — a silent miscompile / link-ABI fault. | `query/Key.zig`, `query/Cache.zig` | `query/Phase.zig:63` — the only statement of the "aarch64 blobs must not alias across targets" soundness rationale. |
| **C11** | **Determinism / byte-identity** (== `LOCK #7`). The same source always lowers to the same per-fn `FnCode` bytes regardless of dispatch (`-j1` vs `-jN`) and cache state, because every stage is pure over its inputs, ids are assigned in monotonic source order, and parallel units own disjoint write spans. Enforced by `--verify` (re-lower twice on a miss / compare to the stored blob on a hit — a real runtime check, not a debug assert compiled out under ReleaseFast). | `query/Engine.zig`, `driver/Graph.zig`, `ir/Ir.zig`, `codegen/*`, `opt/{fold,dce}.zig`, `types.zig`; gate: `examples/diff.sh` | — |

## `LOCK #N` — M12 IR-backend design locks

A separate, numbered family of IR design decisions — *not* the `[Cn]` cache/determinism
contracts. Ruling: the **gate** locks (`#1`, `#7`) are owned by the harness
`examples/diff.sh`; the **design** locks (`#2`/`#3`/`#4`) are self-defined inline at each
site (no central ledger); `#6` marks dead code.

| Lock | Meaning | Home |
|---|---|---|
| **#1** | A compiled program is output-identical (exit code + stdout match its `# expect:` directives) AND deterministic. | `examples/diff.sh` (definition) |
| **#2** | Value model: scalar / control-flow merge values are single-definition SSA `Value`s; named locals and all aggregates live in memory slots; merges carried as block params via store-before-br ("phi = memory"). | inline (`lower.zig`, `ir/Ir.zig`, `codegen/CodegenIr.zig`, `codegen/frame/FrameLayout.zig`) |
| **#3** | The ABI (register/stack placement, x8 sret, NGRN/NSAA) is decided in exactly one place — codegen's `Abi` module; the IR carries no registers, offsets, or calling convention. | inline (`ir/Ir.zig`, `codegen/abi/Abi.zig`) |
| **#4** | Spill-everything frame layout: every slot AND every value gets its own never-reused, type-sized, 8-byte-aligned frame cell in one deterministic IR-order pass. | inline (`codegen/frame/FrameLayout.zig`, `codegen/CodegenIr.zig`) |
| **#6** | The `smod` / `%` modulo opcode sketch, retained per the design but DEAD (the source has no `%`). | inline (`ir/Ir.zig`) — scheduled for removal |
| **#7** | VERIFY byte-identity (== `[C11]`): re-lowering every fn twice asserts per-fn `FnCode` byte-identity. | `examples/diff.sh` (definition) |

## `[design N]` tags

`[design N]` references point at out-of-repo design docs and are **not** contracts —
they carry no in-repo spec a reader can resolve, and are scheduled for comment
debridement. Do not treat them as a checkable invariant.
