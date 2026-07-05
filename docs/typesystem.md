# Type system

The reference for `toyc`'s type checker (`src/types.zig`), run after name
resolution. Inference is **bidirectional and local over concrete types**: there
are no type variables, no polymorphism, no generics. Every expression has a
fully concrete `Type` the moment it is checked.

## Types

`Type` is a byte-foldable struct (a `Kind` byte plus two id fields), never a
tagged union — codegen, the fingerprint, and the red-green cache all depend on
it being trivially comparable and hashable.

| Kind | Spelling | Notes |
|------|----------|-------|
| `int` | `int` | machine integer |
| `bool` | `bool` | |
| `str` | `str` | first-class string |
| `unit` | `()` | the empty type; one value, also spelled `()` |
| `@"struct"` | a struct name | carries `struct_id` into the struct table |
| `@"enum"` | an enum name | carries `enum_id` into the enum table |
| `never` | — (not spellable) | the bottom type; a value that never exists |
| `invalid` | — (not spellable) | the poison/error type |

Two special types are not user-spellable:

* **`never`** is **bottom**. A `never` expression (a break-less `loop`, a call to
  a diverging path) never yields a value, so it fits wherever any value is
  wanted. It is the *identity* of the `merge` join.
* **`invalid`** is **poison**. The first time an operand is `invalid`, the
  containing expression silently becomes `invalid` too, so one root mistake
  produces exactly one diagnostic and never cascades. It *absorbs* in both
  `assignable` and `merge`.

## Bidirectional checking: synth / check

Checking is two cooperating modes over a known, concrete world:

* **synth(e) → T** — bottom-up. Read the type of `e` from its parts. Literals are
  obvious; an identifier takes its declaration's type; `+ - * /` are `int → int`;
  comparisons and `== !=` yield `bool`; a unary `-`/`!` constrains its operand; a
  call yields the callee's declared return; a struct/enum init yields its named
  type.
* **check(e, T)** — top-down. Push a known *expected* type `T` into `e`. This is
  how an inferred placeholder (a bare enum variant `.V`, a struct literal) resolves
  against the slot it flows into (a parameter, a field, a return).

In the code these are `typeOf` (synth) and `typeOfExpected` / `typeOfBlockExpected`
(check). This milestone formalizes the previously ad-hoc one-shot `expected`
parameter into this synth/check split; it adds no inference power.

## Assignability — the one relation

> Is a value of type `got` acceptable where a `want` is expected?

```
assignable(want, got) :=
      want.kind == invalid  -> true     (poison absorbs; already reported)
   or got.kind  == invalid  -> true     (poison absorbs)
   or got.kind  == never    -> true     (bottom fits any slot)
   else                     -> eql(want, got)
```

Assignability is **structural equality with NO implicit coercion** (`int` is not
assignable to `bool`, a `struct A` is not assignable to a `struct B`). It is the
single relation behind every "expected X, got Y" check:

| Site | want | got |
|------|------|-----|
| call argument | parameter type | argument type |
| `name = e` assignment | place type | rhs type |
| `return e` / bare `return` | declared return | return-value type |
| struct-init field | field type | value type |
| enum-variant payload (tuple/struct) | declared payload type | value type |
| non-unit fn fall-off | declared return | trailing-expr type |

Callers guard with `if (!Type.assignable(want, got)) emit(...)`, so the
poison/`never` early-`true`s collapse the old scattered `kind != .invalid` /
`kind != .never` manual guards into the relation itself.

## Join — `merge(a, b)`

`merge` is `assignable`'s sibling: the **join** used when two control-flow
branches must agree on one result type (`if`/`else`, `match` arms, `break`
values).

```
merge(a, b) :=
      a == invalid or b == invalid -> invalid          (poison absorbs)
   or a == never                   -> b                (never is identity)
   or b == never                   -> a
   or eql(a, b)                    -> a                (agreement)
   else  emit "branches yield different types (A vs B)" ; invalid
```

`merge` stays **verbatim**; `assignable` does not replace it. They share the same
lattice discipline (invalid absorbs, never is the unit, `eql` otherwise) viewed
from two angles — assignability is directional (want ← got), join is symmetric.

## Statement and declaration rules

* `name := e` — infers `name : synth(e)`. `e` must produce a value; `()` is an
  error (nothing to bind).
* `name: T = e` — explicit typed local. Parse `T`, `check(e, T)`, bind `name : T`.
* `name = e` — assignment. `assignable(typeof(name), synth(e))`.
* `return e` / bare `return` — `assignable(declared_ret, T)` where `T` is the
  value type (or `()` for a bare return).
* Every `return e` **and** the body's trailing expression are checked against the
  declared return type.

## Function signatures

* **Parameters must be user-typed** — `fn f(x: int)`. *(documented existing.)*
  The parser enforces this: `parseFnDecl` does `expect(.colon, ...)` then
  `parseType()` for every parameter (`src/parse.zig`), so a typeless parameter is
  a parse error, never reaching the checker.
* **Return type — unit sugar.** Omitting `-> T` is **sugar for `-> ()`**. There is
  no return-type inference: a function with no arrow returns unit, and any
  non-unit return value must be annotated. *(`decodeFnSig` defaults a missing
  return-type node to `Type.unit`.)*
* **`main` must return `int` or `()`** — any other return type (bool/str/struct/
  enum) is rejected with a diagnostic at `main`'s name token. Enforced in the
  checker as a Pass-A rule (`checkMainReturn`): it reads only the frozen fn
  signatures, so it never blocks Pass-C body parallelism, and its diagnostic is
  emitted into the same `(module, byte_offset)`-sorted stream as every other type
  error (byte-identical at `-j1` and `-jN`). The entry is the first fn named
  `main` in the entry module, matching codegen's own entry selection. The
  codegen-entry guard in the driver is a defense-in-depth backstop.

## Unit `()`

`()` is spellable in both positions: *(documented existing — already parsed.)*

* as a **type** — `fn f() -> ()`,
* as a **value** — `return ()`.

A function whose return type is `()` is checked in statement context (its body
need not supply a trailing value).

## Kept-unchanged rules

These rules are unchanged by the formalization and remain authoritative:

* **struct / enum** declaration, field, and init checking;
* **match exhaustiveness** — every variant covered (or a wildcard);
* **loop value** — a `break v` join type, `never` for a break-less loop;
* **divergence** — `never` propagation through diverging control flow;
* **definite return** — a non-unit fn returns on every path or supplies a typed
  trailing expression;
* **pub-signature coherence** — a `pub` symbol's signature is consistent across
  the program.

## Generics syntax reservation (M1)

The generics front-end (`fn f[T, U](..)`, `struct Box[T]`, `enum E[T]`, a type
application `Box[int]`, explicit call type-args `f[int](..)`) landed append-only in
M1. Generic **functions** (M2) and generic **structs** (M4) now have semantics (see
below); only a generic **enum** decl still emits **T0013 "generics not yet
supported"** (M6).

The `[..]` bracket is disambiguated by **position**, and this rule is reserved so a
future value-index never collides:

* **type position** — a trailing `[..]` on a type-ref is a **type application**
  (`x: Box[int]`, `mod.Box[int]`);
* **postfix-call position** — a `[..]` immediately after a name / qualified
  `mod.fn` callee is **explicit type-arguments** (`id[int](7)`); it wraps only that
  callee, and the following `(..)` forms the call.

**Value indexing `v[i]` is deliberately NOT parsed** — it is reserved to a distinct
future form so it can never collide with type-application. Only an `identifier` or a
`field_access` base is wrapped; any other `[..]` is a syntax error today.

A third `[..]` position lands in M4: **construction position** — a `[..]` on a name
followed by `{ .. }` (`Box[int]{ v: 1 }`) builds a `struct_init` whose lhs is the
`type_app` (no new `Node.Tag`, so `ParseHeader.version` is unchanged).

## Generic structs + the composite `App` type (M4)

Generic **functions** (M2) and generic **structs** (M4) are monomorphized to
concrete value types — there are no runtime dictionaries, no code sharing, and the
byte-foldable `Type` never widens. Only a generic **enum** decl remains gated with
T0013 (M6).

A generic-struct application `Box[int]` is a **check-time composite type**:
`Kind.app` (appended, frozen ordinal) reusing `Type.struct_id` as an index into a
content-addressed intern table (`symbols/Composite.zig`). Structurally-equal
applications intern to the SAME index within a run, so `Type.eql` comparing two
`.app`s by that index is exactly structural equality. `App`s are formed while
checking (`refs.typeFromTypeApp`), used to type construction (`Box[int]{ .. }`) and
field access (`b.v` reads the field pattern substituted through the args), and admit
a **ground** `App` as a monomorphization type-arg.

**Reification (the mono tail, serial):** every reachable ground `App` is minted a
FRESH ordinary `struct_id` — its field patterns substituted through the concrete
args (a nested `Box[T]` field reifies bottom-up first), its `Layout` registered on
the live tables **before** the snapshot — and every `.app` in every `node_types` /
instance sig / non-generic fn sig is rewritten to that `structT`. So lower / codegen
/ fingerprint / cache see only plain concrete structs; an `App` (or a `type_var`)
reaching `lower` trips a Debug assert. The reified `struct_id` is assigned in an
index-INDEPENDENT structural-key order, so it — and any `s<id>` mangling downstream —
is a pure function of source (`-j1` == `-jN`). The fingerprint folds the reified
LAYOUT (name + fields + offsets), never the id, so which id an instance lands on is
invisible to the cache.

**Explicit type args only.** `Box[int]{ .. }` must name its args; construction-site
inference (`Box{ v: 1 }`) is deferred to M5.

**Termination guard (T0017).** Generic structs are the first construct that makes
unbounded instantiation expressible: `fn go[T](x: T) { go[Box[T]](..) }` forms
type-args of strictly-growing generic-nesting depth. The serial mono worklist caps
that depth and rejects the program with **T0017 "instantiation too deep"** —
deterministically at `-j1`/`-jN`, never hanging or running out of memory — while a
legitimately deep-but-finite generic program still compiles.

## Protocols, the prelude, and builtin-scalar conformance (M11/M12)

`has` is the single conformance relation: `impl T has P { .. }` conforms `T` to a
declared `protocol P`. Coherence is checked **whole-program, serially** (before the
parallel body pass): exactly one impl per `(protocol, type)`, no orphan rules — a
duplicate (even in a sibling module) is **T0020**, a missing method or undeclared
protocol is **T0021**. A protocol stores its method NAMES only; the per-impl
signature-compatibility check (params/return vs the protocol) is **deferred to M13**
(the builtin `Eq` below is signature-correct by construction).

**Prelude (M12).** Protocol/type names that must be universally in scope with no
import (`Eq`, later `Ord`/`Option`/...) are delivered by **native compiler
registration** — the same mechanism as the synthesized `print`, chosen over an
embedded `.toy` module so there is **no new module-graph or content-fingerprint
surface**. `registerPrelude` runs at the head of Phase 0c (serial), giving the `Eq`
protocol global id 0; a bare `Eq` that names no module protocol falls back to that id
in `protocolIdFromNode` (a user protocol of the same name shadows it via first-lookup).

**Multi-space conformance key.** The coherence key spans `(protocol, recv.kind, recv_id)`
— builtin scalar Kinds (int/bool/str/unit — which carry no `struct_id`/`enum_id`) as well
as struct/enum nominals (`recv_id` is 0 for scalars, which the kind byte distinguishes).
(M14 extends this key with the protocol type-args; see the generic-protocols section.)
`registerPrelude` pre-seeds the builtin scalar conformances into the conformance table
and `checkCoherence` seeds its `seen` set from them, so a user `impl int has Eq`
collides with the builtin (T0020). A user impl **may** target a builtin scalar
(`impl int has MyProtocol`), coherence-checked whole-program.

**Builtin `Eq`.** Shipped for **int and bool** only. Dispatch runs through a pure
recognizer `builtinScalarMethod(recv, name)` (not a phantom `t.fns`/`t.methods` entry,
which would desync the `names`/`sigs` parallel arrays): the body checker types
`a.eq(b)` to `bool`, `lower` emits an inline `icmp eq` (no call, no reloc), and the
fingerprint folds a fixed sentinel sig. `str` (needs a heap-free byte-compare — real
backend work) and `unit` (`()` is not a `type_names` scalar) are **deferred**; the key
still spans all four Kinds, so a future user `impl str has Eq` keys correctly and no
builtin blocks it.

## Generic protocols + multi-conformance (M14)

A protocol may carry **type parameters** — `protocol Into[U] { fn into(self) -> U }` —
the associated-type replacement. A conformance names the args: `impl P has Into[int]`.
Because the args are part of the conformance identity, **one type may conform to the
same protocol multiple times** with different args:

```
impl P has Into[int]  { fn into(self) -> int  { self.x } }
impl P has Into[bool] { fn into(self) -> bool { self.x > 0 } }
```

**Conformance key = `(protocol, type, protocol-args)`.** The coherence key is a
serialized byte vector (variable arity forces bytes, over a `StringHashMap`) folding the
protocol id, the receiver kind/id, and each arg's kind/id. So `Into[int]` and
`Into[bool]` on one type do **not** collide, but two identical `Into[int]` still do
(**T0020**). `findConformance` compares the same vector, and the resolved args fold
**structurally** (ordered, never XOR) into the M13 `(e)` fingerprint component — so an
`Into[int]` → `Into[bool]` edit flips exactly the dependent monomorphizations' keys
(no stale-witness miscompile). M14 restricts protocol-args to already-concrete non-`App`
value types (scalar/struct/enum); a composite `App` arg's check-time index is
run-order-dependent, so it is rejected (would break key + fp determinism).

**Ordinal scheme: `Self = type_var(0)`, protocol params = `type_var(1..)`.** When a
protocol method sig is decoded, the synthetic `self` slot is forced to `type_var(0)` and
the protocol's generic params occupy ordinals `1..` (a `""` placeholder reserves index 0
for `Self`, since `""` never matches a source identifier). So `fn into(self) -> U`
decodes to `[type_var(0)] -> type_var(1)`, and `U` never aliases the receiver.
`groundProtoType(ty, recv, args)` grounds `tv(0)→recv` and `tv(k≥1)→args[k-1]` — used
consistently in the protocol-sig decode, the T0024 coherence signature check, and the
bound-as-axiom body dispatch.

**Multi-conformance requires explicit args (T0025).** At a use site `p.into()` where the
receiver conforms to one generic protocol **multiple times** and no protocol type-args
disambiguate, the compiler does **not** pick arbitrarily — it is a use-site error
**T0025** naming the conflicting conformances in source (fn-id) order (deterministic at
any `-jN`). An explicit `p.into[int]()` (parsed as `call(type_app(field_access, [int]))`)
selects the matching conformance. A type conforming **once** (a non-generic protocol OR a
single generic conformance) resolves with **no** explicit args — byte-identical dispatch
to M11/M13. The single disambiguator `resolveConformanceMethod` is shared by all three
consumers (BodyChecker types, `lower` symbols, `AstWalk` fingerprint), so they always
select the identical witness. A `[T has Convert[U]]` bound resolves at the mono worklist
by substituting the bound's args through the instance type-args before `findConformance`.

## Operators: comparison via `Ord` (M16)

The four comparison operators desugar to a **single three-way** compare, `Ord::cmp`,
which returns a prelude enum:

```
enum Ordering { lt, eq, gt }              // native, AST-less; tag = decl index: lt=0, eq=1, gt=2
protocol Ord { fn cmp(self, other: Self) -> Ordering }   // homogeneous (Rhs = Self)
```

Both are delivered by native compiler registration (`registerPrelude`, the `Eq`
precedent): `Ordering` is an AST-less `EnumSym` hand-laid-out at 8 bytes with
`state = .done` (so Phase-0b layout early-returns and never derefs its `Ast.none`
decl node), injected — if-absent, **user-first-wins** — into every module's enum map
(both the resolver's table and the checker's `enum_ids`) so it is nameable with no
import. `Ord` takes protocol id 1 (right after `Eq = 0`). Builtin `Ord` conformances
are pre-seeded for **int / str / bool**.

**Six comparisons from one `cmp`.** The checker types `<`/`>`/`<=`/`>=` to `bool` iff
the (same-typed) operand `conformsToOrd`; otherwise **T0027** at the operator. `lower`
computes the int discriminant `d` (0/1/2) and tests it — no separate comparison
protocol per operator:

| operator | test on the discriminant `d` |
|----------|------------------------------|
| `a < b`  | `d == lt (0)` |
| `a > b`  | `d == gt (2)` |
| `a <= b` | `d != gt (2)` |
| `a >= b` | `d != lt (0)` |
| `a == b` | `d == eq (1)`  *(only when `Eq` is Ord-refined — see below)* |
| `a != b` | `d != eq (1)`  *(only when `Eq` is Ord-refined)* |

**Per-concrete-type discriminant.** `int`/`bool` never route here — they stay a single
inline `icmp` (`bool` orders `false < true`), byte-identical to pre-M16. `str` lowers to
a **heap-free lexicographic 3-way `load_byte` loop** (extending M15's `Eq` byte loop):
the first differing zero-extended byte decides, a proper prefix is less than its
extension, equal spans are equal — no witness call. A struct/enum resolves its `cmp`
witness, calls it into a fresh return slot, and reads `Ordering`'s tag with the proven
`get_tag` match-dispatch idiom.

**`Ord` refines `Eq`, recorded as exactly one `(Eq, T)` entry.** After coherence, a
serial pass appends **one** `(Eq, recv)` conformance for each `Ord` receiver that lacks
an existing `(Eq, recv)` — adding **no** `eq` method — so `==`/`!=` on an Ord-only type
route through `cmp` (`d == eq`). The precedence is:

1. an **explicit** `impl T has Eq` (authoritative — its `eq` method is dispatched);
2. else the **Ord-refinement** `eq` ≡ `cmp == Ordering.eq`;
3. else the structural `Eq` derive (M18).

The refinement writes only the conformance table (never the coherence seen-set) and
skips any receiver that already has `Eq`, so a genuine `impl T has Eq` **alongside**
`impl T has Ord` is **not** a T0020 overlap, still leaves exactly one `(Eq, T)` entry,
and lets M18's structural derive see the slot filled so it never double-fires.

**Total order only.** There is no float type, so every `Ord` is a total order; partial
orders are out of scope. **Known edge:** a user `enum Ordering` shadow wins the name
injection (user-first-wins), but the prelude `Ord::cmp` still returns the *prelude*
`Ordering`, so its variants would not match the user's — out of scope for M16 (no
example/fixture defines a conflicting `Ordering`).

## Structural `Ord` auto-derive (M19)

Structural `Ord` is synthesized on demand at a `<`/`>`/`<=`/`>=` use site — the same
source-less synthetic-codegen-unit channel M18 introduced for `Eq`, keyed on a non-AST
`(kind, layout, resolved-field-witnesses)` fingerprint. It fires iff every field/payload
conforms to `Ord` (recursively, via the shared ground `conforms` query) and no explicit
`impl T has Ord` exists; explicit impls win, and an **unused** derive emits **zero**
codegen. All six comparisons compose from the single derived `cmp -> Ordering`.

**Order.** A struct is compared **lexicographically in field DECLARATION order** (the
first field decides; a tie falls to the next). A payload enum is compared by
**discriminant (case-declaration) order first**, then — on an equal tag — that variant's
payload **lexicographically** (the SE-0266 total order). ⚠ **Behavior/semver hazard:**
reordering a struct's fields, or reordering an enum's cases, **silently changes** the
derived ordering. Nothing pins the order to names, so a refactor that permutes
declarations is a breaking change to every derived comparison.

**A derived `Ord` fills `Eq`.** Because `Ord` refines `Eq` (the single `(Eq, T)` entry,
above), a type used with `<` derives an `Ord` `cmp` that ALSO serves `==`/`!=` (via
`cmp == Ordering.eq`); the barrier registers only the `cmp` method and **suppresses** the
structural `Eq` recipe for that type, so a type used with both `<` and `==` synthesizes
**exactly one** unit (the Ord `cmp`) — no double-fire. The same suppression applies to a
nested field: if a field is used with `<` elsewhere (so it becomes an `Ord` type), an
enclosing struct's derived `Eq` calls that field's `cmp` rather than minting a second unit.

**Payload-enum `Eq` gap closed.** M18 conformed only *empty-payload* enums to `Eq`; M19's
`conforms` recurses every variant's payload fields, so an all-`Eq`-payload enum now derives
`Eq` too (and the derived/enum `Eq` emitter compares the discriminants then, per equal tag,
the variant's payload field-by-field). Consequently **every** ground value type conforms
to both `Eq` and `Ord`, and the "field blocks the derive" diagnostics (T0027 on a struct,
T0029) are unreachable for a ground program — reachable only via a unit operand or a
generic body whose type parameter's bound is not the protocol.

## Structural `Hash` auto-derive (M20)

```
protocol Hash { fn hash(self) -> int }         // 1-ary (self only); id 6, after Div = 5
```

Structural `Hash` is synthesized on demand at an explicit `.hash()` call — the same
source-less synthetic-codegen-unit channel M18/M19 use, keyed on the same non-AST
`(kind, layout, resolved-field-witnesses)` fingerprint. It fires iff every field/payload
conforms to `Hash` (recursively, via the shared ground `conforms` query) and no explicit
`impl T has Hash` exists; explicit impls win, and an **unused** derive emits **zero**
codegen. The builtin scalars conform natively: `int`/`bool` hash to their own value, `str`
to a heap-free byte polynomial, `unit` to a constant.

**Hash is INDEPENDENT of `Eq`/`Ord`** — it is not a refinement of either (unlike the
`Ord`→`Eq` fill). A type used with both `==` and `.hash()` synthesizes **two** units (an
`Eq$eq$…` and a `Hash$hash$…`); there is no shared-slot suppression.

**Fixed seed, deterministic, `Eq`-consistent.** The mixer is a **fixed-seed** multiply-
accumulate polynomial `h := h*MULT + fieldhash` (a struct folds its fields in layout order;
an empty-payload enum folds its discriminant; a payload enum folds the discriminant then the
active variant's payload). The seed is a compile-time constant — **never** randomized or
per-run — so a derived hash is reproducible run-to-run AND byte-identical across `-jN`. The
emitter walks the **same field/variant order the `Eq` derive uses**, so structurally-equal
values hash equal (`Eq`-consistency). ⚠ **Behavior/semver hazard:** reordering a struct's
fields, or reordering an enum's cases, **silently changes** the derived hash (exactly as it
changes the derived ordering) — a refactor that permutes declarations is a breaking change
to every derived hash.

**Op-set constraint (not a shortcut).** The roadmap names a "fixed-seed Wyhash", but the
current IR op set (`Ir.Op`) has no xor/shift/bitwise op, so a runtime Wyhash is not
expressible without touching the backend (which the roadmap forbids). The multiply-
accumulate polynomial is the forced substitute; it satisfies every *locked* constraint
(fixed seed, deterministic, reproducible run-to-run + `-jN`, `Eq`-consistent). As with
`Eq`/`Ord`, **every** ground value type conforms to `Hash`, so the "field blocks the derive"
diagnostic **T0030** is unreachable for a ground program (reachable only via a generic body
whose type parameter's bound is not `Hash`).

## Two-phase checking (and parallelism)

Checking is two phases over one frozen program model:

- **Pass A (serial, whole-program):** register + lay out every struct/enum,
  decode every fn signature into the global fn table, and check pub-signature
  coherence. This builds the immutable `Model` (fns/structs/struct_map/enums/
  enum_map). Because an omitted `-> T` is sugar for `-> ()` (no return
  inference), every `f.ret` is final after Pass A and no fn's signature depends
  on another fn's body.

- **Pass C (per-fn body check):** each fn's body is checked by a `BodyChecker`
  over the frozen `Model`. A BodyChecker holds only per-fn scratch (slot types,
  current return, loop stack, expected) and writes ONLY its own fn's node-type
  span and its own local diagnostics — so the per-fn checks are independent.
  In the whole-graph path with no dependency DAG, they fan out across the
  `-j N` worker pool; with a DAG (incremental builds) or single-file, they run
  serially. Either way the merge is one serial step after the join.

## Determinism

Diagnostics are emitted in source order (serially, this is discovery order).
After the per-fn merge they are **stably** sorted by `(module, byte_offset)`
— ties keep fn-id (source) order — so `-j1` and `-jN` produce byte-identical
diagnostics: same set, same order, same bytes. Ordering is by the sort key,
never by thread-arrival order.
