# Type system

The reference for `toyc`'s type checker (`src/types.zig`), run after name
resolution. Inference is **bidirectional and local**, with no HM-style unification:
a generic call binds its type parameters by one structural match against its
arguments (see [Generic calls](#generic-calls-inference-order)), and a lambda's
parameters take the function type expected where it is written (see
[Lambdas](#lambdas)). Outside a generic template every expression has a fully
concrete `Type` the moment it is checked.

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
| `func` | `fn(A, B) -> R` | check-time; reified to a `{code, env}` pair struct (see below) |
| `never` | `never`, as a return type only | the bottom type; a value that never exists |
| `invalid` | — (not spellable) | the poison/error type |

Two special types are not ordinary value types:

* **`never`** is **bottom**. A `never` expression (a break-less `loop`, a
  `panic(..)` call, a call to a `-> never` fn) never yields a value, so it
  fits wherever a value of a known type is wanted: a fn tail, a `return`, an
  `if` or `match` arm, a binding, an assignment, a call argument, an operator
  operand or a condition. A fn that ends in `panic("bad")` needs no dead
  `return`, and an arm may panic while the others yield the value. An operator
  judges only the sibling of a `never` operand (`1 + panic("x")` is `int`,
  `"a" - panic("x")` is still T0028); a unary operator over `never` is `never`.
  It is the *identity* of the `merge` join.

  `never` is spelled only as a return type: of a fn, method, protocol method,
  local fn, lambda, `extern` or fn type (`fn fail(m: str) -> never { panic(m) }`,
  `g: fn(str) -> never`). `panic` itself is seeded with that signature. The
  body of a `-> never` fn must diverge: one that may fall off its end is
  **T0043**, and a bare `return` or a value tail is **T0042**. A param, field,
  binding or type argument spelled `never` is **T0001**, since no value of it
  could ever be passed or stored. An `extern fn exit(code: int) -> never`
  declares a C fn that does not return.
* **`invalid`** is **poison** and is not spellable. The first time an
  operand is `invalid`, the containing expression silently becomes `invalid`
  too, so one root mistake produces exactly one diagnostic and never cascades.
  It *absorbs* in both `assignable` and `merge`.

A type-level name is never a value: a struct, an enum, a builtin type, an alias, a
protocol or a type parameter named where a value is wanted (`x := Color`,
`id(Color)`, `Color |> id`, `lib.Shape`) is **T0056**, and the hint for an enum names
one of its variants (`Color.red`).

## Integer widths: literals fit, operators wrap

An integer literal takes the width it is expected to have and must fit it (**T0034**).
A negated literal is checked as the negative value it denotes: `-128` fits an `int8`,
and `-1` fits no unsigned width (`v: uint8 = -1`, `id(-1)` with `fn id(x: uint8)` and
`v: uint = -1` are T0034); only `-0` does. A constant built from literals alone (`-1`,
`~1`, `1 + 2`, but not a shift) is a literal too: beside an operand of an integer type
it takes that width, as a bare literal does (`b + -1` and `b += 1 + 2` with `b: int8`
are `int8`), and its folded value must fit the width it takes (`c: int8 = 100 + 100` is
T0034). One given no width (`x := 9223372036854775807 + 1`) is an `int` whose folded
value is not checked. Operators on values never check at run time: they wrap at the
operand width, so two `int8` 100s add to -56, and unary `-` on an unsigned value is
`0 - u` (`u: uint8 = 1` then `-u` is 255).

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
(check). This formalizes the previously ad-hoc one-shot `expected`
parameter into this synth/check split; it adds no inference power.

A part of an expression — a struct literal's field value, a variant's payload, a
generic call's argument — is checked against its own declared type, never against the
expected type of the whole. A non-generic part has that type outright (`P { x: 5 }` with
`x: int8`, `S { tags: [] }`). A part whose type holds the parent's type parameters takes
them from the parent's expected type: a struct or variant with no type args matches it
against its own type (`b: Bag[int] = Bag { xs: [1, 2] }` checks `[1, 2]` as a
`Vec[int]`, `x: Option[int8] = Option.some(5)` makes `5` an `int8`), and a generic call
matches it against the callee's return type (`n: int = first([5, 6])`, with
`first[T](xs: Vec[T]) -> T`, checks the list as a `Vec[int]`; `x: int8 = id(5)` makes
`5` an `int8`). A part whose type still holds a parameter the expected type leaves open,
or whose parent has no expected type, is typed on its own.

A method call's expected type belongs to its result, not its receiver: it reaches a
receiver only as a hint, which a receiver that is itself a literal may take, along with
the receiver's parts (`x: int8 = Box { v: 1 }.get()`), and which never makes an error
inside the receiver (`fn f() -> Box[str] { return Box { v: 1 }.map(fn(v) { v.to_string() }) }`,
or `x: int8 = 300.small()` with `small` on `int`). The method is looked up on the type the
receiver ends with, so a receiver that takes the hint can lack it (below).

A value `if`, `match`, `loop` or labeled block checks every arm against its own expected
type: each branch of an `if`/`else if` chain, each match arm, and each `break` value, so
`if c { Option.some(1) } else { Option.none }` checks whichever branch comes first. A
hint stays a hint there: `(loop { break [1, 2] }).len()` is fine in an `int` slot. A
condition or a match guard is a `bool` whatever the arms yield, so it takes none, and a
match scrutinee takes the arms' type only as a hint (`match 300 { .. }` in an `int8` fn
matches an `int`).

Each arm decides alone whether it takes a hint, so arms can take it in different ways:
in `x: int8 = (if c { 300 } else { 1 }).small()` the `1` fits an `int8` and the `300`
does not. The hint is taken by the whole receiver or scrutinee or not at all: when
typing it under the hint gives an error, it is typed again with no hint, so both arms are
`int` here. Of the two typings the one with fewer errors stands, so an error of the
receiver's own (`1 + true` in one arm) is reported once, under the hint the other arms
needed. A receiver that takes the hint keeps it even when its
type then has no such method: with `small` on `int` only, `x: int8 = 2.small()` and
`x: int8 = (if c { 2 } else { 1 }).small()` are **T0018** on `int8`, while
`r := 2.small()`, with no expected type, calls it. An arm never takes its type from another
arm: `(if c { Option.some(1) } else { Option.none }).map(f)` has no expected type for
`Option.none` and is **T0016**, as it is with no call around it.

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
   else  emit T0058 "branches yield different types (A vs B)" ; invalid
```

`merge` stays **verbatim**; `assignable` does not replace it. They share the same
lattice discipline (invalid absorbs, never is the unit, `eql` otherwise) viewed
from two angles — assignability is directional (want ← got), join is symmetric.

## Statement and declaration rules

* `name := e` — infers `name : synth(e)`. `e` must produce a value; `()` is an
  error (nothing to bind).
* `name: T = e` — explicit typed local. Parse `T`, `check(e, T)`, bind `name : T`.
* `name = e` — assignment. `assignable(typeof(name), synth(e))`. The target is a
  local or a field / tuple-field chain rooted at one, or at an index element (see
  `xs[i].f = e` below). No other temporary is a place: a field of a call result
  (`p.get().x = v`) is **T0075**, and a call result itself (`p.get() = v`) **R0007**.
* `xs[i] = e` — element store on a `Vec[T]`. It means `xs.set(i, e)`: `xs`, then `i`,
  then `e` are evaluated, left to right, then the index is checked and the element
  stored, so an out-of-bounds `i` panics after `e` has run. `e` is checked against `T`
  (**T0041** names "an element of type T"). The handle is read before `e`, so an `e`
  that rebinds `xs` still writes into the `Vec` that `xs` named. A compound
  `xs[i] op= e` evaluates `xs` and `i` once, reads the element (an out-of-bounds `i`
  panics here, before `e` runs), then runs `e`, applies `op` and stores. A `Map`
  store `m[k] = v` is **T0057** with a hint to `m.set(k, v)`.
* `xs[i].f = e` — field store on an element, through any field / tuple-field chain
  (`ps[i].q.a = e`, `ps[i].0 = e`). It is read-modify-write: `xs`, then `i`, then `e`
  are evaluated, then the element is copied in (an out-of-bounds `i` panics here), the
  field is stored into the copy and the copy is written back with `xs.set(i, ..)`. No
  code runs between the copy-in and the write-back, so a write `e` makes to another
  field of the element is kept, and an `e` that grows `xs` still lands. A compound
  `xs[i].f op= e` evaluates `xs` and `i` once and reads the field (an out-of-bounds `i`
  panics here, before `e` runs), then runs `e`, applies `op` and stores the result as
  `xs[i].f = ..` does, copying the element in only after `e`.
* `place op= e` — compound assignment, for `op` in `+ - * / % | & ^ << >>` and the
  float forms `+. -. *. /.`. It means `place = place op e`: the place is read
  before `e` runs, `e` is typed like a binary operand (an integer constant such as `1` or
  `-1` takes the place's width), and the codes are the same as for that long form — **T0044** and
  **T0028** at the operator, the place codes of `=` at the place. A user type gets
  `+=` from its `Add` impl; there is no separate `AddAssign`.
* `return e` / bare `return` — `assignable(declared_ret, T)` where `T` is the
  value type (or `()` for a bare return).
* Every `return e` **and** the body's trailing expression are checked against the
  declared return type.

## Function signatures

* **Parameters must be user-typed** — `fn f(x: int)`. *(documented existing.)*
  The parser enforces this: `parseFnDecl` does `expect(.colon, ...)` then
  `parseType()` for every parameter (`src/parse.zig`), so a typeless parameter is
  a parse error, never reaching the checker. A lambda's parameter may leave its type
  out (see [Lambdas](#lambdas)); a local function's may not.
* **Return type — unit sugar.** Omitting `-> T` is **sugar for `-> ()`**. A declared
  function has no return-type inference: one with no arrow returns unit, and any
  non-unit return value must be annotated. A lambda's return type is inferred from
  its body (see [Lambdas](#lambdas)). *(`decodeFnSig` defaults a missing
  return-type node to `Type.unit`.)*
* **`mut` parameters** — `fn bump(mut n: int)` takes the caller's place by address,
  like a `mut self` receiver, and its writes reach the caller's variable. The
  parameter's type stays by-value for checking (`n: int`); the mode is the fn's own
  `mut_mask` (bit *i* for param *i*, a method's `self` is bit 0), computed in Pass A.
  The argument must be a mutable place — a local, a `Vec` element, or a field /
  tuple-field chain rooted at either — or it is **T0019**; the local it is rooted at
  counts as assigned for the box rule. An element (`bump(xs[i])`, `bump(ps[i].x)`, a
  `mut self` call `ps[i].bump()`, `for x in ps[i]` over an `iter(mut self)`) is
  read-modify-write: its handle and index are evaluated where the argument is, the
  element is copied in there, the callee writes the copy, and after the call each copy
  is written back with `xs.set(i, ..)`, in argument order (the receiver first). Two
  arguments rooted at one element are two copies, and the last write-back wins, even
  when they name different fields of it: `swap(ps[0].x, ps[0].y)` sets only `y`,
  since the copy written back last still holds the old `x`. A write the callee makes
  to the element through another handle is overwritten. Allowed on top-level fns and
  inherent-impl methods (associated fns included); a lambda's, local fn's, extern's or non-self protocol / conformance
  param, or a param past the 64th, is **P0006**. A fn with a `mut` param, or a method
  with one after `self`, is not a value (**T0049**).
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

## Generics syntax reservation

The generics front-end (`fn f[T, U](..)`, `struct Box[T]`, `enum E[T]`, a type
application `Box[int]`, explicit call type-args `f[int](..)`) landed append-only. Generic **functions**, **structs** and **enums** now have semantics
(see below).

The `[..]` bracket is disambiguated by **position**:

* **type position** — a trailing `[..]` on a type-ref is a **type application**
  (`x: Box[int]`, `mod.Box[int]`);
* **postfix-call position** — a `[..]` immediately after a name / qualified
  `mod.fn` callee is **explicit type-arguments** (`id[int](7)`); it wraps only that
  callee, and the following `(..)` forms the call.

**Value indexing `v[i]`** shares the bracket. The parser looks past the `]`: a `(`,
`{` or `.` after it makes a type application (only on an `identifier` or `field_access`
base), and any other token a value index (`xs[i]`). A type application whose base is a
value is read as an index by the checker, so `fs[i](x)` calls the element and
`xs[i].f` reads its field.

A third `[..]` position is **construction position** — a `[..]` on a name
followed by `{ .. }` (`Box[int]{ v: 1 }`) builds a `struct_init` whose lhs is the
`type_app` (no new `Node.Tag`, so `ParseHeader.version` is unchanged).

## Generic structs + the composite `App` type

Generic **functions** and generic **structs** are monomorphized to
concrete value types — there are no runtime dictionaries, no code sharing, and the
byte-foldable `Type` never widens. Only a generic **enum** decl remains gated with
T0013.

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
inference (`Box{ v: 1 }`) is deferred.

**Termination guard (T0017).** Generic structs are the first construct that makes
unbounded instantiation expressible: `fn go[T](x: T) { go[Box[T]](..) }` forms
type-args of strictly-growing generic-nesting depth. The serial mono worklist caps
that depth and rejects the program with **T0017 "instantiation too deep"** —
deterministically at `-j1`/`-jN`, never hanging or running out of memory — while a
legitimately deep-but-finite generic program still compiles.

## Generic calls: inference order

A generic call with no explicit type arguments (`id(7)`, `Box.with(3, true)`, a method
of a generic type `b.map(f)`) infers them by one left-to-right structural match of its
parameters against its arguments' types (`symbols/Infer.zig`); the first binding of a
variable wins and a different later one is **T0015**; one that disagrees with what the
receiver bound is an argument mismatch (**T0040**) instead, since the call infers none of
the receiver's variables. The arguments are typed in this order:

1. the receiver, which binds the impl's type parameters (`T` of `impl Box[T]`);
2. every argument that types itself: a method's argument against its parameter with
   the receiver's bindings in, a bare call's against its parameter through what the
   call's expected type binds against the return type, an associated call's with no
   expectation from the call (a parameter with no type parameter is the argument's
   type outright, as for any part). One that errs under that is typed again against
   its parameter with what the earlier arguments bind, and keeps that typing when it
   errs less: `pair(t, Option.none)`, `pair(t, .none)` and `count(t, [])` take `t`'s
   type, while an argument that types itself binds as before (`f(x8, 5)` with
   `fn f[T](a: T, b: T)` is still **T0015**);
3. each argument that waits for the others, left to right: a lambda with an
   unannotated parameter, or the `_` of the capture being checked. It is typed against
   its parameter with the bindings so far, and the arguments bind again after it, so
   a later lambda sees what an earlier lambda's return fixed;
4. a final match over all of them. A variable still unbound is **T0016**, unless an
   argument already reported an error, or an argument cannot fit its parameter: a
   lambda of the wrong shape binds nothing, so the argument itself is the error
   (**T0040**, `expected fn(int) -> U, got fn(int, int) -> int`; a waiting lambda with
   the wrong parameter count is **T0039**). A type parameter still open in such a
   message prints as its name.

A method of a generic type may declare type parameters of its own
(`fn map[U](self, f: fn(T) -> U) -> Box[U]`); they follow the impl's in its generic
run and bind from the call's arguments, and each instantiation is its own instance. They
may also be written, the impl's coming from the receiver as before: `b.map[str](f)`,
`b |> .map[str](f)`, `Box[int].with[bool](1, true)` on the type, and
`b.none[int]()` for a type parameter no argument names (`fn none[U](self) ->
Option[U]`), which inference cannot reach (**T0016**, whose message shows the explicit
form). The count must be the method's own (**T0054**), the impl's never written there
(`Box.with[bool](..)` is **T0016**: write `Box[int].with[bool](..)`). A method without
type parameters of its own takes protocol args instead, as on a concrete receiver:
`b.into[bool]()` picks the `impl Box[T] has Into[bool]` conformance among the
receiver's, and args no conformance has are **T0018**. A generic method is not a value
(**T0049**). A method of a non-generic type cannot declare its own type parameters yet
(**T0013** at the call): such a method would be a template on a concrete receiver, which
the method table, discovery, the seal and the LSP all treat as one row per method. A method's own type parameter may not reuse a name of its
impl's (`impl Box[T] { fn conv[T](..) }`), and no fn declares one name twice
(**T0053**). An impl's pattern, inherent or a conformance, names each type parameter
once (`impl Pair[A, B]`); one on a single instance (`impl Box[int]`,
`impl Box[int] has Into[int]`) or a partly concrete or repeated one
(`impl Pair[T, int]`, `impl Pair[T, T]`) is **T0013**, once per impl and whether or
not it has methods (`impl Box[int] has Marker {}`), and that is the impl's only
error: its methods are not checked against the protocol (no **T0024**), a call of one,
on a value or through the type, reports nothing more (no **T0018**), and a bound it
would have met is not reported unmet (no **T0023**). So is a conformance rejected for a
composite protocol argument (`impl P has Into[Option[int]]`).

## Associated functions

An `impl` fn without `self` is an associated function, called on its type: `K.make(1)`
on a struct or enum `K`, `lib.K.make(1)` through a module, `Box[int].new()` on a generic
struct (or `Box.new(5)`, inferred as above). The call selects it like a method and
passes no receiver. It is never called on a value (`k.make(1)` is **T0018**), an instance
method is never called on a type (`K.get()` is **T0018**), and on an enum a variant keeps
its name: `E.A(1)` constructs the variant `A` even when an associated fn `A` exists. An
associated fn of a generic enum is not callable yet (the call reads as a variant,
**T0061**), and an associated fn is not a value (`f := K.make` is **T0056**).

## Same-named methods

Two modules may declare a method of one name on one type: a program's own
`impl Vec[T] { fn each .. }` next to the one `std/vec` declares. Which one a call reaches
depends on where the call is written. A call in the program reaches the program's
method, so the pick never depends on the import order and a program keeps its own
method when std gains one of that name. Between two of the program's own modules, the
first in import order wins. A call written in a bundled module (`std/*`, `core/*`)
reaches the bundled method; std's bodies are immune to the program's methods, because
their bounds and results come from their own methods and the builtins. Only a bundled
module may import `core/*` (**R0013**).

A program's method may also take the name of a builtin on a builtin type (`str`'s
`len`, `byte_at`, `concat` and `to_string`, a scalar's `eq`, `hash` and `to_string`,
`into`/`try_into` on a number or `char`, a native `Option`/`Result` method). The program's
calls reach it, and a bundled module's calls keep the builtin. A program's other
methods stay reachable from std: they are how its types implement the protocols std's
generic bodies call (`next`, `display`, `cmp`), and a program's `to_string` or `hash` on
its own type is how that type implements them.

Shadowing a std method or a builtin is warning **W0011** at the program's method. For a
std method it fires whenever the std module is part of the program, imported directly
or through another module, because only then is there a std method to shadow; for a
builtin it always fires. Renaming the method keeps both.

## Function types

`fn(A, B) -> R` is a type like any other: a parameter, a return type, a struct field,
a variant payload, an alias target or a type argument (`Vec[fn(int) -> int]`). `-> R`
is optional (a unit return, as in a declaration) and right-associative:
`fn(int) -> fn(int) -> int` returns a function. Assignability is structural
equality, so an alias names the same type as its spelling.

A function type is `Kind.func` (appended, frozen ordinal): a fn entry in the same
composite table as `App`, holding `params ++ [ret]` and its pre-rendered label, which
diagnostics print (`cannot bind fn(int) -> bool to 'h' of type fn(int) -> int`).
Inference descends it like an `App` (`fn(T) -> U` against `fn(int) -> bool` binds
both). Reification maps every ground fn type to ONE 16-byte struct per grounded
signature, `{code: int @0, env: rawptr @8}`, whose `Layout.fn_sig` records the
signature; the pair is deduplicated on the grounded bytes, so `fn(Vec[T])` at
`T = int` and a written `fn(Vec[int])` are one struct. Past reify only a generic
template's signature keeps a `.func`. A function type conforms to `Display` only (it
renders `<fn>`), not to `Eq`/`Hash`/`Ord`; an aggregate holding one derives `Display`
but not `Eq` or `Hash`. A `Vec` of them gets a trace unit, because the `env` word is a
managed pointer.

## Function values

A named function is a value of its function type: `f := add`, `apply(add, 2, 3)`, a
struct field `Handler { run: double }`, a `Vec[fn(int) -> int]` element, or a return
value. A builtin (`panic`), an `extern` and a generic function have no single code
address and are **T0049** as values. Qualified `mod.f` works the same way.

Any expression of function type can be called: a local `f(1)`, an index `fs[i](x)`, a
call `get_op()(2)`, a pipe `x |> g`. The args are checked against the type's params, so
their expected types flow in as for a direct call. `recv.m(..)` selects a method first
and only then a fn-typed field named `m` (the value `recv.m` takes them in the other
order, see [Bound methods](#bound-methods); a method named like a fn-typed field is
**W0013**). A call of a non-function is **T0055** "called value is not a function", whatever
the callee's shape; a call of a module or an enum name (`io(5)`, so `5 |> io` too) is
T0055 as well. A function value is never generic, so `g[int](1)` is **T0054**, and when
the call has the wrong arity or argument types the message names the value as written
("'f' takes (int, int)", "'h.run' takes (int, str)").

An expression statement whose value is a function it never calls does nothing, so it
is warning **W0012**: a function name (`one`), a bound method (`c.bump`), a field, local,
tuple field or element of function type, a lambda, or a `_` capture. The message spells
the call from the source, `()` for a function of no parameters and `(..)` otherwise
("function value is never called; did you mean 'c.bump()'?"). Only a statement's value
counts: a value bound, assigned, passed, returned, piped, or that is the value of a
block or body that is used, is used. A call that returns a function runs, so its dropped
result does not warn; nor does any value under a return type that is already an error.

`==`/`!=` on two values of the same function type is identity (both words of the pair);
different function types are T0044. A function value's only method is `to_string`
(`"<fn>"`; any other is T0018), and the only protocol bound it meets is `Display`
(any other is T0023), so it renders but is no `Map` key. A generic
function with a function-typed parameter (`fn apply[T](f: fn(T) -> T, x: T)`) infers
through it and instantiates on the value's signature; the value is called through its
pair, unless the call is specialized (see [Specialized generic calls](#specialized-generic-calls)).

At runtime a named function value is `{its thunk, null}`: the thunk `<fn>$fnval(env,
params..)` drops the env and calls the function, so every value is called the same
way, with the env as the hidden first argument.

## Lambdas

`fn(x, y: T) -> R { .. }` in expression position is a lambda, a value of type
`fn(X, T) -> R`. `fn (` starts a lambda; `fn name` is a declaration (a local function
inside a block). It is checked bidirectionally, like a literal:

* A parameter's type is its annotation, else the matching parameter of the function
  type expected where the lambda is written: an annotated binding, an assignment, a
  parameter of a known function or method (a `Vec[fn(int) -> int]`'s `push` too), a
  struct field, a list element, a `return` or a trailing value. An expected function of
  another arity, or a part holding a type variable the call has yet to bind (a generic
  callee's `fn(T) -> T`), provides nothing; a parameter left with no type is **T0050**.
  A template's own type parameters are types like any other there: in
  `fn g[T has Addable](t: T)`, `fn(o) { t.add(o) }` needs no annotation where a
  `fn(T) -> T` is expected. In a generic call a
  lambda is checked after the call's other arguments and its receiver, against what
  they bind (see [Generic calls](#generic-calls-inference-order)), so
  `apply(fn(x) { x * 7 }, 1)` and `b.map(fn(x) { x + 1 })` need no annotation; a type
  parameter only the lambda's return fixes is inferred from it. When nothing else
  fixes a parameter, an expected function type on the call still types the lambda
  (`g: fn(int) -> int = id(fn(x) { x + 1 })`); otherwise it is T0050.
* The return type is the annotation, else the expected function's, else the join of
  the body's trailing value and its `return`s (`()` when there is none). A declared or
  expected non-unit return must be produced on every path (T0043); a trailing value or
  `return` of another type is **T0042** at that value. A lambda held to an expected
  return type declared nothing, so that message spells the type the lambda really has
  ("lambda has type fn(int) -> int, but fn(int) -> bool is expected"). An annotated
  parameter the expected type does not have means the lambda does not fit it, so its
  return type comes from its body and the call's mismatch spells the whole type.
* `return` and `?` exit the lambda: they check against the lambda's return type, and
  `?` needs one that is annotated or expected (T0032 otherwise). A `break`/`continue`
  cannot target a loop or labeled block outside the lambda (**T0051**).
* A lambda may name the locals and parameters of the function around it: it captures
  them by reference. A captured local that nothing assigns is copied when the lambda's
  value is made (a copy and a share cannot be told apart). One that is also assigned
  anywhere in the function (`x = ..`, `x.f = ..`, a `mut self` call on `x`, or `x` as a
  `mut` argument, inside a
  lambda or outside) is **boxed**: it lives in a GC cell for its whole lifetime, and
  every use, in the function and in its lambdas, goes through the cell. Parameters and
  loop variables follow the same rule, and a boxed loop variable gets a fresh cell per
  iteration. A `mut` parameter (`mut self` included) may be captured only by a closure
  that does not escape (closures.md, Escape analysis); one that escapes is **T0052**. A
  local passed as a `mut` argument counts as assigned. A captured
  `Vec`/`Map` is its handle, so pushes through it are shared either way. With the opt
  passes on, a boxed local of 16 B or less that only closures that do not escape capture
  stays in its slot, and they hold its address (closures.md, Escape analysis).

Each lambda is its own function `<owner>$l<rank>(env, params..)`, where the owner is
the enclosing function or generic instance and the rank is the lambda's pre-order
index in it; a lambda in a generic function has one per instance. Its value is
`{its address, env}`, called like any function value. A lambda with captures gets a
fresh GC cell as its env each time its value is made, holding the copied captures and
the cell pointers of the boxed ones (ascending slot, each at the next multiple of its
alignment, a pointer 8 bytes, the cell rounded to 8 bytes); the unit copies them into
locals on entry, and a nested lambda copies them on from there, so every closure over a
boxed local shares its one cell. A lambda that captures nothing (or only zero-sized
values) has a null env, so two evaluations of it compare equal; two evaluations of a
capturing lambda do not.
Env and box cells are word-scanned by the collector, so captured handles, strings,
boxes and other closures stay alive with them; a function holds each of its boxes in a
`rawptr` slot, a precise root.

A `_` that is a whole call argument makes the call a lambda of one parameter, a
**capture**: `add(1, _)` is `fn(v) { add(1, v) }`, and it is checked and lowered as that
lambda. Its parameter type is the expected function type's when that gives one, else the
type the call wants at the `_` (a known function's, method's or constructor's parameter),
so `inc := add(1, _)` is `fn(int) -> int` with no annotation. In a generic call that is
the parameter with what the call's other arguments bind, so `pair(1, _)` is
`fn(int) -> int`; `id(_)`, where nothing else binds `T`, is the callee's **T0016**. With
no type at all the `_` is **T0050**, reported only when nothing else in the call is an
error. A capture passed to a bare generic call is typed by its own call, so
`twice(add(1, _), 5)` infers `T = int`. A `_` at a `mut` position is
**T0019**, and passing a capture where a non-function is wanted (`neg(add(1, _))`) is
**T0040** with a hint. `x |> f(a, _)` is the plain call `f(a, x)`. See `closures.md`.

## Local functions

`fn name(p: P, ..) -> R { .. }` as a statement inside a block declares a local
function: a local `name` of type `fn(P, ..) -> R` (a missing `-> R` is `()`, as for a
top-level function; every parameter is annotated). The local is bound before the body
is checked, so the body can call itself; a later local function is not yet in scope,
so two local functions cannot call each other (R0001). It is checked like an annotated
lambda: `return` and `?` exit it (T0043 if a non-unit one can fall off the end), no
jump leaves it (T0051), and it captures the locals around it by the same copy-or-box
rule. It cannot be generic (**P0003**) or assigned (**R0007**), and one never named
after its declaration is **W0003**.

It lowers like a lambda: its unit is `<owner>$l<rank>`, and its declaration makes the
value `{code, env}` straight into the name's slot. A reference to itself inside its
body is not a capture: the unit rebuilds the pair from its own code and the env it was
called with, which is the value its caller holds, so recursion costs no allocation and
the value compares equal to itself inside and out.

## Bound methods

`recv.m` in value position, where `m` is a method and no field of `recv`, is a bound
method: a closure that captures the receiver and calls `recv.m(..)`. It has the
method's function type without `self` (`acc.plus` for `fn plus(self, x: int) -> int`
is `fn(int) -> int`), so it is passed, stored and called like any function value. A
field of that name wins, and an assignment target never binds (`acc.plus = f` is
T0046, no field). Methods of generic-type instances bind too (`b.get` on a `Box[int]`,
`xs.len` on a `Vec[int]`), through the instance's own method.

So a method and a field of function type with one name reach two functions: with
`struct H { run: fn(int) -> int }` and `impl H { fn run(self, x: int) -> int }`,
`h.run(1)` calls the method and `f := h.run; f(1)` calls the field. Each rule keeps a
use working when the other member appears (a call when a field is added, a field read
when a method is added), so neither changes; instead the method declaration is warning
**W0013** whenever its struct has a field of that name whose type is a function type
(an alias of one included): "method 'run' on 'H' has the same name as its field 'run'
of fn type: 'h.run(..)' calls the method and 'h.run' reads the field; rename one". It
covers inherent and conformance methods and the methods of a generic struct (a field
`f: fn(T) -> T` of `Box[T]`). A field of another type cannot be called, so it does not
warn, even a type parameter `v: T` that an instance grounds to a function type, and
neither does an associated fn, on which `h.run(1)` is already T0018. That error does not
call the field either; its message names both ways in: "'run' is both an associated
function of 'H' and a field of fn type: 'H.run(..)' calls the function and 'h.run' reads
the field".

The receiver is captured like any other variable. A bare local receiver that is
assigned anywhere in the function (including by a `mut self` call) is boxed, and the
value holds its cell, so the value and the function see each other's writes; one that
nothing assigns is copied into the value. Any other receiver (a field `p.inner.m`, a
call result) is evaluated once, when the value is made, and copied. A `mut self`
method writes its receiver through the value, so binding one boxes its receiver, which
must be a local variable: a temporary or a `Vec` element is **T0019**, a field
**T0072**. A method bound on a `mut` parameter (`mut self` included) holds the
caller's variable by address; the bound value must not escape (**T0052**). A
method with a `mut` parameter after `self` cannot be bound (**T0049**). An
associated function has no receiver to bind (**T0018**), a generic method has no single
instance (**T0049**), and a method the compiler expands at each call (a scalar or
`str` builtin such as `s.len`, a conversion, the derivable `hash`/`to_string`, a native
`Option`/`Result` method, a protocol method of a type parameter) is **T0049**; wrap
the call in a lambda instead.

At runtime a bound method is `{its adapter, env}`. The env is a fresh GC cell each time
the value is made, so two evaluations compare unequal: `<witness>$bound(env,
params..)` passes the receiver copy in the env as `self`, and `<witness>$boundbox`
reads the receiver through the boxed local's cell (for `mut self`, it passes the cell
itself as the receiver's address).

## Specialized generic calls

A generic call whose argument for a function-typed parameter is a named function
(`apply(inc, 1)`) or a lambda literal (`apply(fn(x: int) { x + k }, 1)`) goes to a copy
of its instance, `<instance>$sp<p>$<target>` for each such argument in parameter order.
A call of a generic type's method (`b.apply(inc)` on a `Box[int]`) is a generic call of
its instance too; `p` counts `self`, so its first argument is parameter 1.
In the copy a call of that parameter is a direct call of the target: a named function
without an env, a lambda's unit with the env of the pair the parameter holds. The copy
keeps the instance's signature, so the caller passes the same arguments, and a use of
the parameter as a value (stored, passed on, compared) still sees the pair. Any other
argument (a local holding a function, a local function, a field, a call result, a
bound method) goes through the erased instance, which calls the pair indirectly; so
does the parameter when the copy passes it on (`repeat(f, ..)` inside `repeat`).

A parameter is specialized only when its function never assigns it, and a generic
function that contains a lambda or a local function specializes none. Each generic
function or method gets at most 8 copies over all its instances: the first 8 by name, so which
call sites keep theirs does not depend on the order the compiler visits them. Past the
cap a call uses the erased instance. `toy build --no-specialize` sends every call
through the erased instance; it changes the code, never the behavior.

## Protocols, the prelude, and builtin-scalar conformance

`has` is the single conformance relation: `impl T has P { .. }` conforms `T` to a
declared `protocol P`. Coherence is checked **whole-program, serially** (before the
parallel body pass): exactly one impl per `(protocol, type)`, no orphan rules — a
duplicate (even in a sibling module) is **T0020**, a missing method or undeclared
protocol is **T0021**. When one of the two impls is in bundled std (std/float's
`impl float has Display`, say) the T0020 sits on the program's impl and names std's
module, and the program's impl methods are dropped so its calls give no further error. A protocol stores its method NAMES only; the per-impl
signature-compatibility check (params/return vs the protocol) is **deferred**
(the builtin `Eq` below is signature-correct by construction).

**Receiver mode.** A conformance method takes its receiver as its protocol method
declares it, `self` or `mut self`; otherwise **T0076** at the method's receiver. The two
pass the receiver differently (a copy, or the caller's address), and a desugar that
calls a witness (`==`/`!=`, `<`, `+`, `Display`, a `Map` key's `Hash`, `for`'s `next`)
has only the protocol to go by. Every prelude protocol declares `self`; `std/iter`'s
`Iterator.next` declares `mut self`.

**Prelude.** Protocol/type names that must be universally in scope with no
import (`Eq`, later `Ord`/`Option`/...) are delivered by **native compiler
registration**, chosen over an embedded `.toy` module so there is **no new
module-graph or content-fingerprint surface**. `registerPrelude` runs at the head of Phase 0c (serial), giving the `Eq`
protocol global id 0; a bare `Eq` that names no module protocol falls back to that id
in `protocolIdFromNode` (a user protocol of the same name shadows it via first-lookup).

**Multi-space conformance key.** The coherence key spans `(protocol, recv.kind, recv_id)`
— builtin scalar Kinds (int/bool/str/unit — which carry no `struct_id`/`enum_id`) as well
as struct/enum nominals (`recv_id` is 0 for scalars, which the kind byte distinguishes).
(extended with the protocol type-args; see the generic-protocols section.)
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

## Generic protocols + multi-conformance

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
**structurally** (ordered, never XOR) into the `(e)` fingerprint component — so an
`Into[int]` → `Into[bool]` edit flips exactly the dependent monomorphizations' keys
(no stale-witness miscompile). Protocol-args are restricted to already-concrete non-`App`
value types (scalar/struct/enum); a composite `App` arg's check-time index is
run-order-dependent, so it is rejected (would break key + fp determinism).

**Generic conformances overlap when their args unify (T0020).** An impl on a generic
receiver (`impl Box[T] has Conv[..]`) is a pattern over the receiver's own params, so
two such impls of one protocol on one ctor share one variable space. Coherence unifies
their protocol-arg patterns (two-sided, with an occurs check) and reports T0020 at the
second impl when one substitution makes them equal: `Conv[T]` and `Conv[int]` are both
`Conv[int]` on a `Box[int]`, where no call could choose. `Conv[int]` with `Conv[bool]`,
or `Conv[T]` with `Conv[Option[T]]`, never meet and both stand. Bounds on the params
do not count: an overlap is rejected even when a bound would rule the shared instance out.

**Ordinal scheme: `Self = type_var(0)`, protocol params = `type_var(1..)`.** When a
protocol method sig is decoded, the synthetic `self` slot is forced to `type_var(0)` and
the protocol's generic params occupy ordinals `1..` (a `""` placeholder reserves index 0
for `Self`, since `""` never matches a source identifier). So `fn into(self) -> U`
decodes to `[type_var(0)] -> type_var(1)`, and `U` never aliases the receiver.
`groundProtoType(ctx, ty, recv, args)` grounds `tv(0)→recv` and `tv(k≥1)→args[k-1]`,
through `App` args and fn types as well as at the top level (`Option[Item]` grounds to
`Option[int]`) — the one grounding used by the T0024 coherence signature check and the
bound-as-axiom body dispatch (each param and the return of `x.m(..)` on a bounded `T`).

**Multi-conformance requires explicit args (T0025).** At a use site `p.into()` where the
receiver conforms to one generic protocol **multiple times** and no protocol type-args
disambiguate, the compiler does **not** pick arbitrarily — it is a use-site error
**T0025** naming the conflicting conformances in source (fn-id) order (deterministic at
any `-jN`). An explicit `p.into[int]()` (parsed as `call(type_app(field_access, [int]))`)
selects the matching conformance; on a generic receiver (`impl Box[T] has Into[int]`
and `has Into[bool]`) `b.into[bool]()` selects it the same way, and an inferred
`b.into()` there is T0025 as well. The generic receiver's lookup, `pickGenericMethod`,
applies `pickMethodIndex`'s rule to the method templates of the receiver's type
constructor, and every place the checker selects a generic receiver's method by name
uses it: a call, a bound method value (`f := b.into`), the `for`-in desugar's `iter`.
The same holds for two different protocols that share a method name (`Show` and
`Debug`, both with `render`). The expected type does not choose a conformance:
`n: int = p.into()` is T0025 too. A type conforming **once** (a non-generic protocol
OR a single generic conformance) resolves with **no** explicit args — byte-identical dispatch
to the single-conformance path. The single disambiguator `resolveConformanceMethod` is shared by all three
consumers (BodyChecker types, `lower` symbols, `AstWalk` fingerprint), so they always
select the identical witness. A `[T has Convert[U]]` bound resolves at the mono worklist
by substituting the bound's args through the instance type-args before `findConformance`.

**A call on a bounded param takes the bound's conformance.** In `fn use_show[T has
Show](t: T) -> int { t.render() }`, `t.render()` can only mean `Show.render`, so each
instance calls the `Show` witness even when its type also has `Debug.render`: the
bound-as-axiom check records the protocol it resolved the call through, and the
instance's re-check keeps to that protocol, with the bound's args grounded through the
instance (`[T has Into[int]]` takes `Into[int]` among `Into[int]` and `Into[bool]`).
On a generic type (`impl Box[T] has Show` beside `impl Box[T] has Debug`) the re-check
picks the method template of the bound's conformance the same way.

**A generic conformance meets a bound.** `impl Box[T] has Show` makes every `Box[X]`
conform to `Show`, so `use_show(Box { v: 1 })` checks. The impl may bound its own
params, `impl Vec[T has Display] has Display`: `Vec[X]` then conforms only where `X`
does, by any route a bound accepts (an impl, a generic impl, or a structural derive).
`conform.generic` decides it: the row of the receiver's type constructor binds the impl's
params from the receiver's args (`Infer.match` over the impl's receiver pattern), its
protocol-arg pattern must ground to the bound's args, and each impl-param bound must
hold. Where a row covers the receiver but a bound fails, the type does not conform and
the structural derive does not stand in: an explicit impl wins even where it does not
apply, so the **T0023** names the type arg that fails it. The `[T has P]` check at the
mono worklist, the operator and `to_string` gates, and the structural walk (a field of
type `Box[int]`) all ask it. Its witness is an instance of the impl's method: a
`to_string`, `==` or derived field that leans on it records a derive request as a
structural derive does, the mono tail instantiates the impl's methods for it, and each
reached instance gets a concrete conformance row on the reified type, so the derive
synthesis reuses it rather than deriving one.

**The T0025 hint is written from the case at hand**, and following it compiles. Each
conformance whose protocol args no other conflicting conformance shares is named by
them: `.into[int]() or .into[bool]()`. No call syntax names a protocol, so any other
(a protocol without args, or args another one shares) is reached through a fn bounded by
it, which the hint spells out: `fn render_via[T has Show](x: T) -> int { x.render() }`.
A method value takes no protocol args (`f := p.into[int]` is T0025
too), so at one the hint puts them in a lambda's call: `fn() { p.into[int]() }`. In an
unbounded template, which only its instances check, the instances' reports at one site
that differ only by the template's type args become one, written with the template's own
params: `.into[T]()`, or for **T0016** `pick[T, int](..)`.

## Operators: comparison via `Ord`

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
inline `icmp` (`bool` orders `false < true`), byte-identical to the prior behavior. `str` lowers to
a **heap-free lexicographic 3-way `load_byte` loop** (extending the `Eq` byte loop):
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
3. else the structural `Eq` derive.

The refinement writes only the conformance table (never the coherence seen-set) and
skips any receiver that already has `Eq`, so a genuine `impl T has Eq` **alongside**
`impl T has Ord` is **not** a T0020 overlap, still leaves exactly one `(Eq, T)` entry,
and lets the structural derive see the slot filled so it never double-fires.

**Total order only.** `float` conforms to `Eq` but not to `Ord`, so every `Ord` is a total
order; partial orders are out of scope. **Known edge:** a user `enum Ordering` shadow wins the name
injection (user-first-wins), but the prelude `Ord::cmp` still returns the *prelude*
`Ordering`, so its variants would not match the user's — out of scope here (no
example/fixture defines a conflicting `Ordering`).

## Structural `Ord` auto-derive

Structural `Ord` is synthesized on demand at a `<`/`>`/`<=`/`>=` use site — the same
source-less synthetic-codegen-unit channel introduced for `Eq`, keyed on a non-AST
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

**Payload-enum `Eq` gap closed.** The earlier `Eq` derive conformed only *empty-payload* enums; this derive.s
`conforms` recurses every variant's payload fields, so an all-`Eq`-payload enum now derives
`Eq` too (and the derived/enum `Eq` emitter compares the discriminants then, per equal tag,
the variant's payload field-by-field). Consequently an aggregate derives `Eq` and `Ord`
exactly when its fields conform: a `float` field, which is `Eq` but not `Ord`, blocks the
`Ord` derive, and `<` on that type is T0027.

## Structural `Hash` auto-derive

```
protocol Hash { fn hash(self) -> int }         // 1-ary (self only); id 6, after Div = 5
```

Structural `Hash` is synthesized on demand at an explicit `.hash()` call — the same
source-less synthetic-codegen-unit channel the `Eq`/`Ord` derives use, keyed on the same non-AST
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

**The mixer is a polynomial, not a Wyhash.** It satisfies every locked constraint (fixed
seed, deterministic, reproducible run-to-run and across `-jN`, `Eq`-consistent). A field
that does not conform to `Hash`, a `float` or a function value, blocks the derive:
**T0030** names the field.

## `Display`, `to_string` and printing

```
protocol Display { fn display(self) -> str }    // 1-ary (self only); id 7, after Hash = 6
```

**`Display` returns a `str`.** A value's rendering is built once, into a fresh gc `str`,
and returned. Everything that shows a value goes through that one rendering:

- `x.to_string()` is a universal method on every `Display` value. It returns `display`'s
  result; on a `str` it returns the receiver itself, so a `str` never allocates.
- `io.print(x)` and `io.println(x)` (`std/io`) are **ordinary generics**,
  `fn print[T has Display](x: T)`, that write `x.to_string()` through `core/sys`. They
  get no checker or lowering special case: a call is a `[T has Display]` bound check and
  a monomorphized instance like any generic, and its fingerprint follows the instance.
  There is no global `print`; a program imports `std/io`.
- A custom `impl T has Display` provides `fn display(self) -> str` (a unit-returning body
  is T0024) and wins over the structural derive.

**Builtin renderings.** One lowering authority, `displayToSlot` (`lower.zig`), renders a
scalar and calls the witness of a struct or enum, so a field never renders differently
from the same value on its own:

- `int` and every width → decimal digits through the runtime builtin `__int_to_str`, with a
  leading `-` when negative (`i64::MIN` included); a `uint`/`uint64` → `__uint_to_str`,
  never signed, so 2^63 and above print as their magnitude.
- `bool` → the literal `true` or `false`; `()` → the literal `()`.
- `str` → its raw bytes, never quoted.
- a function value (a named fn, a lambda, a closure or a bound method) → the literal
  `<fn>`. A function type conforms to `Display` and to no other protocol; no impl can
  name it, so it has no witness and the derive passes a fn field straight to this arm.
  `f.to_string()` is `"<fn>"`.
- `char` → its UTF-8 encoding (the prelude `char` struct overrides the structural walk).
- `float` → no builtin rendering: std/float's `impl float has Display` is its witness,
  called like a struct's, so a float prints only when std/float is in the program
  (std/io imports it). See `std.md` for the format.

**Structural derive.** `Display` for a struct or enum is synthesized on demand at a
`to_string` use (which `io.print` makes), through the same source-less
synthetic-codegen-unit channel as `Eq`/`Ord`/`Hash`, keyed on the same non-AST
`(kind, layout, resolved-field-witnesses)` fingerprint. It fires iff every field and
payload conforms to `Display` (recursively), none is a managed box, and no explicit impl
exists; an unused derive emits zero codegen. The derived unit returns a `str`: it
appends each fixed run (name and separators, one interned literal) and each field's
rendering into one accumulator `str`.
A scalar field renders through `displayToSlot`; an aggregate field calls its sibling
`display` witness.

**Format** (user-observable; renaming a type or field, or reordering fields, changes it):

| shape | spelling |
|-------|----------|
| struct | `Name{field: <v>, field2: <v2>}`: no space after `Name`, `: ` after each field name, `, ` between fields, no trailing space |
| tuple struct | `Name(<v0>, <v1>)` |
| generic struct instance | its template's bare name: `Pair[int]` renders `Pair{a: 1, b: 2}` (no type arguments, never the mangled `Pair$int`) |
| enum unit variant | `variant` |
| enum payload variant | `variant(<v0>, <v1>)`: payload fields in declaration order |
| nested `str` | its raw bytes, unquoted |

**std renderings.** The boxes print through explicit generic impls in std, each bounded
by `Display` on its params, so an aggregate holding one derives through them:

| type | spelling |
|------|----------|
| `Ref[T]` (std/ref) | `Ref(<v>)`: the value in the cell, never its address |
| `Vec[T]` (std/vec) | `[<v0>, <v1>]`; empty `[]` |
| `Map[K, V]` (std/map) | `{<k0>: <v0>, <k1>: <v1>}` in insertion order; empty `{}` |
| `Set[T]` (std/set) | `{<v0>, <v1>}` in insertion order; empty `{}` |

Each element renders through its own `display`, so `[[1], []]` and `Ref([1, 2])` show
contents. A render walks handles without a guard: a cycle through them (a `Vec` that
holds itself through a `Ref`) does not terminate. A collection's render is written once,
into a buffer of its final length (`core/text.join`).

**What cannot print.** A type that does not conform to `Display` is a compile error,
never a fallback rendering (no addresses):

| type | today |
|------|-------|
| `float` without std/float | its `Display` is an impl in std/float: a `[T has Display]` bound is **T0023** naming std/float, `(1.5).to_string()` is T0018 |
| a struct or enum holding a `float` field without std/float | the derive is blocked: **T0023** at an `io.print`, **T0031** at a `to_string` naming the field |
| `rawptr` | no `Display`; it exists only inside core |
| a generic `T` without a `Display` bound | checked per instance: `fn f[T has Eq](x: T) { io.print(x) }` accepts `f(2)` and `f(1.5)` (std/io brings std/float); an instance whose type does not conform is **T0023** at the call |
| `Vec[X]`, `Map[K, V]`, `Set[X]` or `Ref[X]` whose type argument does not conform | std's impl bounds its params (`impl Vec[T has Display]`): **T0023** names the type argument that fails it |
| a managed box with no impl (the private `gc_array`), and any aggregate holding one | a box never derives `Display` (its one field is a cell address); only an explicit impl makes it print. **T0023** at an `io.print`, whose hint names the box; T0031 at a struct's `to_string` naming the field. `Eq` (cell identity) and `Hash` (no box keys) keep their own rules |

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
