# Closures, function values and places

Status: **spec, not implemented**. Decisions were locked in a /grill-me pass on 2026-09-29. The roadmap is `ROADMAP.md`.

## Summary

Functions become values. A lambda `fn(x) { x + 1 }` can capture the variables around it, can be stored and passed, and can be called later. Variables are **places**: the compiler decides where a variable lives. User code never writes `&` or `*` to share or mutate state.

```toy
import std/vec

fn counter() -> fn() -> int {
    n := 0
    fn() {                         # captures n by reference: n moves to a GC box
        n = n + 1
        n
    }
}

fn bump(mut x: int) { x = x + 1 }  # changes the caller's variable, no heap

fn main() -> int {
    next := counter()
    next()
    total := 0
    [1, 2, 3].each(fn(v) { total = total + v })
    bump(total)                    # total is now 7
    return next() + total          # 2 + 7 = 9
}
```

## Function types

- **Spelling:** `fn(A, B) -> R`. `fn()` takes no parameters and returns unit. `-> R` is optional, the same as in declarations.
- **One type for both:** a named function and a lambda with the same signature have the same type.
  - `f := add`, `apply(add)` and `apply(fn(a, b) { a + b })` all type-check.
- **Where they are allowed:** a function type is an ordinary type. It can be a parameter, a return type, a struct field, a type argument (`Vec[fn(int) -> int]`) or a local.
- **Equality:** `==` is identity: it compares the code pointer and the environment pointer.
  - A named function equals itself.
  - Two evaluations of the same lambda are equal only if it captures nothing: its environment is then null. A capturing lambda gets a new environment each time, so its evaluations are not equal.
- **Protocols:** function values do not conform to `Hash` or `Display` (see issue #22).

## Lambdas

```toy
fn(x) { x + 1 }                    # param type from the expected fn type
fn(x: int, y: int) -> int { x * y } # fully annotated
fn() { io.println("hi") }
```

- **Body:** a block. Its trailing expression is the value, the same as a function body.
- **Parameter types:**
  - They come from the expected function type, through bidirectional checking, the same way literals get their types.
  - In a generic call a lambda is checked after the call's other arguments and its receiver: `apply(fn(x) { x * 7 }, 1)` and `b.map(fn(x) { x + 1 })` need no annotation. A type parameter that only the lambda's return fixes is inferred from it.
  - When nothing else fixes a parameter, an expected function type on the call still does (`g: fn(int) -> int = id(fn(x) { x + 1 })`). Otherwise the parameter needs a type annotation. A missing one is a coded error (T0050).
- **Return type:** inferred from the body.
- **Early exit:** `return` and `?` exit the **lambda**, not the enclosing function. `?` requires the lambda's own return type to be `Option` or `Result`.
- **Loops:** a lambda cannot `break` or `continue` a loop outside it, labeled or not. That is a coded error. Loops inside the lambda work as usual.
- **Parsing:** in expression position, `fn(` starts a lambda, and `fn name` is a declaration.

## Local named functions

A block can declare a named function:

```toy
fn main() -> int {
    base := 10
    fn fact(n: int) -> int { if n <= 1 { base } else { n * fact(n - 1) } }
    return fact(3)
}
```

- It can call itself, and it captures like a lambda.
- **Recovery:** inside a block, `fn name(` is a nested declaration, so it is no longer a place where a broken block stops.
  - `pub`, `struct`, `enum`, `impl` and `import` still stop a broken block.
  - To make up for this, a "missing `}`" hint is emitted when EOF is reached inside a function that contains nested functions.

## Capture: by reference, implicit

- **Reference, not copy:** a lambda refers to the **variables** it names, not to copies. Assignments inside and outside the lambda see each other.
- **What gets boxed:** the compiler moves a local into a GC box **only** if it is both captured by a closure and assigned anywhere.
  - "Assigned" includes `x = ..`, field stores `x.f = ..`, and `mut self` method calls on `x`.
  - Parameters and loop variables follow the same rule, except a `mut` parameter: closures hold its address and it is never boxed.
  - Every use of a boxed local, in the outer function too, goes through the box.
  - With the opt passes on (`-O1`), a local of 16 B or less that only closures that do not escape capture is not boxed: they hold its address (see Escape analysis).
- **Read-only captures** are copied into the environment. With no assignments, a copy and a share are indistinguishable.
- **Values keep today's rules:** `y := x` copies a struct and shares a `Vec` / `Map` / `Ref` handle.
- **Loop variables:** each iteration has a fresh binding. `for i in 0..3 { fs.push(fn() { i }) }` gives closures that return 0, 1 and 2.

## Representation

- **A closure value is a 16-byte pair `{fn, env}`:**
  - `fn` is the code pointer.
  - `env` points to a GC cell, which holds the copied captures and pointers to the boxed captures. The collector scans the cell word by word: it has no type descriptor yet.
  - `env` is null for a named function and for a lambda that captures nothing, so a named function as a value costs no allocation.
  - At `-O1`, a closure that does not escape keeps its env in a slot of the frame that makes it (see Escape analysis).
- **Calling a function value** is a `call_indirect`, with `env` as a hidden first argument.

## Generics

`fn map[T, U](xs: Vec[T], f: fn(T) -> U) -> Vec[U]` is instantiated in a hybrid way:

- **Known at compile time:** when the argument is a lambda literal or a named function, the call site gets its own instance. The call is then direct and can be inlined.
- **A function value:** a variable, a field or a return value goes through an erased instance, keyed only on `T, U`, with the fat pair and `call_indirect`.
- **A cap per generic function** on specialized instances limits code size. Past the cap, calls use the erased instance.
- **Methods with their own type parameters:** a method of a generic type can declare its own (`impl Box[T] { fn map[U](self, f: fn(T) -> U) -> Box[U] }`), and so can an associated fn (`fn with[U](v: T, tag: U)`). They are inferred from the arguments, never written (`b.map[int](..)` stays T0018), and each instantiation is its own instance. On a non-generic type they are T0013 for now. A method's own parameter may not reuse a name of its impl's (`fn conv[T]` in `impl Box[T]` is T0053). A lambda of the wrong shape is reported as itself, T0039 for a wrong parameter count or T0040 for a wrong type (`expected fn(int) -> U, got fn(int, int) -> int`), not as a parameter nothing could infer.

## `mut` parameters

```toy
fn bump(mut n: int) { n = n + 1 }
bump(count)                          # count changes
```

- **Mechanism:** a `mut` parameter refers to the caller's variable by address, with no heap. This is the same mechanism as `mut self`.
- **The argument must be a mutable place:** a local or a field of one. The error is T0019.
- **No marker at the call site.** The LSP shows "mutated by bump".
- **Escaping closures:** a `mut` parameter captured by a closure that escapes is T0052 ("copy it to a local first"), because the address would outlive the caller's frame. A closure that does not escape may capture it, at every opt level: it holds the caller's address, as the parameter does.
- **Limits:**
  - **Where `mut` may appear.** On the params of top-level fns and of inherent-impl methods, static (associated) methods included, and on any `self` receiver where `mut self` is allowed today. It is P0006 on a lambda's params, a local fn's params, an `extern` fn's params, a protocol sig's or a conformance method's params after `self`, and any param past the 64th.
  - **The argument.** A local (a param, a forwarded `mut` param, a loop variable and a match binding included) or a field / tuple-field chain rooted at one. Not a place (T0019): a literal, a call result, an index element `xs[i]` or `xs[i].f`, a struct literal or its field, a top-level fn name, a local fn name, a bound method `p.m`.
  - **Captures.** A lambda, a local fn, a `_` capture or a bound method may capture a `mut` param (`mut self` included) when it does not escape; one that escapes is T0052.
  - **Not a value.** A fn with any `mut` param, or a method with a `mut` param after `self`, cannot be used as a value: T0049. A call through a function value passes every argument by value.
  - **Boxing.** A caller local passed as a `mut` argument counts as assigned, so if a closure captures it, it is boxed.
  - **Aliasing** is allowed: `swap(x, x)`, and `f(x, x)` with one `mut` and one by-value param. A by-value argument is the copy taken where it is evaluated, so a later argument that writes the variable through a `mut` param (`f(q, touch(q))`) does not change it. An assignment through a `mut` param builds the whole value before it writes any of it, so `a = P { x: b.y, y: b.x }` is right when `a` and `b` name one variable.

## `Ref[T]` becomes a plain library type

There is one sharing model, so the `&x` / `*r` operators are removed.

- **What stays:** a shared mutable cell that you store in a data structure. It is an ordinary std type, and its API is in `std/ref` (`import std/ref`):
  - `Ref.new(v)` makes one. It infers `T` from `v`; `Ref[T].new(v)` pins it.
  - `r.get()` reads it.
  - `r.set(v)` writes it.
- **Handles:** every copy of a `Ref` names the same cell, so a `set` through one copy is seen through all of them. `==` compares cells (identity). A `Ref` has no `Hash`, so a type that holds one has no automatic `Hash` either.
- **The type itself** stays a prelude type: `Ref[int]` is nameable in a type with no import, and so is `Ref` in value position. Its methods, like every impl's, are there when std/ref is part of the program: one module's `import std/ref` serves every module. In a program without it, a call of `new`, `get` or `set` is T0018 with the hint "add 'import std/ref'".
- **Old syntax:** P0015, a parse error whose message names the replacement: "'&x' was removed; make a shared cell with 'Ref.new(x)'", and "'*r' was removed; read a Ref with 'r.get()' and write one with 'r.set(v)'". A module with a parse error is not checked further, so P0015 is the only error it reports. Infix `a & b` and `a * b` are unchanged.
- **Associated-fn inference** is general: a bare `Box.new(5)` on any generic struct infers its type params from the value args, as a generic fn call does. A param-less associated fn (`Vec.new()`) still needs them written (`Vec[int].new()`, T0016).

## Bound methods

`f := acc.plus` is a closure that captures `acc` and calls `acc.plus(..)`.

- The receiver is captured like any other variable.
- For a `mut self` method, the receiver is assigned through the closure, so it is boxed.
- On a `mut` parameter (`mut self` included), the receiver is held by address, like a lambda's capture of it.

## `_` capture expressions

- **Meaning:** `f(a, _)` anywhere means `fn(v) { f(a, v) }`. This is Gleam's function capture. The call can be a fn, a fn value (`g(1, _)`), a method (`acc.plus(_)`), a tuple variant (`Shape.circle(_)`) or a tuple struct (`Pt(_, 4)`).
- **What a hole is:** a `_` that is a whole call argument, so a `_` followed by `,` or `)`. It is a hole even when a local named `_` is in scope. `f(_ + 1)`, `f((_))`, `f(_.x)`, `[_]` and `xs[_]` hold no hole.
- **One per call:** a call takes one `_`, because the function it makes takes one parameter. `add(_, _)` is P0016; write `fn(a: int, b: int) { add(a, b) }`.
- **Each call has its own `_`:** `f(g(_))` follows Gleam: the `_` captures only `g`'s call, so `f` gets `fn(v) { g(v) }`. That is what `xs.map(add(1, _))` needs. When `f` (a fn, a method or a constructor) wants a non-function there, the error is a type mismatch (T0040 for a call) with a hint to write the lambda around the outer call, `fn(v: int) { f(g(v)) }`. `f(g(_), _)` is two captures.
- **Every other `_` keeps its meaning:** a match wildcard, `_x` bindings, `_ := e`, `for _ in`, a `_` param, and a read of a `_` local in any other position. A `_` that is not a hole and names no local is R0001, with a hint that `_` makes a capture only as a whole call argument. Two things that read a `_` local before now make a capture: a `_` local passed as a whole argument, and a written `fn(_) { f(_) }`, whose body is now a capture of its own.
- **Inferred variants:** `.some(_)` is not a call, so it is not a capture; write `Option[int].some(_)`, or `Option.some(_)` where an expected fn type gives its `T`.
- **The parameter type**, in order:
  1. the expected fn type's parameter, when it is ground (as for any lambda);
  2. otherwise the type the call wants at the `_`, including what a generic call's other arguments bind: `inc := add(1, _)` is `fn(int) -> int`, `b.put(_)` on a `Box[int]` takes an `int`;
  3. otherwise T0050 "cannot infer the type of '_'", reported only when nothing else in the call is an error.

  `pair(1, _)` is `fn(int) -> int`; `id(_)`, where nothing binds `T`, stays the callee's T0016. A capture passed to a bare generic call is typed by its own call, so `twice(add(1, _), 5)` works.
- **Evaluation:** the other arguments are evaluated each time the capture is called, as in the lambda it means. `add(tick(), _)` calls `tick` on every call, and a `?` or `return` among them exits the capture. A capture has no `-> R` to write, so such a `?` needs an expected fn type that returns an Option or Result (T0032 otherwise), and a `break` or `continue` there is T0051, as in a lambda.
- **As a statement:** a capture dropped as a statement (`show(_)`) never runs its call, so it is W0008. Before captures, that statement passed a `_` local; rename the local to keep that meaning.
- **`mut` positions:** a `_` at a `mut` parameter is T0019. The capture's parameter is a copy of the value it is called with, so the change would be lost.
- **Pipes:** `x |> f(a, _)` is the call `f(a, x)`: no closure and no env. The piped value is evaluated at its position, in argument order, so in `x |> f(a, _)` the `a` runs first.
- **Methods:** `acc.plus(_)` is a short form of the bound method with an explicit argument. The receiver is captured like any other argument: a `mut self` receiver is boxed as an assigned capture is (so at `-O1` it may stay unboxed, see Escape analysis), and a `mut` parameter receiver is held by address.
- **Known gap:** in a bounded template (`fn g[T has P](t: T)`), a `_` whose position wants the template's own `T` is a false T0050, as a lambda with an untyped param is there. Write the lambda with a typed parameter, `fn(o: T) { t.add(o) }`.

## Method pipe

`x |> .m(args)` means `x.m(args)`: the piped value is the receiver. Pipes can then chain methods, and mix them with free fns: `xs |> .filter(pos) |> .map(double) |> sum`.

- **No arguments:** `x |> .m` is `x.m()`, as `x |> f` is `f(x)`.
- **Turbofish:** at most one, as in `acc |> .into[int]()`. It means exactly what `x.m[T]()` means, errors included: explicit protocol arguments on a receiver of a generic type (`b.scale[int](3)` on a `Box[T]`) are T0018 in both forms for now, "takes no explicit type arguments yet".
- **A value on the left:** the receiver must be a value, so `io |> .println(s)` and `Shape |> .circle(1)` are "not a value" errors, not the qualified calls `io.println(s)` and `Shape.circle(1)`.
- **`?` and chains:** `v |> .get(0)?` is `v.get(0)?`, and `v |> .get(0)? |> add(1)` unwraps before the next step. The target keeps a fixed shape, so `x |> .m().n()` and `x |> .m.n()` are P0014; chain with another `|>`.
- **`mut self`:** the piped value must be a place, as for `x.m()`: `mk() |> .bump()` is T0019. The call counts as a mutation of the receiver, so a closure that runs `c |> .bump()` boxes `c`.
- **No `_`:** the piped value already fills the receiver, so a `_` in the arguments is P0016.

## Standard library additions

| Type | Methods |
|---|---|
| `Vec` | `map`, `filter`, `each`, `fold(init, f)`, `any`, `all`, `find` → `Option`, `sort()` (for `Ord`), `sort_by(less)` |
| `Option` / `Result` | `map(f: fn(T) -> U)`, `then(f: fn(T) -> Option[U])` (and-then; `fn(T) -> Result[U, E]` on `Result`), `or_else(f: fn() -> T) -> T` (`fn(E) -> T` on `Result`), `map_err(f: fn(E) -> F) -> Result[T, F]` |
| `str` | `split(sep) -> Vec[str]`, `sep.join(parts: Vec[str])`, `trim`, `contains`, `starts_with`, `ends_with`, `replace(from, to)`, `chars()` → `Chars[char]`, an iterator of `char` |
| `Map` | `each(f: fn(K, V))`, `filter(f: fn(K, V) -> bool) -> Map[K, V]`, `map(f: fn(V) -> W) -> Map[K, W]` (maps the values), `keys() -> Vec[K]`, `values() -> Vec[V]` |
| `Set` | `each(f: fn(T))`, `filter(f: fn(T) -> bool) -> Set[T]`, `map(f: fn(T) -> U) -> Set[U]` (`U` Hashable) |

- **Sorting:** `sort` and `sort_by` sort in place and are stable; `less(a, b)` is true when `a` goes first. A method takes one arity per name, so the comparator form has its own name.
- **Walks:** they visit the elements present when the call starts, in index order. A callback that pushes onto the same `Vec`, or sets a new key in the same `Map`, does not extend the walk.
- **Order:** a `Map` or `Set` iterates in insertion order: the order keys were first added. Setting an existing key keeps its place, and there is no removal. `for`, `each`, `keys`, `values`, `filter` and `map` all follow it, and the maps and sets that `filter` and `map` return keep it. `Set.map` keeps one of two elements that map to one value, at the first one's place.
- **`keys` and `values`:** they return a `Vec`, so they are in `std/mapvec` (`import std/mapvec`). `std/map` does not import `std/vec`: that would put `Vec` into the graph of every program that uses a `Map`, and renumber the struct ids its binary embeds. Without the import, a call is T0018 "add 'import std/mapvec'".
- **Escapes:** a std method only calls the functions passed to it, so they never escape (see Escape analysis).
- **Option and Result:** the methods are in `std/option` (`import std/option`), as `Ref`'s methods are in `std/ref`. Without the import, a call is T0018 "add 'import std/option'". `is_some`, `unwrap`, `unwrap_or` and the other built-in methods need no import.
- **`or_else(f)`** is the lazy `unwrap_or`: `f` runs only on `none` or `err`.
- **Constructors in a `then` lambda:** write the constructor's type args (`Option[str].none`, `Result[int, str].ok(v)`) or annotate the lambda's return (`fn(v) -> Option[str] { .. }`). The lambda's return type is not known while `U` is open.
- **Same-named methods:** a program's own `impl Option[T] { fn map .. }` next to `import std/option`, or `impl str { fn trim .. }` next to `import std/str`, is not an error: the program's own method is the one every call reaches, whatever the import order, and std's of that name is not used. The same holds for a program's own `Vec`, `Map` or `Set` method.
- **str:** the methods are in `std/str` (`import std/str`); without the import, a call is T0018 "add 'import std/str'". `len`, `byte_at`, `concat` and `to_string` stay built in.
  - Offsets are bytes. The search is a byte-wise scan, which on valid UTF-8 never matches inside a char.
  - `split` and `trim` results share the original's bytes; nothing is copied.
  - `trim` strips ASCII whitespace only: space and bytes 9 to 13.
  - `split` keeps empty parts, so n separators give n + 1 parts. An empty separator gives `[self]`, and `replace("", to)` gives `self`.
  - `join` is called on the separator: `", ".join(parts)`. It sums the lengths first and copies each byte once, so `join`, and `replace` through it, are linear in the result.
  - `chars()` decodes UTF-8 and never panics: each invalid sequence (a stray byte, a truncated sequence, an overlong form, a surrogate, a code point past U+10FFFF) yields one U+FFFD. `C0` and `C1` never start a sequence, so `"\xC0\xAF"` gives two.

The names `unwrap`, `unwrap_or`, `is_err` and `map_err` may change; see issue #24. `or_else` follows the name `unwrap_or` gets.

## Escape analysis

- **Does not escape:** a closure each use of which is a call of it, or passes it whole to a by-value parameter that does not escape. A `:=`-bound closure is judged by its local's uses, and a local fn by its name's uses (its own body included).
- **Escapes:** returning it, storing it (a field, a struct, enum or list literal, a `Vec`), comparing it, capturing it in another closure, assigning the local, passing it at a `mut` position, or passing it through a fn value or to a builtin, an extern or an intrinsic.
- **Summaries:** which parameters of a function may escape, per function and per generic instance, computed bottom-up over the call graph. A call back into the caller's own cycle of calls that pass functions (recursion) escapes.
- **Legality is the same at every level:** `toy check`, `-O0` and `-O1` agree on T0052. It can depend on another module's body: a callee that starts keeping its fn param makes a `mut`-param capture passed to it an error.
- **`-O1` storage:** any one opt pass turns it on, not the level as such: `-O1` with every pass off (`--no-opt` for each) stores closures as `-O0` does, and `-O0 --opt=dce` stores them as `-O1` does.
  - a non-escaping env goes in the frame of the body that makes it, up to 256 B per function;
  - past that, or when the frame would exceed codegen's limit, it goes in a GC cell;
  - a captured-and-mutated local of 16 B or less that only non-escaping closures capture is not boxed: those closures hold its address;
  - bound values keep heap envs, and a bound receiver stays boxed.
- **`-O0`:** every env and box is on the heap. A captured `mut` param's address is a plain word in the env cell.
- **Gate:** an `-O0` against `-O1` differential run of the corpus (same exit code and stdout).

## Tooling

- **Tree-sitter:** lambdas, function types, nested functions, `_` capture and method pipes are in the grammar. Each lands with the syntax that introduces it. A `_` capture and a method pipe parse with the existing call, identifier and variant rules, so highlight queries alone tell them apart.
- **LSP captures:**
  - Hover on a captured name says "captured: copied (N bytes)" or "captured: shared (boxed)".
  - An optional inlay hint on a lambda lists its captures.
  - A `mut` argument shows "mutated by f".
- **LSP function values:** hover and signature help on calls of function-typed values and bound methods.

## Out of scope

- Deterministic `Display` for function values: issue #22.
- Renaming `Option` / `Result` methods: issue #24.
- `<T>` type arguments in expression position: issue #23.
- Full, HM-style inference of lambda parameters from their uses.
