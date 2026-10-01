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
  - With no expected type, every parameter needs a type annotation. A missing one is a coded error.
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
  - Parameters and loop variables follow the same rule.
  - Every use of a boxed local, in the outer function too, goes through the box.
- **Read-only captures** are copied into the environment. With no assignments, a copy and a share are indistinguishable.
- **Values keep today's rules:** `y := x` copies a struct and shares a `Vec` / `Map` / `Ref` handle.
- **Loop variables:** each iteration has a fresh binding. `for i in 0..3 { fs.push(fn() { i }) }` gives closures that return 0, 1 and 2.

## Representation

- **A closure value is a 16-byte pair `{fn, env}`:**
  - `fn` is the code pointer.
  - `env` points to a GC cell, which holds the copied captures and pointers to the boxed captures. The collector scans the cell word by word: it has no type descriptor yet.
  - `env` is null for a named function and for a lambda that captures nothing, so a named function as a value costs no allocation.
- **Calling a function value** is a `call_indirect`, with `env` as a hidden first argument.

## Generics

`fn map[T, U](xs: Vec[T], f: fn(T) -> U) -> Vec[U]` is instantiated in a hybrid way:

- **Known at compile time:** when the argument is a lambda literal or a named function, the call site gets its own instance. The call is then direct and can be inlined.
- **A function value:** a variable, a field or a return value goes through an erased instance, keyed only on `T, U`, with the fat pair and `call_indirect`.
- **A cap per generic function** on specialized instances limits code size. Past the cap, calls use the erased instance.

## `mut` parameters

```toy
fn bump(mut n: int) { n = n + 1 }
bump(count)                          # count changes
```

- **Mechanism:** a `mut` parameter refers to the caller's variable by address, with no heap. This is the same mechanism as `mut self`.
- **The argument must be a mutable place:** a local or a field of one. The error is T0019.
- **No marker at the call site.** The LSP shows "mutated by bump".
- **Escaping closures:** a `mut` parameter captured by a closure that escapes is a coded error, because the address would outlive the caller's frame. The hint is "copy it to a local first".
  - Until escape analysis lands, every capture of a `mut` parameter is that error.
- **Limits:**
  - **Where `mut` may appear.** On the params of top-level fns and of inherent-impl methods, static (associated) methods included, and on any `self` receiver where `mut self` is allowed today. It is P0006 on a lambda's params, a local fn's params, an `extern` fn's params, a protocol sig's or a conformance method's params after `self`, and any param past the 64th.
  - **The argument.** A local (a param, a forwarded `mut` param, a loop variable and a match binding included) or a field / tuple-field chain rooted at one. Not a place (T0019): a literal, a call result, an index element `xs[i]` or `xs[i].f`, a struct literal or its field, a top-level fn name, a local fn name, a bound method `p.m`.
  - **Captures.** Any capture of a `mut` param, by a lambda, a local fn or a bound method on it, is T0052 ("copy it to a local first").
  - **Not a value.** A fn with any `mut` param, or a method with a `mut` param after `self`, cannot be used as a value: T0049. A call through a function value passes every argument by value.
  - **Boxing.** A caller local passed as a `mut` argument counts as assigned, so if a closure captures it, it is boxed.
  - **Aliasing** is allowed: `swap(x, x)`, and `f(x, x)` with one `mut` and one by-value param. A by-value argument is the copy taken where it is evaluated, so a later argument that writes the variable through a `mut` param (`f(q, touch(q))`) does not change it. An assignment through a `mut` param builds the whole value before it writes any of it, so `a = P { x: b.y, y: b.x }` is right when `a` and `b` name one variable.

## `Ref[T]` becomes a plain library type

There is one sharing model, so the `&x` / `*r` operators are removed.

- **What stays:** a shared mutable cell that you store in a data structure. It is an ordinary std type:
  - `Ref.new(v)` makes one.
  - `r.get()` reads it.
  - `r.set(v)` writes it.
- **Old syntax:** a coded parse error with a fix hint, for example "`&x` was removed; use `Ref.new(x)`".

## Bound methods

`f := acc.plus` is a closure that captures `acc` and calls `acc.plus(..)`.

- The receiver is captured like any other variable.
- For a `mut self` method, the receiver is assigned through the closure, so it is boxed.

## `_` capture expressions

- **Meaning:** `f(a, _)` anywhere means `fn(v) { f(a, v) }`. This is Gleam's function capture.
- **Rules:** one `_` per call, and only as a direct argument. `f(g(_))` is an error.
- **Pipes:** `x |> f(a, _)` means `(fn(v) { f(a, v) })(x)`, which is `f(a, x)`. So the pipe needs no special case, as `pipe.md` promised.
- **Methods:** `acc.plus(_)` is a short form of the bound method with an explicit argument.

## Method pipe

`x |> .m(args)` means `x.m(args)`. Pipes can then chain methods: `xs |> .filter(pos) |> .map(double)`.

## Standard library additions

| Type | Methods |
|---|---|
| `Vec` | `map`, `filter`, `each`, `fold(init, f)`, `any`, `all`, `find` → `Option`, `sort()` (for `Ord`), `sort(by)` |
| `Option` / `Result` | `map`, `then` (and-then), `or_else(f)`, `map_err` |
| `str` | `split`, `join`, `trim`, `contains`, `starts_with`, `ends_with`, `replace`, `chars()` → an iterator of `char` |
| `Map` / `Set` | `each`, `filter`, `map` (values, for `Map`), `keys()`, `values()` |

The names `unwrap`, `unwrap_or`, `is_err` and `map_err` may change; see issue #24. `or_else` follows the name `unwrap_or` gets.

## Escape analysis (`-O1`)

- **What it finds:** a closure does not escape if it is only called, or only passed to parameters that do not escape.
  - A per-function summary records which parameters escape. It is computed bottom-up over the call graph. Recursion and unknown callees count as escaping.
- **What that allows:**
  - The environment of a non-escaping closure goes on the stack.
  - Its captured-and-mutated locals are not boxed.
  - A `mut` parameter may be captured by a non-escaping closure.
- **At `-O0`** everything is always boxed.
- **Gate:** an `-O0` against `-O1` differential run of the corpus (same exit code and stdout).

## Tooling

- **Tree-sitter:** lambdas, function types, nested functions, `_` capture and method pipes are in the grammar. Each lands with the syntax that introduces it.
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
