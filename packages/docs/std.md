# The standard library: builtins, core and std

Status: **implemented** for the layers and rules below. Later std milestones add their
own API sections to this page. Source: `packages/lib/core/*.toy` and
`packages/lib/std/*.toy`, bundled into the compiler at build time (`build.zig`, the
`BundledStd` module list).

A toy program sees three layers. Each one may use the layers below it, and a user
program may use only builtins and std.

| layer | where | who may name it |
|-------|-------|-----------------|
| builtins | the compiler (`symbols/Prelude.zig`, `symbols/Intrinsic.zig`, `resolve_graph.zig` `seeded_builtins`) | everyone, as scoped below |
| core | `packages/lib/core/*.toy` | bundled modules only (std and core) |
| std | `packages/lib/std/*.toy` | everyone, through `import std/<name>` |

## Builtins

Builtins have no module. The compiler registers them natively, so they need no import
and add no module-graph or content-fingerprint surface.

- **The prelude** (`symbols/Prelude.zig`): the scalar types (`int` and its widths,
  `float`, `bool`, `str`, `()`, `char`), the protocols `Eq`, `Ord`, `Add`, `Sub`, `Mul`,
  `Div`, `Hash`, `Display`, `From`, `Into`, `TryInto` and `Hashable`, the enums
  `Ordering`, `Option`, `Result` and `ConvErr`, and the box types `Ref` and `gc_array`.
  The protocol ids are load-bearing (see the header of `Prelude.zig`). The methods of
  `Option`, `Result`, `Ref` and `str` live in std and exist only when the program
  imports `std/option`, `std/ref` or `std/str`.
- **Seeded builtin fns** (`resolve_graph.zig` `seeded_builtins`): `panic` is global.
  Every other one is an intrinsic, listed once in `symbols/Intrinsic.zig` with its
  scope:
  - `.bundled`: `ref_new`, `ref_get`, `ref_set`, `gc_collect` and `gc_stats`. They take
    no pointer, so std may name them without breaking memory safety.
  - `.core`: the raw heap and pointer intrinsics (`gc_alloc`, `load`, `store`,
    `offset`, `size_of`, `descriptor_of`, `load_byte`, `store_byte`, ...). Only core
    names them, and `load`/`store` need an `unsafe { }` block (T0038).
- **Universal methods**: `x.to_string()` on every `Display` value, and `x.hash()` on
  every `Hash` value (see `typesystem.md`).

## core

core is the private layer between the compiler and std. It is the only place where:

- an `extern fn` may be declared (R0010 elsewhere), so every C symbol enters here;
- a `.core` intrinsic may be named.

| module | holds |
|--------|-------|
| `core/sys` | the libSystem `write` extern and `print(s)`, which writes to fd 1 |
| `core/ffi` | the libm and `labs` externs and their safe wrappers |
| `core/mem` | the growable `gc_array` and the hash table behind Vec, Map and Set |
| `core/text` | raw `str` parts: `slice`, and the `alloc`/`put`/`take` buffer that builds a str once |
| `core/mem_selftest`, `core/desc_selftest` | heap and descriptor self-checks |

## std

std is the public library. It wraps core in safe APIs: std never declares an `extern`
and never names a `.core` intrinsic.

| module | holds |
|--------|-------|
| `std/io` | `print` and `println`, ordinary generics over `T has Display` |
| `std/vec`, `std/map`, `std/set` | the collections, over core/mem |
| `std/str` | methods on `str`, over core/text |
| `std/option` | the closure methods of `Option` and `Result` |
| `std/ref` | `Ref.new`, `get` and `set` |
| `std/iter` | the `Iterator` protocol that `for`-in desugars to |
| `std/math` | integer and float math, over core/ffi |
| `std/gc` | `collect()` and `collections()` |

## Conventions

Every std and core module follows these rules.

- **Environment failures are an Option or a Result, never a panic.** A missing file, an
  unset variable, end of input or a bad number is a value the caller matches on. A panic
  is for a broken invariant in the program itself, such as an index out of range.
- **No hidden global state.** A std function's result depends only on its arguments and
  the outside world it names, such as a file or the clock. State a program keeps, like a
  random number generator, is a value the program holds and passes.
- **stdout and stderr are unbuffered.** Each print is one or more `write` calls before it
  returns, so output interleaves in program order, survives a panic or an exit, and needs
  no flush.
- **Each Darwin-only symbol sits behind one core wrapper.** A private or
  platform-specific symbol (for example `_NSGetArgv`, `__error`) is declared once in core
  and reached only through that one wrapper fn, so porting to another target changes the
  wrapper bodies and nothing in std.
- **core is private (R0013).** A user module that imports `core/*` gets R0013, which
  names the std module to import instead. Each core module has a row in
  `StdNames.core_replacements`, and a unit test walks the bundle so a new core module
  cannot ship without one. The self-check modules map to `std/gc`. The dev-only
  `--dev-core-imports` flag lifts the rule.
- **Tests force a collection through `std/gc`.** No test or fixture imports core: a test
  that must observe a collection calls `gc.collect()` and reads `gc.collections()`, which
  also counts collections an allocation triggered.
- **Only what a program calls is linked.** The linker drops every fn unreachable from the
  entry, so a std addition costs nothing in a program that does not call it.
