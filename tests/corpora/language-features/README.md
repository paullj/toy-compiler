# examples

Each `.toy` file exercises **one** language feature and is annotated with a
machine-readable header so the set doubles as a test corpus. Everything here
either runs cleanly or, under `errors/`, fails compilation on purpose.

## Layout

| Dir             | Feature area                                              |
| --------------- | --------------------------------------------------------- |
| `basics/`       | literals, arithmetic, bindings, blocks, unit, exit code   |
| `functions/`    | calls, forward references, recursion, the `never` type    |
| `control-flow/` | if/else, while/loop/for, break/continue, labels, `&&`/`\|\|` |
| `structs/`      | construction, by-value copy, field stores, struct returns |
| `io/`           | runtime output via `print`                                |
| `closures/`     | function types, function values, lambdas, local fns, bound methods, specialized generic calls, `_` captures and escape analysis (frame envs at -O1) |
| `generics/`     | generic fns and structs, type-arg inference (incl. lambdas and `_` from the other args), methods with their own type params |
| `places/`       | `mut` parameters: by-address arguments (locals, fields), forwarding, generics, aliasing, captured by non-escaping closures (incl. bound methods) |
| `ref/`          | `Ref[T]` shared cells from `std/ref`: `Ref.new`, `get` / `set`, cell identity `==`, inferred `T` |
| `vec/`          | `Vec[T]` from `std/vec`: literals, `push` / `get`, indexing, and the closure methods `each`, `map`, `filter`, `fold`, `any`, `all`, `find`, `sort`, `sort_by` (stable, in place) |
| `option/`       | the prelude `Option[T]` / `Result[T, E]`: constructors, `?`, the built-in `is_*` / `unwrap` / `unwrap_or`, and the closure methods from `std/option`: `map`, `then`, `or_else` (the lazy `unwrap_or`), `map_err` |
| `map/`          | `Map[K, V]` from `std/map`: `set` / `get`, `for k, v`, and the closure methods `each`, `filter`, `map` (the values), plus `keys()` / `values()` as a `Vec`, all in insertion order |
| `set/`          | `Set[T]` from `std/set`: `add` / `contains`, `for x`, and the closure methods `each`, `filter`, `map`, in insertion order |
| `string/`       | the prelude `str`: `concat`, `to_string`, and the methods from `std/str`: `split`, `join`, `trim`, `contains`, `starts_with`, `ends_with`, `replace`, and `chars()` (UTF-8 decoding, U+FFFD for invalid input) |
| `gc/`           | the mark-sweep collector: roots, tracing, reclaiming churned garbage, and `std/gc`'s `collect()` / `collections()` |
| `frames/`       | function frames past the 4 KiB a single instruction addresses (about 5 KiB, 40 KiB and 1 MiB), with heap roots at the far end across collections |
| `errors/`       | programs that are *supposed* to fail to compile           |

## Annotation convention

Every file starts with a `# Feature:` line and one or more `# expect:` lines:

```
# Feature: <one line describing the feature exercised>
# expect: exit <N>             # process exit code (main's int return, & 0xFF)
# expect: stdout "<bytes>"     # exact stdout; \n / \t / \" escapes
# expect: compile-error "<substr>"  # must fail to compile; substr appears in a diagnostic
```

A file may carry several `expect` lines (e.g. both `stdout` and `exit`). The
prose after `# Feature:` and any further `#` comments explain the expected
result so the example is self-documenting.

## Running

The manifest-driven corpus harness (`packages/compiler/tests/corpus.zig`, enumerated
by `corpus_manifest.zon`) compiles each example (and, for non-error cases, runs it)
and verifies every `# expect:` directive in-process. It runs under `zig build test`
(or `zig build test-bin` then `./zig-out/bin/toy-corpus-test`); the single-file
examples here and the multi-file programs under `modules/` are both covered. It reports
every failing fixture, and `TOY_CORPUS=<path substring>` runs only the matching ones
(`TOY_CORPUS=closures/capture_loop ./zig-out/bin/toy-corpus-test`). The manifest holds
each corpus's exact fixture count, so adding or removing a fixture means updating it,
and a fixture that goes missing fails the suite.
