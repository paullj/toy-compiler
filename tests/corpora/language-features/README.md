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
| `option/`       | the prelude `Option[T]` / `Result[T, E]`: constructors, `?`, the built-in `is_*` / `expect` / `or`, and the closure methods from `std/option`: `map`, `then`, `or_else` (the lazy `or`), `map_error` |
| `map/`          | `Map[K, V]` from `std/map`: `set` / `get`, `for k, v`, and the closure methods `each`, `filter`, `map` (the values), plus `keys()` / `values()` as a `Vec`, all in insertion order |
| `set/`          | `Set[T]` from `std/set`: `add` / `contains`, `for x`, and the closure methods `each`, `filter`, `map`, in insertion order |
| `string/`       | the prelude `str`: `concat`, `to_string`, and the methods from `std/str`: `split`, `join`, `trim`, `contains`, `starts_with`, `ends_with`, `replace`, and `chars()` (UTF-8 decoding, U+FFFD for invalid input) |
| `os/`           | `std/os`: `args()` (with `# args:`), `env(name)` (with `# env:`), and `exit(code)` |
| `time/`         | `std/time`: `now()` and `Instant.elapsed()`, `Duration` math, `sleep(d)`, and `unix()` / `unix_ns()` |
| `rand/`         | `std/rand`: `seeded(seed)` and `from_os()` Rngs, `next`, `below`, `range`, `float`, `bool`, and `choose` / `shuffle` |
| `fs/`           | `std/fs`: `read_file`, `write_file`, `append_file`, `remove` and `exists` in the fixture's cwd, each `IoError`, and `File` handles from `open`, `create` and `append` (chunked reads, writes, use after close) |
| `float/`        | `std/float`: float `Display` (shortest round-trip digits, exponent outside [1e-4, 1e16), inf/NaN, fields) and `float.parse` round trips |
| `gc/`           | the mark-sweep collector: roots, tracing, reclaiming churned garbage, and `std/gc`'s `collect()` / `collections()` |
| `frames/`       | function frames past the 4 KiB a single instruction addresses (about 5 KiB, 40 KiB and 1 MiB), with heap roots at the far end across collections |
| `errors/`       | programs that are *supposed* to fail to compile           |

## Annotation convention

Every file starts with a `# Feature:` line and one or more `# expect:` lines:

```
# Feature: <one line describing the feature exercised>
# expect: exit <N>             # process exit code (main's int return, & 0xFF)
# expect: stdout "<bytes>"     # exact stdout; \n / \r / \t / \" escapes
# expect: stderr "<substr>"   # stderr CONTAINS substr (repeatable); same escapes
# expect: compile-error "<substr>"  # must fail to compile; substr appears in a diagnostic
```

`# expect: stderr` is a substring match, not an exact one: a panic's backtrace
ends in unsymbolized frames whose addresses change with ASLR.

A program's run can also be given input:

```
# stdin: "<bytes>"             # piped to stdin; repeats concatenate; empty by default
# args: "a" "b c"              # argv[1..], each word quoted; repeats append
# env: K=V                     # the environment holds only these variables
```

stdin is always a pipe, never the harness's own, and the environment starts empty.
Each program runs with its worker's scratch directory as cwd.

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
