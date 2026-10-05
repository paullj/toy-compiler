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
(or `zig build test-bin` then `./zig-out/bin/toy-integration-test`); the single-file
examples here and the multi-file programs under `modules/` are both covered. A moved,
renamed, or mis-annotated fixture fails the suite.
