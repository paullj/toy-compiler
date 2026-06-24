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

`./examples/check.sh` builds the compiler, then for each example compiles (and,
for non-error cases, runs) it and verifies every `# expect:` directive. Exit 0
means the whole corpus matches its annotations.
