# examples/modules

Multi-file (M14) programs. Unlike the single-file corpus in `examples/`, each
program here is a **directory** whose entry file is `main.toy`; the compiler
discovers the rest of the module graph by following that entry's transitive
imports (`import a/b` -> `<root>/a/b.toy`, root = the entry file's directory).

These live in their own subtree because the non-entry files have no `main` and
must never be compiled standalone — `examples/check.sh` prunes `modules/`, and
this directory ships its own harness.

## Layout

| Program                       | Exercises                                                       |
| ----------------------------- | -------------------------------------------------------------- |
| `app/`                        | diamond import, `as` alias, cross-module type by value, prelude, pub/private |
| `errors/private_ref/`         | referencing a non-`pub` decl across modules                    |
| `errors/cycle/`               | an import cycle (graph must be a DAG)                           |
| `errors/missing/`             | importing a module file that does not exist                    |
| `errors/main_not_in_entry/`   | `main` defined only in an imported module                      |
| `errors/collision/`           | two imports sharing a last path segment without `as`           |

## Annotation convention

Identical to the single-file corpus (see `../README.md`): the `# expect:`
directives live in the entry `main.toy`.

```
# expect: exit <N>
# expect: stdout "<bytes>"
# expect: compile-error "<substr>"   # must fail to compile; substr appears in a diagnostic
```

## Running

`./examples/modules/check.sh` resolves `zig-out/bin/toy` (building it if
absent — never via `zig build test`), then for each `main.toy` compiles from its
directory (so the import root is correct) and verifies the directives. Exit 0
means the module corpus matches its annotations.
