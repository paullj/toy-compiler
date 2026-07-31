# examples/modules

Multi-file programs. Unlike the single-file corpus in `examples/`, each
program here is a **directory** whose entry file is `main.toy`; the compiler
discovers the rest of the module graph by following that entry's transitive
imports (`import a/b` -> `<root>/a/b.toy`, root = the entry file's directory).

These live in their own subtree because the non-entry files have no `main` and
must never be compiled standalone — the corpus harness prunes `modules/` from the
single-file walk and drives this directory as its own `module_dir` corpus, keying
on each `main.toy` entry.

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

The corpus harness (`packages/compiler/tests/corpus.zig`) drives this subtree as a
`module_dir` corpus: for each `main.toy` it discovers the module graph from that
entry and verifies the directives in-process. Run it under `zig build test` (or
`zig build test-bin` then `./zig-out/bin/toy-integration-test`).
