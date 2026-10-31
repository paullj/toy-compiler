# examples/modules

Multi-file programs. Unlike the single-file corpus in `examples/`, each
program here is a **directory** whose entry file is `main.toy`; the compiler
discovers the rest of the module graph by following that entry's transitive
imports (`import a/b` -> `<root>/a/b.toy`, or `<root>/a/b/mod.toy` for a folder
module; root = the entry file's directory).

These live in their own subtree because the non-entry files have no `main` and
must never be compiled standalone — the corpus harness prunes `modules/` from the
single-file walk and drives this directory as its own `module_dir` corpus, keying
on each `main.toy` entry.

## Layout

| Program                       | Exercises                                                       |
| ----------------------------- | -------------------------------------------------------------- |
| `app/`                        | diamond import, `as` alias, cross-module type by value, prelude, pub/private |
| `fn_values/`                  | a qualified fn (`ops.add`) as a value; fn identity across modules |
| `mut_params/`                 | an imported module's `mut` params and `mut self` take this module's places |
| `own_generic_method/`         | an imported generic type's method with its own type param, two instances on one receiver |
| `own_generic_method_concrete/` | an imported non-generic struct's method and associated fn with their own type param, inferred and explicit |
| `display_generic/`            | an imported generic struct's instance renders under its template name (`Bx{v: 2}`) |
| `associated_fn/`              | an imported non-generic struct's and enum's associated fns, called through the module (`shapes.Rect.square(3)`) |
| `qualified_literal/`          | `lib.Pt { .. }`, inferred `lib.Pair { .. }` and `shapes.Shape.sq { .. }` beside same-named local types |
| `selective/`                  | `import a/b.{x, y as z}` binding a fn, generic fn, struct, enum, protocol and type alias bare |
| `reexport/`                   | `pub import` re-exporting selected items and a whole namespace, used selectively and qualified (`util.extra.twice`) |
| `qualified_alias/`             | a pub type alias named through its module (`shapes.Meters`) in every type position, and through re-exports |
| `qualified_patterns/`          | match patterns naming variants through modules (`util.Dir.up`, `util.extra.Dir.at(n)`, `extra.Dir.box { w, h }`) |
| `errors/private_alias/`       | a type alias without `pub` named through its module            |
| `errors/private_pattern/`     | a pattern naming a variant of a non-`pub` enum through its module |
| `errors/bad_pattern_qualifier/` | a pattern qualifier with a segment that is not a module (`util.Dir.zz.down`) |
| `search_roots/`               | `# include:` extra roots (the CLI's `-I`) beside the module root |
| `errors/root_clash/`          | a module both the module root and an extra root hold (R0014)    |
| `folder/`                     | a folder module (`import util` -> `util/mod.toy`) importing and re-exporting its children |
| `errors/folder_clash/`        | `util.toy` beside `util/mod.toy` (R0014)                        |
| `errors/folder_mod_path/`     | `import util/mod` instead of `import util` (R0003)              |
| `errors/path_alias/`          | one file imported under two paths through nested roots (R0014)  |
| `errors/private_ref/`         | referencing a non-`pub` decl across modules                    |
| `ffi_extern/`                 | a module's `pub extern` called qualified and selected inside `unsafe`; a private block extern behind a safe wrapper |
| `ffi_dylib/`                  | two modules name one user library (loaded once); safe wrappers over a C handle |
| `errors/extern_conflict/`     | one C symbol bound to libz in one module and to libSystem in another (R0018) |
| `errors/private_extern/`      | an extern without `pub` named through its module               |
| `private_fields/`             | `pub` fields used from another module; private ones only through the owner's methods and derives |
| `errors/private_literal/`     | a module-qualified literal of a non-`pub` struct                |
| `errors/private_field/`       | reading a field without `pub` from another module               |
| `errors/cycle/`               | an import cycle (graph must be a DAG)                           |
| `errors/reexport_cycle/`      | re-exports that loop are an import cycle                        |
| `errors/reexport_private/`    | a plain import's names are not re-exported                      |
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
# include: "<dir>"                   # an extra import root, relative to the program's folder
```

## Running

The corpus harness (`packages/compiler/tests/corpus.zig`) drives this subtree as a
`module_dir` corpus: for each `main.toy` it discovers the module graph from that
entry and verifies the directives in-process. Run it under `zig build test` (or
`zig build test-bin` then `./zig-out/bin/toy-integration-test`).
