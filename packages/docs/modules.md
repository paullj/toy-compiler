# Modules and imports

A module is a file. `import a/b` loads `<root>/a/b.toy`, or `<root>/a/b/mod.toy` for a
folder (see [Folder modules](#folder-modules)), where the root is the entry file's folder
unless `--root` says otherwise (see [Search roots](#search-roots)); `std/*`
paths name the bundled standard library. Imports form a DAG: a cycle is an error. Run
examples: `tests/corpora/language-features/modules/`.

Only `pub` items cross a module boundary: fns, structs, enums, protocols and type
aliases. A struct's methods and conformances, and an enum's variants, come with it.

## Whole-module imports

`import a/b` binds the namespace `b`, and the program names each item through it:
`b.item`. `import a/b as c` binds it as `c` instead.

```toy
import std/io

fn main() {
    io.println("hi")
}
```

A whole-module import also makes the module's `pub` structs and enums nameable bare
(`Vec` after `import std/vec`), unless the importing module has its own type of that
name.

A type is named through its module in any type position (`shapes.Meters`, a `pub` type
alias included), and so is a variant in a match pattern, through any depth of
re-exported namespaces: `util.Dir.up`, `util.extra.Dir.at(n)`,
`extra.Shape.box { w, h }`.

## Selective imports

`import a/b.{x, y as z}` binds only the listed names: `x`, and `y` under the name `z`.
It does not bind the module name. To use the namespace too, write a second,
whole-module import.

```toy
import std/io.{println}
import geo/shapes.{
    Rect,
    Color as C,
    area,
}

fn main() {
    r := Rect { w: 2, h: 3 }
    println(area(r))
    _c := C.green
}
```

The list can span lines and may end with a comma. Each selected name is the item
itself, so diagnostics and editor hovers show the item's own name (`Color`, not `C`).

A selected name is local to the importing module: a module that imports this one does
not see it, unless the import is a `pub import`.

## Re-exports

`pub import` re-exports what the import binds. `pub import a/b.{x}` makes `x` an item
of the importing module, and a whole-module `pub import a/b` makes the namespace `b`
one. Importers see a re-exported name as if the module declared it: selectively
(`import util.{x}`) and qualified (`util.x`, `util.b.item`). It is still the item
itself, so a re-exported type keeps its methods, conformances and variants, and a
whole-module `import util` makes a re-exported struct or enum nameable bare as well.

```toy
// util.toy
pub import util/shapes.{Rect, area as size}
pub import util/extra

// main.toy
import util.{Rect, size}
import util

fn main() -> int {
    r := Rect { w: 2, h: 3 }
    return size(r) + util.size(r) + util.extra.twice(1)
}
```

This lets a module gather its children's items into one surface. A plain `import`
re-exports nothing, and only `pub` items can be re-exported. A `pub import` is used by
the modules that import this one, so it never warns W0004.

A `pub import` is an import, so re-exports that loop are an import cycle (R0012). No
module outside `core/` may re-export a `core/` module, whole or selectively (R0013):
that would hand the standard library's private layer to every importer.

## Folder modules

A folder can be a module. `import util` loads `util.toy`, or `util/mod.toy` when the
folder holds one; `import util/math` still loads `util/math.toy`. The module is named
`util` either way, in diagnostics and in the namespace it binds; `import util/mod` is
R0003, since that file is the module `util`. Exactly one of the two files may exist: a
`util.toy` beside a `util/mod.toy` is R0014, which names both.

`mod.toy` imports its children by their full path and can re-export their items, so a
folder presents one surface:

```toy
// util/mod.toy
pub import util/geo/rect.{area}
import util/math

pub fn side() -> int { math.sq(2) }

// main.toy
import util.{area}
import util

fn main() -> int {
    return area(3, 4) + util.side()
}
```

Folder modules work in every search root, and a missing module's R0003 lists both files
it looked for in each root (`'util.toy', 'util/mod.toy'`). Turning `util.toy` into
`util/mod.toy`, or back, needs no clean build.

Because children are imported by their path from the module root, checking
`util/mod.toy` on its own (`toy check util/mod.toy`, or opening it in an editor with no
`root` set) roots imports at `util/` and cannot find `util/math`. Pass `--root .`, or set
the language server's `root`.

## Search roots

Every command that compiles (`toy build`, `toy run`, `toy check`, and a bare `toy`)
takes the same two options:

- `--root <dir>` sets the module root. The default is the entry file's folder.
- `-I <dir>` (or `--include <dir>`) adds an extra search root. It can be given more
  than once.

A `std/` or `core/` path always names the bundled library. Any other import path is
looked up in the module root and in every `-I` root, and exactly one of them must hold
the file (`<path>.toy` or `<path>/mod.toy`). A module that two roots both hold is R0014,
which lists every match, so no root silently shadows another. The same file reached
through two roots (an `-I` that repeats the module root) is one module. One file
imported under two paths is R0014 too: with `-I util`, `import util/math` and
`import math` cannot both appear in one program.

```
project/
  src/main.toy      # import extra
  lib/extra.toy
```

`toy build src/main.toy -I lib` builds it; `toy build src/main.toy --root .` does too
if `main.toy` says `import lib/extra` instead. A root name and an import path never
combine to leave the root: `..`, absolute and empty path segments are R0003. Paths in
diagnostics are the root as given joined with the import path, so cwd-relative roots
give cwd-relative locations (`lib/extra.toy:3:5`).

A directory that does not exist is a usage error (`invalid value 'nope' for -I (a
directory)`, exit 2). Changing the roots between builds is safe: imports are looked up
again on every build, and the cache only reuses work for the files they resolve to.

### In an editor

The language server (`toy lsp`) resolves imports with the same code. It reads the roots
from the client's `initializationOptions` when the session starts:

```json
{ "root": "src", "include": ["lib", "/opt/toy/vendor"] }
```

`root` is the module root and `include` the extra roots, in order. A relative path is
taken against the workspace folder (`rootUri`). With no `root`, each document's own
folder is its root, as with the CLI. Clients expose this as a server setting (Neovim's
`init_options`, for example); restart the server after changing it.

## Editor support

The language server knows the module system:

- **Import paths.** After `import ` or a `/` in a path, completion offers the next path
  segment: a module (`x.toy`, or a folder with a `mod.toy`) or a folder of modules, from
  the module root, every `include` root and the bundled `std/`. Open, unsaved files
  count. The file being edited, `mod` and the private `core/` are never offered.
- **Selection lists.** Inside `import a/b.{` completion lists what `a/b` exports: its
  `pub` items, and the names its `pub import`s bind, under their re-exported names.
  `b.` after a whole-module import lists the same set.
- **Navigation.** Go-to-definition and hover on a selected name or its `as` alias reach
  the item itself, through any chain of re-exports, as do uses of that name in the
  module (`square(2)`, `util.square(2)`). On a path segment they reach the module's
  file, and hover shows `module a/b`.

Find-references is not supported yet.

The tree-sitter grammar (`packages/tree-sitter-toy`) parses every import form. Its
highlight query marks path segments and whole-module aliases as `@module`, and a
selected name or alias as `@type` when it is capitalised and as `@function` otherwise,
since the grammar cannot tell which item a name selects.

## Errors

| Code | When |
|---|---|
| R0003 | the imported module does not exist in any search root, its path leaves the root, or it names a folder module's `mod` file |
| R0004 | two imports, or an import and a top-level declaration, bind one name |
| R0005 | a referenced or selected item is not `pub`, or is imported without `pub import` |
| R0006 | the module has no item of a referenced or selected name |
| R0012 | imports, or re-exports, form a cycle |
| R0013 | a user module imports a private `core/` module, or any module outside `core/` re-exports one |
| R0014 | more than one file could be the imported module (two search roots hold it, or `util.toy` sits beside `util/mod.toy`), or one file is imported under two paths |
| T0011 | a selected type is bound under a builtin type's name (`Shape as int`) |
| W0004 | an import, or a selected name, is never used |

`toy explain <code>` shows the page for each.
