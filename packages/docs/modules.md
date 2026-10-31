# Modules and imports

A module is a file. `import a/b` loads `<root>/a/b.toy`, where the root is the entry
file's folder; `std/*` paths name the bundled standard library. Imports form a DAG: a
cycle is an error. Run examples: `tests/corpora/language-features/modules/`.

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
not see it.

## Errors

| Code | When |
|---|---|
| R0003 | the imported module does not exist |
| R0004 | two imports, or an import and a top-level declaration, bind one name |
| R0005 | a referenced or selected item is not `pub` |
| R0006 | the module has no item of a referenced or selected name |
| R0013 | a user module imports a private `core/` module, whole or selectively |
| W0004 | an import, or a selected name, is never used |

`toy explain <code>` shows the page for each.
