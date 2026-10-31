# Calling C (FFI)

A program calls a C function by declaring it as an `extern` and calling it inside
`unsafe { }`. Run examples: `tests/corpora/language-features/ffi/` and
`tests/corpora/language-features/modules/ffi_extern/` and `ffi_dylib/`.

```toy
extern fn putchar(c: int32) -> int32

extern {
    fn labs(x: int) -> int
    fn toupper(c: int32) -> int32
}

fn main() -> int {
    unsafe { putchar(toupper(104)) }
    unsafe { putchar(10) }
    return unsafe { labs(-7) }
}
```

## Declaring externs

- A bare `extern fn name(params) -> R` declares one C function. It has no body, no type
  parameters and no `mut` params (P0006), and its name is the C symbol, never
  qualified by the module.
- An `extern "lib" { .. }` block declares several functions of one library, one per
  line, and links that library (see [Libraries](#libraries)). A block without a name,
  `extern { .. }`, and a bare `extern fn` both bind to libSystem, which holds libc and
  libm.
- An extern follows the `pub` rules of any fn: without `pub` it is private to its
  module, and `pub extern fn` or a `pub fn` inside a block exports it to importers,
  qualified (`c.labs`) or selected (`import c.{labs}`). `pub` goes on each function
  of a block, not on the block (P0003).
- The bundled `core/` modules declare the standard library's own externs. A module
  outside `core/` cannot name them (R0005); std wraps them in safe functions.

## Types

An extern's params and return use C ABI types: an integer of any width, `float` (C
`double`), `bool` and `rawptr` (C `void*`). Omitting `-> R` is a `void` return, and
`-> never` is a C function that does not return (`exit`, `abort`). Any other type is
T0037, reported once at the type; the extern stays declared, so its calls add no
second error.

Calls follow Apple's arm64 C convention: a narrow integer or `bool` returned by C is
widened to its toy width, and arguments past the eighth that go on the stack are packed
at their own size and alignment, as C expects.

A `str` is not allowed in a program's extern (T0037): it is a pointer and a length
with no NUL at the end, so a C function reading a `const char*` would run past it.
Use `rawptr` until the `cstr` type arrives (#150). Only `core/` uses a `str` there,
spread to a C `buf, len` pair for functions such as `write`.

## unsafe

The compiler cannot check what a C function does with its arguments, so every call to
an extern sits lexically inside `unsafe { }`, or it is T0079. The rule covers a bare
call, a qualified call (`c.labs(x)`), a pipe (`x |> labs`) and a call in a lambda
written inside the `unsafe` block. `unsafe { .. }` is an expression with the value of
its last line, so it wraps just the call: `n := unsafe { labs(x) }`. In a condition,
put it in parentheses: `if (unsafe { read(fd, p, 1) }) <= 0 { .. }`.

An extern is a symbol of another library, not toy code, so it cannot be used as a
value (T0049): wrap it in a toy function to pass it around.

A safe API wraps the call once and checks what C cannot:

```toy
extern { fn putchar(c: int32) -> int32 }

pub fn put(c: int32) {
    unsafe { putchar(c) }
}
```

## Libraries

Each library a program names becomes one `LC_LOAD_DYLIB` in the binary, and each C
function binds to the library of its block, so dyld looks it up there (a two-level
namespace). libSystem is always loaded first.

```toy
extern "sqlite3" {
    fn sqlite3_libversion_number() -> int32
}

extern "toyffi" {
    fn toyffi_add(a: int, b: int) -> int
}

fn main() -> int {
    if (unsafe { sqlite3_libversion_number() }) < 3000000 { return 1 }
    unsafe { toyffi_add(40, 2) }
}
```

```
toy build main.toy -L path/to/lib     # finds path/to/lib/libtoyffi.dylib
toy run main.toy -L path/to/lib -l z  # also loads libz
```

**Finding a library.** A name such as `"toyffi"` or `-l toyffi` is looked up as
`libtoyffi.dylib` in each `-L` folder in order, then in `/usr/local/lib` and
`/usr/lib`, then in the dyld shared cache, where macOS keeps its system libraries
(`z`, `sqlite3`, `curl`, ...) with no file on disk. A name that contains a `/` or ends
in `.dylib` is a path, used as given, relative to the working directory. `"c"`, `"m"`,
`"pthread"`, `"dl"` and `"System"` name libSystem itself and add nothing. The same
library named twice, by blocks in several modules or by `-l`, is loaded once. A block
with no functions binds nothing, so it links nothing; use `-l` to load a library
without declaring a function of it.

**`-l` and `-L`.** `build`, `run` and `check` take both, repeatable, attached or not
(`-lz`, `-L lib`). `-l name` loads a library even when no block names it; a function
still binds to a library only through an `extern "name"` block, and a bare `extern fn`
always binds to libSystem. `check` resolves the libraries as `build` does, so it
reports the same errors. A `-L` that is not a folder is a usage error.

**Errors.** Each is reported once, at the block's library string, or for a `-l` as a
usage error (exit 2) naming the flag:

| code | when |
|---|---|
| R0015 | the name is a static archive or object (`.a`, `.o`), or only `lib<name>.a` exists |
| R0016 | no `lib<name>.dylib` in any searched folder; the error lists them |
| R0017 | the file is not an arm64 Mach-O dynamic library |
| R0018 | one C symbol is bound to two different libraries, or a block binds one the toy runtime or std imports from libSystem |

Static linking (`.a`, `.o`) is not supported; build the library with `cc -shared`.

**Where dyld finds it at run time.** The binary records each library by its own
install name (its `LC_ID_DYLIB`), as the system linker does:

- An absolute install name (`/usr/lib/libz.1.dylib`, `/opt/homebrew/opt/x/lib/libx.dylib`)
  is recorded as is; the library must be there when the program runs.
- `@rpath/libx.dylib` (what Zig and CMake write) is recorded as is, plus an
  `LC_RPATH` for the folder the library was found in, so the program runs where it
  was built. To ship it elsewhere, put the library where the rpath points, or run
  with `DYLD_LIBRARY_PATH`.
- `@loader_path/..` and `@executable_path/..` are recorded as is: the library must sit
  at that place relative to the binary.
- A bare install name (`libx.dylib`, what `cc -shared` writes without
  `-install_name`) would make dyld look in the working directory, so the absolute path
  the library was found at is recorded instead.

The libraries are worked out again on every build from the sources and flags; the
linked image is never cached, so a warm build after changing `-l`, `-L` or a block's
library relinks against the new set.

## Not yet

- Static libraries (`.a`, `.o`), and libraries in a project manifest.
- `cstr`, `f32` and C structs (`extern struct`): #150, #151.
- C callbacks and varargs (`printf`): #152, #153.
