# Calling C (FFI)

A program calls a C function by declaring it as an `extern` and calling it inside
`unsafe { }`. Run examples: `tests/corpora/language-features/ffi/` and
`tests/corpora/language-features/modules/ffi_extern/`.

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
  line. The block records the library name for the linker; linking a library other
  than libSystem arrives with `-l`/`-L` (#149). A block without a name, `extern { .. }`,
  and a bare `extern fn` both bind to libSystem, which holds libc and libm.
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

## Not yet

- Libraries other than libSystem (`-l`, `-L`): #149.
- `cstr`, `f32` and C structs (`extern struct`): #150, #151.
- C callbacks and varargs (`printf`): #152, #153.
