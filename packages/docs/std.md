# The standard library: builtins, core and std

Status: **implemented** for the layers and rules below. Later std milestones add their
own API sections to this page. Source: `packages/lib/core/*.toy` and
`packages/lib/std/*.toy`, bundled into the compiler at build time (`build.zig`, the
`BundledStd` module list).

A toy program sees three layers. Each one may use the layers below it, and a user
program may use only builtins and std.

| layer | where | who may name it |
|-------|-------|-----------------|
| builtins | the compiler (`symbols/Prelude.zig`, `symbols/Intrinsic.zig`, `resolve_graph.zig` `seeded_builtins`) | everyone, as scoped below |
| core | `packages/lib/core/*.toy` | bundled modules only (std and core) |
| std | `packages/lib/std/*.toy` | everyone, through `import std/<name>` |

## Builtins

Builtins have no module. The compiler registers them natively, so they need no import
and add no module-graph or content-fingerprint surface.

- **The prelude** (`symbols/Prelude.zig`): the scalar types (`int` and its widths,
  `float`, `bool`, `str`, `()`, `char`), the protocols `Eq`, `Ord`, `Add`, `Sub`, `Mul`,
  `Div`, `Hash`, `Display`, `From`, `Into`, `TryInto` and `Hashable`, the enums
  `Ordering`, `Option`, `Result` and `ConvErr`, and the box types `Ref` and `gc_array`.
  The protocol ids are load-bearing (see the header of `Prelude.zig`). The methods of
  `Option`, `Result`, `Ref` and `str` live in std and exist only when the program
  imports `std/option`, `std/ref` or `std/str`.
- **Seeded builtin fns** (`resolve_graph.zig` `seeded_builtins`): `panic` is global.
  Every other one is an intrinsic, listed once in `symbols/Intrinsic.zig` with its
  scope:
  - `.bundled`: `ref_new`, `ref_get`, `ref_set`, `gc_collect` and `gc_stats`. They take
    no pointer, so std may name them without breaking memory safety.
  - `.core`: the raw heap and pointer intrinsics (`gc_alloc`, `load`, `store`,
    `offset`, `size_of`, `descriptor_of`, `load_byte`, `store_byte`, ...). Only core
    names them, and `load`/`store` need an `unsafe { }` block (T0038).
- **Universal methods**: `x.to_string()` on every `Display` value, and `x.hash()` on
  every `Hash` value (see `typesystem.md`).

## core

core is the private layer between the compiler and std. It is the only place where:

- an `extern fn` may be declared (R0010 elsewhere), so every C symbol enters here;
- a `.core` intrinsic may be named.

| module | holds |
|--------|-------|
| `core/sys` | the libSystem `write` and `read` externs: `print(s)` to fd 1, `write_fd`, `read_fd_all` and `read_line_fd` |
| `core/ffi` | the libm and `labs` externs and their safe wrappers |
| `core/os` | the `_NSGetArgc`, `_NSGetArgv`, `getenv`, `strlen`, `exit`, `__error`, `clock_gettime_nsec_np` and `nanosleep` externs: `argc`, `argv(i)`, `env`, `exit_process`, `errno`, `monotonic_ns`, `realtime_ns`, `sleep_ns`, and `from_cstr` (copies) / `to_cstr` |
| `core/fs` | the `open` (two arguments, never `O_CREAT`), `creat`, `close`, `unlink` and `access` externs: `open_read`, `open_append`, `create`, `close_fd`, `remove`, `exists`, `write_all`, `read_some` and `read_all`, each failure as a negated errno |
| `core/mem` | the growable `gc_array` and the hash table behind Vec, Map and Set |
| `core/fmt` | the Darwin-private `__dtoa`/`__freedtoa` and the `strtod` externs: `shortest(f)` (the shortest round-trip digits, their decimal point and sign) and `parse(s)` |
| `core/text` | raw `str` parts: `slice`, and the `alloc`/`put`/`window`/`take` buffer that builds a str once |
| `core/mem_selftest`, `core/desc_selftest` | heap and descriptor self-checks |

## std

std is the public library. It wraps core in safe APIs: std never declares an `extern`
and never names a `.core` intrinsic.

| module | holds |
|--------|-------|
| `std/io` | `print`, `println`, `eprint` and `eprintln` over `T has Display`; `read_line` and `read_all` on stdin. It imports std/float, so a printing program can print a float |
| `std/float` | `impl float has Display` and `parse(s) -> Option[float]`, over core/fmt |
| `std/vec`, `std/map`, `std/set` | the collections, over core/mem |
| `std/str` | methods on `str`, including `parse_int`, over core/text |
| `std/option` | the closure methods of `Option` and `Result` |
| `std/ref` | `Ref.new`, `get` and `set` |
| `std/iter` | the `Iterator` protocol that `for`-in desugars to |
| `std/math` | integer and float math, over core/ffi |
| `std/gc` | `collect()` and `collections()` |
| `std/os` | `args()`, `env(name)` and `exit(code)`, over core/os |
| `std/time` | `Instant`, `Duration`, `now()`, `sleep(d)`, `unix()` and `unix_ns()`, over core/os |
| `std/fs` | `IoError`, `read_file`, `write_file`, `append_file`, `remove`, `exists`, and `File` handles from `open`, `create` and `append`, over core/fs |

## std/io: output and input

```toy
pub fn print[T has Display](x: T)      # to stdout
pub fn println[T has Display](x: T)    # to stdout, then "\n"
pub fn eprint[T has Display](x: T)     # to stderr
pub fn eprintln[T has Display](x: T)   # to stderr, then "\n"
pub fn read_line() -> Option[str]      # the next line of stdin, or none at end of input
pub fn read_all() -> str               # the rest of stdin
```

- Output is unbuffered (see the conventions below). `eprint` writes again after a
  short write, so a large message reaches stderr whole.
- `read_line` strips the line's `\n` or `\r\n`. An empty line is `some("")`, a last
  line with no final newline is returned as it is, and at end of input every call is
  `none`.
- stdin is unbuffered too: `read_line` reads one byte at a time and never takes bytes
  past the newline, so `read_line` and `read_all` can follow each other freely. A
  buffered line reader is later work.
- A read error ends the input as end of input does.

```toy
import std/io
import std/str

fn main() -> int {
    total := 0
    done := false
    while !done {
        match io.read_line() {
            .some(l) -> match l.parse_int() {
                .some(n) -> { total += n },
                .none -> io.eprintln("not a number: ".concat(l)),
            },
            .none -> { done = true },
        }
    }
    io.println(total)
    return 0
}
```

## std/str: parse_int

`s.parse_int() -> Option[int]` reads a decimal `int` with an optional `+` or `-` sign.
It is `none` for an empty string, a lone sign, any other byte (spaces and `_`
included) and a value outside `int`'s range. Call `trim` first to accept surrounding
whitespace.

## std/float: printing and parsing floats

A float prints the shortest digits that read back as the same value:

- plain for a magnitude in [1e-4, 1e16), and as `<d>.<ddd>e<exp>` outside it;
- always with a digit after the point, so `1.0`, `100.0`, `1.0e16`, `1.0e-5`;
- `inf`, `-inf` and `NaN`; a negative zero is `-0.0`.

A float has no unary minus yet, so a printed negative such as `-2.5` is not toy source.
Struct, enum and `Option` fields of type float print the same way, and `x.to_string()`
works on a float. std/io imports std/float, so printing needs no extra import; a
`[T has Display]` fn of your own without either import gets T0023, which names
std/float.

`float.parse(s) -> Option[float]` reads `s` as C `strtod` does and is `none` unless it
reads all of `s`: an empty string, leading whitespace or anything left over is `none`.
`parse(x.to_string()) == x` for every float but NaN. `inf`, `nan` and an overflowing
`1e999` (inf) parse too.

## std/os: arguments, environment and exit

- `os.args() -> Vec[str]` holds every program argument, the program's path (argv[0])
  first. Each call builds a fresh Vec, so changing one changes no other.
- `os.env(name) -> Option[str]` is the variable's value, `some("")` for one set to the
  empty string, and `none` for an unset one.
- `os.exit(code: int) -> never` ends the process with `code`. Output written before it
  is not lost, since stdout and stderr are unbuffered. It is typed `never`, so it can end
  a value arm.
- Every str these return is a copy of the C string, never a pointer into C memory.
- `toy run f.toy -- a b` runs the program with `a` and `b` as its arguments. toy reads no
  word after the `--`, so `toy run f.toy -- -O1` hands `-O1` to the program. A second word
  without the `--` is a usage error.

```toy
import std/io
import std/os
import std/vec

fn main() {
    xs := os.args()
    if xs.len() < 2 {
        io.eprintln("usage: greet NAME")
        os.exit(2)
    }
    greeting := match os.env("GREETING") {
        .some(g) -> g,
        .none -> "hello",
    }
    io.println(greeting.concat(", ").concat(xs[1]))
}
```

## std/time: clocks and sleep

```toy
pub fn now() -> Instant           # a monotonic reading
pub fn ns(n: int) -> Duration     # also us(n), ms(n) and secs(n)
pub fn sleep(d: Duration)         # at least d; a zero or negative d returns at once
pub fn unix() -> int              # wall-clock seconds since 1970
pub fn unix_ns() -> int           # wall-clock nanoseconds since 1970
```

- An `Instant` comes from the monotonic clock: it never goes back, and it keeps counting
  while the machine sleeps. `t.elapsed()` is the `Duration` since `t`, and
  `t.since(earlier)` the one between two readings. Instants compare with `<` and `==`.
- A `Duration` holds nanoseconds. It is built through `ns`, `us`, `ms` or `secs`, since
  another module cannot write its struct literal (#103). `d.ns()`, `d.us()`, `d.ms()` and
  `d.secs()` read it back, truncated toward zero. Durations add, subtract and compare.
- `unix()` and `unix_ns()` read the wall clock, which jumps when the system clock is set,
  so measure with `now()` instead.
- `sleep` sleeps again for the remainder when a signal cuts it short.

```toy
import std/io
import std/time

fn main() {
    start := time.now()
    time.sleep(time.ms(20))
    if start.elapsed() >= time.ms(20) { io.println("slept") }
    io.println((time.secs(1) + time.ms(500)).ms())   # 1500
}
```

## std/fs: whole files

```toy
pub fn read_file(path: str) -> Result[str, IoError]
pub fn write_file(path: str, s: str) -> Result[(), IoError]    # creates or replaces
pub fn append_file(path: str, s: str) -> Result[(), IoError]   # creates when missing
pub fn remove(path: str) -> Result[(), IoError]
pub fn exists(path: str) -> bool
```

- A relative path resolves against the process's working directory.
- `IoError` names why a call failed, without the path (the caller has it):
  `not_found`, `permission_denied`, `already_exists`, `is_directory`, `not_directory`,
  `invalid_path` and `other(errno)`. Its Display is a short phrase such as `not found`
  or `os error 28`.
- A path holding a NUL byte is `invalid_path`, and makes no call: as a C string it would
  name a shorter path. `exists` is false for one.
- Reading a directory is `is_directory`, and a path through a file is `not_directory`.
- A new file gets mode 0644 before the umask. Directories are not covered yet.

```toy
import std/fs
import std/io

fn main() {
    match fs.write_file("notes.txt", "one\n") {
        .ok(_) -> {},
        .err(e) -> io.eprintln("write failed: ".concat(e.to_string())),
    }
    match fs.append_file("notes.txt", "two\n") {
        .ok(_) -> {},
        .err(e) -> io.eprintln(e),
    }
    match fs.read_file("notes.txt") {
        .ok(s) -> io.print(s),
        .err(e) -> io.eprintln(e),
    }
}
```

## std/fs: file handles

```toy
pub fn open(path: str) -> Result[File, IoError]      # for reading
pub fn create(path: str) -> Result[File, IoError]    # for writing; creates or empties
pub fn append(path: str) -> Result[File, IoError]    # for writing at the end; creates

impl File {
    fn read(self, n: int) -> Result[str, IoError]    # at most n bytes; "" at the end
    fn read_all(self) -> Result[str, IoError]        # the rest of the file
    fn write(self, s: str) -> Result[(), IoError]    # all of s
    fn close(self) -> Result[(), IoError]
}
```

- Close is explicit: nothing closes a File for you, so a File you drop keeps its fd open.
- After `close`, every op on that File, a second `close` included, is `other(9)` (EBADF)
  and makes no call. The next open reuses the closed fd number, so passing it on would
  reach some other file. Copies of a File share its state, so a close through one copy
  (or through a helper that took it by value) closes them all.
- `read(n)` returns at most 64 KiB per call; a negative `n` panics.

```toy
import std/fs
import std/io

fn main() {
    f := match fs.open("notes.txt") {
        .ok(f) -> f,
        .err(e) -> {
            io.eprintln(e)
            return
        },
    }
    loop {
        chunk := match f.read(4096) {
            .ok(s) -> s,
            .err(e) -> {
                io.eprintln(e)
                break
            },
        }
        if chunk == "" { break }
        io.print(chunk)
    }
    match f.close() {
        .ok(_) -> {},
        .err(e) -> io.eprintln(e),
    }
}
```

## Conventions

Every std and core module follows these rules.

- **Environment failures are an Option or a Result, never a panic.** A missing file, an
  unset variable, end of input or a bad number is a value the caller matches on. A panic
  is for a broken invariant in the program itself, such as an index out of range.
- **No hidden global state.** A std function's result depends only on its arguments and
  the outside world it names, such as a file or the clock. State a program keeps, like a
  random number generator, is a value the program holds and passes.
- **stdout and stderr are unbuffered.** Each print is one or more `write` calls before it
  returns, so output interleaves in program order, survives a panic or an exit, and needs
  no flush.
- **Each Darwin-only symbol sits behind one core wrapper.** A private or
  platform-specific symbol (for example `_NSGetArgv`, `__error`) is declared once in core
  and reached only through that one wrapper fn, so porting to another target changes the
  wrapper bodies and nothing in std.
- **core is private (R0013).** A user module that imports `core/*` gets R0013, which
  names the std module to import instead. Each core module has a row in
  `StdNames.core_replacements`, and a unit test walks the bundle so a new core module
  cannot ship without one. The self-check modules map to `std/gc`. The dev-only
  `--dev-core-imports` flag lifts the rule.
- **Tests force a collection through `std/gc`.** No test or fixture imports core: a test
  that must observe a collection calls `gc.collect()` and reads `gc.collections()`, which
  also counts collections an allocation triggered.
- **Only what a program calls is linked.** The linker drops every fn unreachable from the
  entry, so a std addition costs nothing in a program that does not call it.
