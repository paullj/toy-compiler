# Pipe operator `|>`

Status: **spec, not implemented**. Decisions were locked in a /grill-me pass on 2026-09-28.

## Summary

`|>` passes the value on its left as the **first argument** of the call on its right.
It is pure syntax. The parser rewrites it into an ordinary `.call` node, so the resolver,
the type checker, lowering and codegen do not see a pipe.

```toy
import std/io
import std/math

fn double(x: int) -> int { x * 2 }
fn add(a: int, b: int) -> int { a + b }

fn main() -> int {
    -7 |> math.abs |> double |> add(1) |> io.println    # prints 15
    0
}
```

## Desugaring

| Source | Result |
|--------|--------|
| `x \|> f` | `f(x)` |
| `x \|> f(a, b)` | `f(x, a, b)` |
| `x \|> m.f(a)` | `m.f(x, a)` (module-qualified fn) |
| `x \|> v.push` | `v.push(x)` (method on another value) |
| `x \|> T.ctor` | `T.ctor(x)` (e.g. `5 \|> Option.some`) |
| `x \|> f[T]` | `f[T](x)` |
| `x \|> f[T](a)` | `f[T](x, a)` |
| `x \|> T[A].ctor` | `T[A].ctor(x)` (e.g. `5 \|> Option[int].some`) |
| `x \|> f?` | `f(x)?` |
| `x \|> f(a)?` | `f(x, a)?` |
| `x \|> f \|> g(a)` | `g(f(x), a)` |

The callee path is not classified at parse time. `m.f` and `v.push` have the same
syntax, and the resolver decides later what each one is. So a pipe into a method on
**another** value works with no special code.

## Grammar

```
pipe_expr  := expr "|>" pipe_rhs
pipe_rhs   := path call_args? "?"?
path       := identifier ("." identifier | turbofish)*    # at most one turbofish
turbofish  := "[" type ("," type)* "]"
call_args  := "(" (expr ("," expr)*)? ")"
```

- **Precedence:** lowest of all binary operators (below `||`). `a + b |> f` is `f(a + b)`.
- **Associativity:** left. `x |> f |> g` is `(x |> f) |> g`.
- In `pipe_rhs`, a `name[...]` is **always** a turbofish, never an index. The parser does
  not use the `turbofishFollows` heuristic there.
- The RHS stops after the optional `?`. The next token must be another `|>`, or a token
  that no infix operator starts (terminator, `,`, `)`, `]`, `}`, `{`, EOF). Anything else
  is **P0014**:
  - a postfix (`.field`, `[i]`, `(...)` on the call result);
  - a binary operator (`x |> f == y`, `x |> f + 1`).
  Without this rule, precedence climbing would continue after the fixed-shape RHS and read
  `x |> f == y` as `f(x) == y`. That result is not the same as "lowest precedence".
  Parentheses make the intent explicit: `(x |> f) == y`.
- Binary operators on the **left** bind tighter: `y == x |> f` is `f(y == x)`.
- `..` is not an operator. It exists only in a `for` head, and each side is a full
  expression: `for i in 0..3 |> f { }` is `0..f(3)`.
- **Nesting cap:** each pipe nests its lhs one call deeper, so a pipe chain counts
  against the same depth cap (256) as written nested calls. A longer chain is **P0005**.

## Evaluation order

The pipe evaluates exactly like the call it desugars to. The piped value is argument 0,
so it is evaluated before the written arguments, which then go left to right. The
receiver of a method target (the `v` in `x |> v.push`) follows the normal method-call
rule. The order is visible only when the lhs has a side effect on that receiver.

## Multi-line pipes

Both styles are legal:

```toy
xs_total := 3
    |> double
    |> add(1)

ys_total := 3 |>
    double |>
    add(1)
```

- **Leading `|>`:** the lexer does not insert the ASI `.newline` terminator when the next
  non-trivia token is `|>`. Comments and blank lines between the lines are trivia.
- **Trailing `|>`:** this needs no change. `|>` cannot end a statement, so no terminator
  is inserted after it.

## Diagnostics

### P0014 — invalid pipe target (new)

This error occurs when the RHS does not match `pipe_rhs`, or when a pipe has no value on
its left:

| Source | Why | Message |
|--------|-----|---------|
| `x \|> 5` | literal, not a function | `pipe target must be a function or a call` |
| `x \|> (f)` | parenthesized callee | same |
| `x \|>` then newline / `}` / `)` / `]` / `,` / `\|>` / a declaration / EOF | no target (the caret is on the `\|>`) | same |
| `x \|> f().len()` | postfix after the RHS | same, plus `; wrap the pipe in parentheses to use its result` |
| `x \|> f == y` | binary operator after the RHS | same as the row above |
| `y := \|> f`, `return⏎\|> f(x)`, `break⏎\|> f` | no value on the left | `a pipe needs a value on its left` |
| `if c { .. }⏎\|> f`, `continue⏎\|> f` | a statement form is not a value | `only a value can be piped, and this statement has none` |

The last two rows come from the multi-line rule. `}`, `return`, `break` and `continue` end a
statement, so the lexer joins a following leading-`|>` line onto them. A block **expression**
(`x := if c { 1 } else { 2 }⏎|> f`, or a statement-level `match`) is a value and pipes normally.

Recovery, with one diagnostic per pipe:

- **Bad or missing target:** the parser reads the bad operand (if any) and puts an
  `.error_node` in place of the call. A missing target consumes nothing, so a run of
  `|> |> |>` is handled one pipe at a time by the infix loop, not by recursion.
- **Postfix or operator after the RHS:** the parser keeps the call and continues as if the
  user had written the parentheses. The cascade latch suppresses follow-on errors in the
  same statement.
- **Statement form on the left:** the statement is kept, and recovery skips to the next
  statement.

The code has a `toy explain` page (`P0014.md`).

### Piped-call attribution (T0039 / T0040)

A piped `.call` node stores the `|>` token as its `main_token`. A plain call stores `(`.
Synthetic calls already store `0`, so no consumer depends on `(`. This gives a "piped"
flag without an AST schema change. `Ast.isPipedCall` reads it.

- **T0039 arity mismatch:** the argument count includes the piped value. The message
  adds `(the piped value is argument 1)`. The caret starts at the `|>` of the failing
  step, not at the lhs, so it does not cover the whole chain before that step.
- **T0040 argument type mismatch:** for argument 1, the message adds `(the piped value)`
  and the caret covers the lhs expression.

```
error[T0039]: expected 2 argument(s), got 1 (the piped value is argument 1); 'add' takes (int, int)
error[T0040]: argument 1 (the piped value): expected int, got bool; 'neg' takes (int)
```

The uncoded count and type messages (generic fns, methods of generic impls, protocol-bound
methods, built-in methods, enum variant and tuple-struct constructors) get the same hints
and the same `|>`-first caret.

## Out of scope

| Item | Reason / where it goes |
|------|------------------------|
| `_` placeholder (`x \|> f(a, _)`) | Deferred to the closures roadmap. With closures it becomes a Gleam-style function capture. The first-arg rule stays the same, so no rework is necessary. |
| Leading-dot method form (`x \|> .m()`) | Rejected. Normal `.m()` chaining already does this. |
| UFCS / free-fn std twins | Deferred. Today pipes reach free fns only (`io.*`, `math.*`, user fns). |
| A `.pipe` AST node | Rejected. The future lossless CST keeps the pipe for the formatter and lints (see "parens wait for CST"). |

## Forward compatibility

When closures and function values arrive:

- `x |> g`, where `g` is a local of function type, still means `g(x)`.
- `x |> f(a, _)` becomes "apply the capture `fn(v) { f(a, v) }` to `x`", which is `f(a, x)`.

Both results are the same as a syntactic rewrite, so no program written against this spec
changes meaning.
