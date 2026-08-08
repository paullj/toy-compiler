# tree-sitter-toy

A [Tree-sitter](https://tree-sitter.github.io/) grammar for the toy language, used for
editor syntax highlighting and for a compiler-agreement test.

## What is here

| Path | Role |
|------|------|
| `grammar.js` | The grammar source. Mirrors the compiler lexer (`packages/compiler/src/lex.zig`) at token granularity. |
| `src/scanner.c` | External scanner: a zero-width `_newline` terminator reproducing the compiler's Go-style statement-terminator insertion. |
| `src/parser.c`, `src/grammar.json`, `src/node-types.json` | **Generated and checked in.** The build compiles `src/parser.c` directly, so the tree-sitter CLI is NOT needed to build — only to regenerate. |
| `src/tree_sitter/*.h` | CLI-emitted runtime headers, also checked in. |
| `queries/highlights.scm` | Neovim (nvim-treesitter) highlight query. |
| `tree-sitter.json` | Grammar metadata; pins ABI 15. |

## Regenerating

Only needed after editing `grammar.js`. Requires the tree-sitter CLI (0.26.x, matching the
version that produced the checked-in `parser.c`):

```
mise run ts:generate      # == (cd packages/tree-sitter-toy && tree-sitter generate)
```

The regenerated `src/` must be committed; the checked-in `parser.c` is authoritative.

## Design notes

- The grammar is deliberately **parse-permissive**: a superset of what the compiler parser
  accepts. It only needs to tokenize the same way and parse every valid program without
  ERROR nodes. Semantically-illegal-but-syntactically-fine constructs (e.g. a value index
  on a non-`Map`) are accepted here and rejected by a later compiler stage.
- Newlines are trivia; statement/arm boundaries come from the external `_newline` scanner,
  which is excluded from the token-agreement comparison. A differently-grouped but
  ERROR-free parse still agrees on the leaf token stream.
- `self`, `Self`, `int`, `bool`, `str`, `rawptr`, `_` are plain identifiers, never keywords
  — exactly as the compiler lexer treats them.

## Neovim install (manual)

Neovim highlighting is validated by hand; the automated gate is the Zig agreement test plus
the zero-ERROR corpus parse.

```lua
-- 1. Register the parser (nvim-treesitter):
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()
parser_config.toy = {
  install_info = {
    url = "/path/to/toy-compiler/packages/tree-sitter-toy",
    files = { "src/parser.c", "src/scanner.c" },
  },
  filetype = "toy",
}

-- 2. Associate the extension:
vim.filetype.add({ extension = { toy = "toy" } })
```

Then `:TSInstall toy` and copy `queries/highlights.scm` to your nvim-treesitter
`queries/toy/highlights.scm`.
