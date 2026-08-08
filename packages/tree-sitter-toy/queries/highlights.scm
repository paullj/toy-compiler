; Syntax highlighting for the toy language (nvim-treesitter default capture names).
; These captures add editor-only richness (type vs function vs variable roles); the
; compiler-agreement test folds them all back onto its coarser token taxonomy.

; ---- Literals ----
(number) @number
(float) @number.float
(string) @string
(char) @character
(boolean) @boolean
(comment) @comment

; ---- Keywords ----
[
  "fn"
  "struct"
  "enum"
  "impl"
  "protocol"
  "type"
  "import"
  "extern"
  "unsafe"
] @keyword

[
  "pub"
  "mut"
  "has"
  "as"
] @keyword.modifier

[
  "if"
  "else"
  "match"
] @keyword.conditional

[
  "while"
  "loop"
  "for"
  "in"
  "break"
  "continue"
] @keyword.repeat

"return" @keyword.return

; ---- Operators ----
[
  "+" "-" "*" "/" "%"
  "==" "!=" "<" "<=" ">" ">="
  "&&" "||" "|" "&" "^" "~" "<<" ">>"
  "+." "-." "*." "/." "<." ">." "<=." ">=."
  "=" ":=" "->" "?" ".."
] @operator

; ---- Punctuation ----
["(" ")" "[" "]" "{" "}"] @punctuation.bracket
["," ":" "." "@"] @punctuation.delimiter

; ---- Identifier roles ----
((identifier) @type.builtin
  (#any-of? @type.builtin "int" "bool" "str" "char" "byte" "rawptr"
     "int8" "int16" "int32" "int64" "uint8" "uint16" "uint32" "uint64"))

((identifier) @variable.builtin
  (#any-of? @variable.builtin "self" "Self"))

(type_name (identifier) @type)
(type_parameter (identifier) @type)

(struct_declaration name: (identifier) @type)
(enum_declaration name: (identifier) @type)
(protocol_declaration name: (identifier) @type)
(type_alias name: (identifier) @type)

(function_declaration name: (identifier) @function)
(call_expression function: (identifier) @function.call)
(call_expression function: (field_expression field: (identifier) @function.call))

(label (identifier) @label)
(variant_expression (identifier) @constant)
(variant_pattern (identifier) @constant)

; Fallback: a plain identifier is a variable.
(identifier) @variable
