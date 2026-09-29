/// Tree-sitter grammar for the toy language.
///
/// This grammar mirrors the compiler's hand-written lexer (packages/compiler/src/lex.zig
/// and the `Tag` enum in packages/compiler/src/ast/Token.zig). Its purpose is
/// token/highlight-level agreement with that lexer, not a faithful reproduction of the
/// compiler's parser: it is deliberately parse-*permissive* (a superset of what the toy
/// parser accepts) so that every language-features fixture parses with zero ERROR nodes,
/// which is all the agreement test requires. Where a construct is semantically illegal
/// (e.g. a value index on a non-Map) the toy compiler rejects it at a later stage; the
/// grammar still accepts it structurally.
///
/// Newlines are trivia here. The compiler lexer performs Go-style terminator insertion,
/// but because agreement is at token granularity (not AST shape), a differently-grouped
/// but ERROR-free parse still agrees on the leaf token stream.

const PREC = {
  // Lowest: `a + b |> f` pipes the whole `a + b` (packages/docs/pipe.md).
  pipe: 0,
  or: 1,
  and: 2,
  bit_or: 3,
  bit_xor: 4,
  bit_and: 5,
  equality: 6,
  comparison: 7,
  shift: 8,
  range: 9,
  additive: 10,
  multiplicative: 11,
  unary: 12,
  postfix: 13,
  struct_literal: 14,
};

function sepBy1(sep, rule) {
  return seq(rule, repeat(seq(sep, rule)));
}
function sepBy(sep, rule) {
  return optional(sepBy1(sep, rule));
}
// A comma-separated list with an optional trailing comma.
function commaList(rule) {
  return seq(rule, repeat(seq(",", rule)), optional(","));
}

module.exports = grammar({
  name: "toy",

  word: ($) => $.identifier,

  // `_newline` is a zero-width statement/arm terminator produced by the external scanner
  // (src/scanner.c), mirroring the compiler lexer's Go-style terminator insertion. It is
  // trivia-excluded from the agreement comparison.
  externals: ($) => [$._newline],

  extras: ($) => [/[ \t\r\n]/, $.comment],

  // GLR conflicts that cannot be resolved by precedence alone. The dominant one is
  // block-vs-struct-literal in a condition position (`if X { .. }`): `X { .. }` could be
  // a struct literal or `X` followed by the then-block. No valid fixture depends on the
  // choice (agreement is invariant to which branch wins — identical leaves either way).
  conflicts: ($) => [
    // `break @l <block>` — `@l` as break's label + a block value, vs `@l <block>` as a
    // single labeled-block value. Identical leaves either way.
    [$.labeled_expression, $._expression],
    // `X { .. }` in a scrutinee/condition position (`if X { .. }`, `match X { .. }`): the
    // `{` could open a struct-literal body or the following block/match body. Declaring
    // the conflict (rather than forcing one via precedence) lets GLR keep both stacks and
    // discard whichever hits an ERROR — so a genuine struct literal (`P{ x: 1 }`) and a
    // condition-then-block (`if a == b { .. }`) each resolve to their error-free parse.
    [$._expression, $.struct_literal],
  ],

  rules: {
    source_file: ($) => repeat($._declaration),

    _declaration: ($) =>
      choice(
        $.function_declaration,
        $.struct_declaration,
        $.enum_declaration,
        $.impl_block,
        $.protocol_declaration,
        $.type_alias,
        $.import_declaration,
      ),

    // ---- Declarations -----------------------------------------------------

    function_declaration: ($) =>
      seq(
        optional("pub"),
        optional("extern"),
        "fn",
        field("name", $.identifier),
        optional($.type_parameters),
        $.parameter_list,
        optional(seq("->", field("return_type", $._type))),
        optional($.block),
      ),

    type_parameters: ($) => seq("[", commaList($.type_parameter), "]"),

    type_parameter: ($) =>
      seq($.identifier, optional(seq("has", sepBy1("+", $._type)))),

    parameter_list: ($) => seq("(", optional(commaList($.parameter)), ")"),

    // Covers `self`, `mut self`, and `name: Type` uniformly. `self`/`Self` are plain
    // identifiers (never keywords), exactly as the compiler lexer treats them.
    parameter: ($) =>
      seq(optional("mut"), $.identifier, optional(seq(":", field("type", $._type)))),

    struct_declaration: ($) =>
      seq(
        optional("pub"),
        "struct",
        field("name", $.identifier),
        optional($.type_parameters),
        choice($.struct_field_block, $.tuple_field_list),
      ),

    struct_field_block: ($) =>
      seq("{", optional(commaList($.struct_field)), "}"),

    struct_field: ($) => seq($.identifier, ":", field("type", $._type)),

    tuple_field_list: ($) => seq("(", optional(commaList($._type)), ")"),

    enum_declaration: ($) =>
      seq(
        optional("pub"),
        "enum",
        field("name", $.identifier),
        optional($.type_parameters),
        "{",
        optional(commaList($.enum_variant)),
        "}",
      ),

    // A variant is nullary (`N`), tuple-payload (`C(int, int)`), or record-payload
    // (`Rect { w: int, h: int }`, reusing the struct field block).
    enum_variant: ($) =>
      seq($.identifier, optional(choice($.tuple_field_list, $.struct_field_block))),

    impl_block: ($) =>
      seq(
        "impl",
        field("type", $._type),
        optional(seq("has", field("protocol", $._type))),
        "{",
        repeat($.function_declaration),
        "}",
      ),

    protocol_declaration: ($) =>
      seq(
        optional("pub"),
        "protocol",
        field("name", $.identifier),
        optional($.type_parameters),
        "{",
        repeat($.function_declaration),
        "}",
      ),

    type_alias: ($) =>
      seq(
        optional("pub"),
        "type",
        field("name", $.identifier),
        optional($.type_parameters),
        "=",
        field("value", $._type),
      ),

    import_declaration: ($) =>
      seq("import", $.import_path, optional(seq("as", field("alias", $.identifier)))),

    import_path: ($) => sepBy1("/", $.identifier),

    // ---- Types ------------------------------------------------------------

    _type: ($) => choice($.unit_type, seq($.type_name, optional($.type_arguments))),

    unit_type: ($) => seq("(", ")"),

    type_name: ($) => sepBy1(".", $.identifier),

    type_arguments: ($) => seq("[", commaList($._type), "]"),

    // ---- Statements -------------------------------------------------------

    block: ($) => seq("{", repeat($._statement), "}"),

    // A label (`@name`) prefixing a block or loop form. Kept as one rule so a label
    // attaches in exactly one place, avoiding a spurious clash with each loop form.
    labeled_expression: ($) =>
      seq($.label, choice($.block, $.loop_expression, $.while_expression, $.for_expression)),

    // Each statement may be closed by a `_newline` terminator (the common case) or run up
    // to the enclosing `}`. The terminator is what stops a value-ending line from merging
    // with a following line that begins with a prefix/binary operator (`&x` ⏎ `*p = ..`).
    _statement: ($) =>
      seq(
        choice(
          $.let_declaration,
          $.assignment,
          $.return_statement,
          $.break_statement,
          $.continue_statement,
          $.expression_statement,
        ),
        optional($._newline),
      ),

    let_declaration: ($) =>
      choice(
        seq(field("name", $.identifier), ":=", field("value", $._expression)),
        seq(
          field("name", $.identifier),
          ":",
          field("type", $._type),
          "=",
          field("value", $._expression),
        ),
      ),

    assignment: ($) => seq(field("left", $._expression), "=", field("right", $._expression)),

    // `prec.right` makes a bare `return`/`break` greedily absorb a following expression
    // rather than treating it as the next statement. Newlines are trivia here, so the two
    // groupings share the same leaf stream — agreement is invariant to the choice.
    return_statement: ($) => prec.right(seq("return", optional($._expression))),

    break_statement: ($) => prec.right(seq("break", optional($.label), optional($._expression))),

    continue_statement: ($) => prec.right(seq("continue", optional($.label))),

    expression_statement: ($) => $._expression,

    label: ($) => seq("@", $.identifier),

    // ---- Expressions ------------------------------------------------------

    _expression: ($) =>
      choice(
        $.unary_expression,
        $.reference_expression,
        $.binary_expression,
        $.pipe_expression,
        $._primary_expression,
        $.if_expression,
        $.match_expression,
        $.loop_expression,
        $.while_expression,
        $.for_expression,
        $.labeled_expression,
        $.unsafe_expression,
        $.block,
      ),

    _primary_expression: ($) =>
      choice(
        $.identifier,
        $.number,
        $.float,
        $.string,
        $.char,
        $.boolean,
        $.unit_expression,
        $.variant_expression,
        $.parenthesized_expression,
        $.list_expression,
        $.call_expression,
        $.index_expression,
        $.field_expression,
        $.try_expression,
        $.struct_literal,
      ),

    unit_expression: ($) => seq("(", ")"),

    // A leading-`.` enum variant used as a value: `.none`, `.Empty`. A payload form like
    // `.some(x)` is a `call_expression` over this primary (`.some` applied to `(x)`).
    variant_expression: ($) => seq(".", $.identifier),

    parenthesized_expression: ($) => seq("(", $._expression, ")"),

    list_expression: ($) => seq("[", optional(commaList($._expression)), "]"),

    boolean: ($) => choice("true", "false"),

    unary_expression: ($) =>
      prec(PREC.unary, seq(choice("-", "!", "~", "*"), $._expression)),

    reference_expression: ($) => prec(PREC.unary, seq("&", $._expression)),

    binary_expression: ($) => {
      const table = [
        ["||", PREC.or],
        ["&&", PREC.and],
        ["|", PREC.bit_or],
        ["^", PREC.bit_xor],
        ["&", PREC.bit_and],
        ["==", PREC.equality],
        ["!=", PREC.equality],
        ["<", PREC.comparison],
        ["<=", PREC.comparison],
        [">", PREC.comparison],
        [">=", PREC.comparison],
        ["<.", PREC.comparison],
        [">.", PREC.comparison],
        ["<=.", PREC.comparison],
        [">=.", PREC.comparison],
        ["<<", PREC.shift],
        [">>", PREC.shift],
        ["..", PREC.range],
        ["+", PREC.additive],
        ["-", PREC.additive],
        ["+.", PREC.additive],
        ["-.", PREC.additive],
        ["*", PREC.multiplicative],
        ["/", PREC.multiplicative],
        ["%", PREC.multiplicative],
        ["*.", PREC.multiplicative],
        ["/.", PREC.multiplicative],
      ];
      return choice(
        ...table.map(([op, p]) =>
          prec.left(p, seq($._expression, op, $._expression)),
        ),
      );
    },

    // Parse-permissive like the rest of this grammar: the compiler restricts the RHS to
    // `path [(args)] [?]` (P0014), but agreement only needs an ERROR-free parse.
    pipe_expression: ($) =>
      prec.left(PREC.pipe, seq(field("value", $._expression), "|>", field("target", $._expression))),

    call_expression: ($) =>
      prec(PREC.postfix, seq(field("function", $._primary_expression), $.arguments)),

    arguments: ($) => seq("(", optional(commaList($._expression)), ")"),

    // A single unified bracket postfix: type-application (`Vec[int]`), explicit call
    // type-args / turbofish (`id[int]`, `v.into[int]`), and value indexing
    // (`counts["the"]`) share one production — the grammar need not tell them apart.
    index_expression: ($) =>
      prec(PREC.postfix, seq(field("value", $._primary_expression), "[", commaList($._expression), "]")),

    // Member access. The name after `.` is an identifier (field/method) or an integer
    // (`x.0` tuple index). A `float` is intentionally NOT accepted here so `x.0.0` and
    // `0..5` tokenize as the compiler lexer does (post-`.` position excludes floats).
    field_expression: ($) =>
      prec(PREC.postfix, seq(field("value", $._primary_expression), ".", field("field", choice($.identifier, $.number)))),

    try_expression: ($) => prec(PREC.postfix, seq($._primary_expression, "?")),

    // No forcing precedence: the `_primary_expression`/`struct_literal` conflict is left
    // for GLR to resolve by discarding the ERROR branch (see `conflicts`).
    struct_literal: ($) =>
      seq(field("type", $._primary_expression), $.field_initializer_list),

    field_initializer_list: ($) =>
      seq("{", optional(commaList($.field_initializer)), "}"),

    field_initializer: ($) => seq($.identifier, ":", $._expression),

    // ---- Control flow -----------------------------------------------------

    if_expression: ($) =>
      prec.right(
        seq(
          "if",
          field("condition", $._expression),
          field("consequence", $.block),
          optional(seq("else", field("alternative", choice($.block, $.if_expression)))),
        ),
      ),

    match_expression: ($) =>
      seq("match", field("value", $._expression), "{", repeat($.match_arm), "}"),

    match_arm: ($) =>
      seq(
        $._pattern,
        repeat(seq("|", $._pattern)),
        optional(seq("if", field("guard", $._expression))),
        "->",
        field("value", $._expression),
        optional(","),
        optional($._newline),
      ),

    loop_expression: ($) => seq("loop", $.block),

    while_expression: ($) =>
      seq("while", field("condition", $._expression), $.block),

    for_expression: ($) =>
      seq(
        "for",
        field("pattern", $.identifier),
        optional(seq(",", $.identifier)),
        "in",
        field("iterable", $._expression),
        $.block,
      ),

    unsafe_expression: ($) => seq("unsafe", $.block),

    // ---- Patterns ---------------------------------------------------------

    _pattern: ($) =>
      choice($.variant_pattern, $.path_pattern, $.literal_pattern, $.identifier),

    // A dot-leading variant pattern (`.some(n)`, `.N`, `.Rect { w, h }`). The payload is
    // a tuple of sub-patterns or a record destructure binding field names.
    variant_pattern: ($) =>
      seq(".", $.identifier, optional($._pattern_payload)),

    // A qualified variant pattern (`Color.greem`) — the compiler accepts the enum-qualified
    // form as well as the dot-leading one.
    path_pattern: ($) =>
      seq($.identifier, repeat1(seq(".", $.identifier)), optional($._pattern_payload)),

    _pattern_payload: ($) =>
      choice(
        seq("(", optional(commaList($._pattern)), ")"),
        seq("{", optional(commaList($.identifier)), "}"),
      ),

    literal_pattern: ($) =>
      choice($.number, $.float, $.string, $.char, $.boolean),

    // ---- Tokens (mirror lex.zig) ------------------------------------------

    identifier: ($) => /[A-Za-z_][A-Za-z0-9_]*/,

    number: ($) =>
      token(
        choice(
          /0[xX][0-9a-fA-F_]+/,
          /0[oO][0-7_]+/,
          /0[bB][01_]+/,
          /[0-9][0-9_]*/,
        ),
      ),

    float: ($) =>
      token(
        seq(
          /[0-9][0-9_]*/,
          choice(
            seq(".", /[0-9][0-9_]*/, optional(seq(/[eE]/, optional(/[+-]/), /[0-9][0-9_]*/))),
            seq(/[eE]/, optional(/[+-]/), /[0-9][0-9_]*/),
          ),
        ),
      ),

    // `\\[\s\S]` = a backslash followed by ANY byte (including a newline), mirroring
    // lexQuoted's `\`-skip-2 scan; the closing quote ends the literal. A raw newline is
    // excluded from the unescaped set so it ends the literal (the compiler treats such a
    // literal as unterminated — an error token — so it never appears in a clean fixture).
    string: ($) => token(/"(\\[\s\S]|[^"\\\n])*"/),

    char: ($) => token(/'(\\[\s\S]|[^'\\\n])*'/),

    comment: ($) => token(seq("#", /[^\n]*/)),
  },
});
