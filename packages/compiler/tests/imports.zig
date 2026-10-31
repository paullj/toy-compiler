//! Warm-cache soundness for selective imports and re-exports. A selection binds a local
//! name to another module's item, so changing what a name selects, or what a `pub import`
//! re-exports under it, must rebuild every fn that names it, even though the fn's own text
//! is unchanged.

const flip = @import("places.zig").flip;

test "warm cache: re-pointing a selected fn's alias rebuilds its caller" {
    const lib =
        \\pub fn one() -> int { 1 }
        \\pub fn two() -> int { 2 }
        \\
    ;
    try flip(
        ".toy-test-imports-fn",
        &.{ .{ .name = "main.toy", .src = "import lib.{one as f}\nfn main() -> int { f() }\n" }, .{ .name = "lib.toy", .src = lib } },
        1,
        &.{.{ .name = "main.toy", .src = "import lib.{two as f}\nfn main() -> int { f() }\n" }},
        2,
    );
}

test "warm cache: re-pointing a selected struct's alias rebuilds the fns that touch it" {
    const lib =
        \\pub struct A { pub x: int }
        \\pub struct B { pub pad: int, pub x: int }
        \\
    ;
    try flip(
        ".toy-test-imports-struct",
        &.{ .{ .name = "main.toy", .src = "import lib.{A as S}\nfn get(s: S) -> int { s.x }\nfn main() -> int { get(S { x: 3 }) }\n" }, .{ .name = "lib.toy", .src = lib } },
        3,
        &.{.{ .name = "main.toy", .src = "import lib.{B as S}\nfn get(s: S) -> int { s.x }\nfn main() -> int { get(S { pad: 9, x: 4 }) }\n" }},
        4,
    );
}

test "warm cache: a selected fn whose module changes its body serves the new body" {
    try flip(
        ".toy-test-imports-body",
        &.{ .{ .name = "main.toy", .src = "import lib.{val}\nfn main() -> int { val() }\n" }, .{ .name = "lib.toy", .src = "pub fn val() -> int { 5 }\n" } },
        5,
        &.{.{ .name = "lib.toy", .src = "pub fn val() -> int { 6 }\n" }},
        6,
    );
}

test "warm cache: re-pointing a re-export rebuilds the importer's caller" {
    const lib =
        \\pub fn one() -> int { 1 }
        \\pub fn two() -> int { 2 }
        \\
    ;
    try flip(
        ".toy-test-imports-reexport",
        &.{ .{ .name = "main.toy", .src = "import mid.{f}\nimport mid\nfn main() -> int { f() + mid.f() }\n" }, .{ .name = "mid.toy", .src = "pub import lib.{one as f}\n" }, .{ .name = "lib.toy", .src = lib } },
        2,
        &.{.{ .name = "mid.toy", .src = "pub import lib.{two as f}\n" }},
        4,
    );
}

test "warm cache: a re-exported struct whose layout changes rebuilds the fns that touch it" {
    try flip(
        ".toy-test-imports-reexport-struct",
        &.{ .{ .name = "main.toy", .src = "import mid.{S, mk}\nfn get(s: S) -> int { s.x }\nfn main() -> int { get(mk(3)) }\n" }, .{ .name = "mid.toy", .src = "pub import lib.{S, mk}\n" }, .{ .name = "lib.toy", .src = "pub struct S { pub x: int }\npub fn mk(x: int) -> S { S { x: x } }\n" } },
        3,
        &.{.{ .name = "lib.toy", .src = "pub struct S { pub pad: int, pub x: int }\npub fn mk(x: int) -> S { S { pad: 9, x: x + 1 } }\n" }},
        4,
    );
}
