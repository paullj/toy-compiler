//! The stdlib public symbols the front/middle-end reaches for by name — the single
//! source of truth for those strings. Surface sugar (for-in, list literals, indexing)
//! desugars against these methods/types, and the checker, discovery, and lowerer must
//! agree on the exact spelling. A rename in the corresponding std/*.toy otherwise
//! silently breaks that sugar with no compiler error; centralizing the strings makes
//! the rename a one-line edit here.

const std = @import("std");
const Ast = @import("../ast/Ast.zig");
const Callee = @import("Callee.zig").Callee;

/// The list/vector struct (`std/vec.toy`) list-literal and index sugar target.
pub const vec_struct = "Vec";

/// The hashed-container struct (`std/map.toy`): the index gate rejects `m[k]` by this
/// name, and the `for k, v` desugar reads `Entry`'s two fields to bind key/val.
pub const map_struct = "Map";
pub const entry_struct = "Entry";

/// The protocol a `for-in` receiver's `iter()` must yield (`std/iter.toy`), matched by
/// name against `model.protocols`.
pub const iter_protocol = "Iterator";

/// Associated / instance method names the surface sugar desugars to.
pub const method_new = "new";
pub const method_push = "push";
pub const method_iter = "iter";
pub const method_next = "next";

/// The `core/mem` element primitives an index desugars to: `xs[i]` reads through `at`,
/// `xs[i] = v` writes through `set_at`. Discovery, the fingerprint fold and lower all name
/// the primitive through this one table, so the instance discovered is the one called.
pub const ElemPrim = enum {
    at,
    set_at,

    pub fn name(p: ElemPrim) []const u8 {
        return switch (p) {
            .at => "core/mem.ga_at",
            .set_at => "core/mem.ga_set_at",
        };
    }

    /// The primitives an `.index` node calls, in the order lower emits them, from the
    /// mark the checker left in its callee slot: a read calls `at`, a plain store `set_at`,
    /// and an element read and written back both.
    pub fn of(mark: Callee) []const ElemPrim {
        return switch (mark.tag) {
            .elem_store => &.{.set_at},
            .elem_write => &.{ .at, .set_at },
            else => &.{.at},
        };
    }
};

/// Where a receiver-method desugar reads the receiver type it selects the method on.
pub const RecvSource = enum {
    /// The sugar node's own type (`node_types[i]`).
    node,
    /// The node's `rhs` child type (`node_types[n.rhs]`).
    rhs,
};

/// One receiver method a surface sugar expands to.
pub const DesugarMethod = struct { recv: RecvSource, name: []const u8 };

/// The receiver methods each surface sugar expands to, in expansion order — the single
/// enumeration mono discovery (`scanCalls`) and the lowerer both read, so the set they
/// discover and the set they emit provably agree (a method added here reaches both, and
/// lower's positional reads assert the length so a divergence is a compile error, not a
/// miscompile). Sugars that desugar to a free primitive rather than a receiver method
/// (`xs[i]` -> `ElemPrim`) are not receiver-method desugars and are not listed here.
pub fn desugarMethods(tag: Ast.Node.Tag) []const DesugarMethod {
    return switch (tag) {
        .empty_list => &.{.{ .recv = .node, .name = method_new }},
        .list_literal => &.{
            .{ .recv = .node, .name = method_new },
            .{ .recv = .node, .name = method_push },
        },
        .for_in_stmt => &.{
            .{ .recv = .rhs, .name = method_iter },
            .{ .recv = .node, .name = method_next },
        },
        // `for k, v in m` desugars identically to `for x in m`: `iter()` on the iterable
        // (`.rhs`), `next()` on the resulting iterator type (`.node` = node_types[stmt],
        // set by checkForIn2). The two bindings are a lower-only concern (the Entry field
        // split), so the discovered method set is the same as the single-binding form.
        .for_in2_stmt => &.{
            .{ .recv = .rhs, .name = method_iter },
            .{ .recv = .node, .name = method_next },
        },
        else => &.{},
    };
}

/// A missing-import hint for a stdlib symbol named without its `import`: the
/// `std/<module>` path that brings it into scope, and — when the typed name is not
/// itself the export — the spelling to use instead. The print family is ordinary
/// library code under `io` now, not a global, so `print` must become `io.print`.
pub const ImportHint = struct {
    module: []const u8,
    spelling: ?[]const u8 = null,
};

/// The tail of every missing-import diagnostic, formatted with the module name.
pub const add_import = "add 'import std/{s}'";

pub const ref_methods = [_][]const u8{ "new", "get", "set" };
pub const option_methods = [_][]const u8{ "map", "then", "or_else" };
pub const result_methods = [_][]const u8{ "map", "then", "map_err", "or_else" };
pub const str_methods = [_][]const u8{ "contains", "starts_with", "ends_with", "trim", "split", "join", "replace", "chars", "parse_int" };

/// A prelude type some of whose methods live in a std module the program imports: `Ref`,
/// `Option`, `Result` and `str`.
pub const MethodFamily = enum { ref, option, result, str };

/// The methods `std/<module>` declares on a family's type. A call of one of them that finds
/// no such method means the module is not in the program, so the checker names it.
pub const StdMethods = struct { family: MethodFamily, module: []const u8, methods: []const []const u8 };

pub const std_methods = [_]StdMethods{
    .{ .family = .ref, .module = "ref", .methods = &ref_methods },
    .{ .family = .option, .module = "option", .methods = &option_methods },
    .{ .family = .result, .module = "option", .methods = &result_methods },
    .{ .family = .str, .module = "str", .methods = &str_methods },
};

/// The std module that declares `member` on `family`'s type, or null.
pub fn methodModule(family: MethodFamily, member: []const u8) ?[]const u8 {
    for (std_methods) |e| {
        if (e.family != family) continue;
        for (e.methods) |m| if (std.mem.eql(u8, m, member)) return e.module;
    }
    return null;
}

/// The std module that declares a family's methods (`Ref`'s, say), whatever the member.
pub fn familyModule(family: MethodFamily) []const u8 {
    for (std_methods) |e| if (e.family == family) return e.module;
    unreachable;
}

/// Stdlib names a program is likely to reach for WITHOUT the import that binds them,
/// keyed by EXACT spelling: a typo (`Vecc`) is not a key, so the resolver's fuzzy
/// "did you mean" keeps ownership of typos and only a genuine stdlib name yields the
/// deterministic import hint. `Option`/`Result` are prelude generics (no import) and
/// are absent; `str`'s methods live in `std/str` and there is no `string` type, so
/// `string`/`String` are absent too. `Ref` is a prelude type whose methods live in
/// `std/ref`, so it names that module.
const import_hints = std.StaticStringMap(ImportHint).initComptime(.{
    .{ "Vec", ImportHint{ .module = "vec" } },
    .{ "Ref", ImportHint{ .module = "ref" } },
    .{ "Map", ImportHint{ .module = "map" } },
    .{ "Set", ImportHint{ .module = "set" } },
    .{ "io", ImportHint{ .module = "io" } },
    .{ "math", ImportHint{ .module = "math" } },
    .{ "iter", ImportHint{ .module = "iter" } },
    .{ "gc", ImportHint{ .module = "gc" } },
    .{ "os", ImportHint{ .module = "os" } },
    .{ "time", ImportHint{ .module = "time" } },
    .{ "vec", ImportHint{ .module = "vec" } },
    .{ "map", ImportHint{ .module = "map" } },
    .{ "set", ImportHint{ .module = "set" } },
    .{ "print", ImportHint{ .module = "io", .spelling = "io.print" } },
    .{ "println", ImportHint{ .module = "io", .spelling = "io.println" } },
    .{ "console", ImportHint{ .module = "io", .spelling = "io.print" } },
});

/// The `std/` import (and, when the typed name is wrong, the correct spelling) for an
/// unimported stdlib symbol, or null when `name` is not a known stdlib name — leaving
/// typo/near-miss handling to the caller.
pub fn importHintFor(name: []const u8) ?ImportHint {
    return import_hints.get(name);
}

/// The std module(s) a user program imports instead of a private `core/` module, named
/// by R0013. A `core/` module joins this table when it is added, so the R0013 hint never
/// goes blank (a unit test walks the bundle).
const core_replacements = std.StaticStringMap([]const u8).initComptime(.{
    .{ "core/mem", "std/vec, std/map or std/set" },
    .{ "core/text", "std/str" },
    .{ "core/sys", "std/io" },
    .{ "core/ffi", "std/math" },
    .{ "core/os", "std/os or std/time" },
});

/// The self-check modules only force or count collections for their tests, which std/gc
/// does through a supported surface.
const selftest_suffix = "_selftest";
const selftest_replacement = "std/gc";

/// The std replacement R0013 names for the `core/` module `path`, or null for a path
/// that is not a bundled core module.
pub fn coreReplacement(path: []const u8) ?[]const u8 {
    if (core_replacements.get(path)) |r| return r;
    if (std.mem.startsWith(u8, path, "core/") and std.mem.endsWith(u8, path, selftest_suffix)) return selftest_replacement;
    return null;
}

test "method names are distinct, non-empty, and the element primitives are qualified" {
    const methods = [_][]const u8{ method_new, method_push, method_iter, method_next };
    for (methods) |m| try std.testing.expect(m.len != 0);
    for (methods, 0..) |a, i| for (methods[i + 1 ..]) |b|
        try std.testing.expect(!std.mem.eql(u8, a, b));
    for (std.enums.values(ElemPrim)) |p| try std.testing.expect(std.mem.startsWith(u8, p.name(), "core/mem."));
    try std.testing.expect(vec_struct.len != 0 and iter_protocol.len != 0);
}

test "desugarMethods enumerates each sugar's receiver methods in expansion order" {
    const empty = desugarMethods(.empty_list);
    try std.testing.expectEqual(@as(usize, 1), empty.len);
    try std.testing.expect(std.mem.eql(u8, empty[0].name, method_new) and empty[0].recv == .node);

    const list = desugarMethods(.list_literal);
    try std.testing.expectEqual(@as(usize, 2), list.len);
    try std.testing.expect(std.mem.eql(u8, list[0].name, method_new) and list[0].recv == .node);
    try std.testing.expect(std.mem.eql(u8, list[1].name, method_push) and list[1].recv == .node);

    const fin = desugarMethods(.for_in_stmt);
    try std.testing.expectEqual(@as(usize, 2), fin.len);
    try std.testing.expect(std.mem.eql(u8, fin[0].name, method_iter) and fin[0].recv == .rhs);
    try std.testing.expect(std.mem.eql(u8, fin[1].name, method_next) and fin[1].recv == .node);

    const fin2 = desugarMethods(.for_in2_stmt);
    try std.testing.expectEqual(@as(usize, 2), fin2.len);
    try std.testing.expect(std.mem.eql(u8, fin2[0].name, method_iter) and fin2[0].recv == .rhs);
    try std.testing.expect(std.mem.eql(u8, fin2[1].name, method_next) and fin2[1].recv == .node);

    try std.testing.expect(map_struct.len != 0 and entry_struct.len != 0);

    // A sugar with no receiver-method desugar (indexing goes through `ElemPrim`).
    try std.testing.expectEqual(@as(usize, 0), desugarMethods(.index).len);
}

test "an index reads, a store writes, and a written-back element reads then writes" {
    try std.testing.expectEqualSlices(ElemPrim, &.{.at}, ElemPrim.of(.{}));
    try std.testing.expectEqualSlices(ElemPrim, &.{.set_at}, ElemPrim.of(Callee.elem_store));
    try std.testing.expectEqualSlices(ElemPrim, &.{ .at, .set_at }, ElemPrim.of(Callee.elem_write));
}

test "core/mem declares every element primitive" {
    const BundledStd = @import("bundled_std");
    const mem = for (BundledStd.modules) |m| {
        if (std.mem.eql(u8, m.path, "core/mem")) break m.source;
    } else return error.TestExpectedCoreMem;
    for (std.enums.values(ElemPrim)) |p| {
        const short = p.name()["core/mem.".len..];
        var buf: [64]u8 = undefined;
        const decl = try std.fmt.bufPrint(&buf, "pub fn {s}[T](", .{short});
        try std.testing.expect(std.mem.indexOf(u8, mem, decl) != null);
    }
}

test "no bundled module writes a desugared method call" {
    // Lower resolves a desugar's methods at the program's site (`Typecheck.findMethod`),
    // so one in std would reach a program's shadowing `push` or `next`; a call written
    // out in std reaches std's own.
    const BundledStd = @import("bundled_std");
    const Lexer = @import("../lex.zig");
    const Parser = @import("../parse.zig");
    const gpa = std.testing.allocator;
    for (BundledStd.modules) |m| {
        const tokens = try Lexer.tokenize(gpa, m.source);
        defer gpa.free(tokens);
        const tree = try Parser.expectTree(gpa, tokens, m.source);
        defer {
            gpa.free(tree.nodes);
            gpa.free(tree.extra);
            if (tree.pub_bits.len != 0) gpa.free(@constCast(tree.pub_bits));
        }
        for (tree.nodes) |n| if (desugarMethods(n.tag).len != 0) {
            std.debug.print("{s} desugars a '{s}'\n", .{ m.path, @tagName(n.tag) });
            return error.TestUnexpectedResult;
        };
    }
}

test "importHintFor: exact stdlib names hint; typos, prelude generics, and unknowns do not" {
    const vec = importHintFor("Vec").?;
    try std.testing.expectEqualStrings("vec", vec.module);
    try std.testing.expect(vec.spelling == null);

    try std.testing.expectEqualStrings("io", importHintFor("io").?.module);
    try std.testing.expect(importHintFor("io").?.spelling == null);

    const print = importHintFor("print").?;
    try std.testing.expectEqualStrings("io", print.module);
    try std.testing.expectEqualStrings("io.print", print.spelling.?);
    try std.testing.expectEqualStrings("io.println", importHintFor("println").?.spelling.?);
    try std.testing.expectEqualStrings("io.print", importHintFor("console").?.spelling.?);

    inline for (.{ "Map", "Set", "map", "set", "vec", "math", "iter", "os", "time" }) |n|
        try std.testing.expect(importHintFor(n) != null);
    try std.testing.expectEqualStrings("gc", importHintFor("gc").?.module);
    try std.testing.expectEqualStrings("ref", importHintFor("Ref").?.module);
    try std.testing.expect(importHintFor("Ref").?.spelling == null);

    // A typo is NOT a key: near-miss keeps ownership of typos.
    try std.testing.expect(importHintFor("Vecc") == null);
    try std.testing.expect(importHintFor("prnt") == null);
    // Prelude generics never hint (no import needed); there is no std/string yet.
    try std.testing.expect(importHintFor("Option") == null);
    try std.testing.expect(importHintFor("Result") == null);
    try std.testing.expect(importHintFor("String") == null);
    try std.testing.expect(importHintFor("string") == null);
    try std.testing.expect(importHintFor("") == null);
}

test "std_methods are the methods each std module declares, and all of them" {
    const BundledStd = @import("bundled_std");
    // Methods of the module's own types, which no family lists: `Chars.iter`, `Chars.next`.
    const others = [_]struct { module: []const u8, n: usize }{.{ .module = "str", .n = 2 }};
    var buf: [64]u8 = undefined;
    var bracket: [64]u8 = undefined;
    for (std_methods) |e| {
        const path = try std.fmt.bufPrint(&buf, "std/{s}", .{e.module});
        const src = for (BundledStd.modules) |m| {
            if (std.mem.eql(u8, m.path, path)) break m.source;
        } else return error.NoStdModule;
        var want: usize = 0;
        for (std_methods) |f| if (std.mem.eql(u8, f.module, e.module)) {
            want += f.methods.len;
        };
        for (others) |o| if (std.mem.eql(u8, o.module, e.module)) {
            want += o.n;
        };
        for (e.methods) |name| {
            const plain = try std.fmt.bufPrint(&buf, "    fn {s}(", .{name});
            const generic = try std.fmt.bufPrint(&bracket, "    fn {s}[", .{name});
            try std.testing.expect(std.mem.indexOf(u8, src, plain) != null or std.mem.indexOf(u8, src, generic) != null);
        }
        try std.testing.expectEqual(want, std.mem.count(u8, src, "    fn "));
    }
    try std.testing.expectEqualStrings(importHintFor("Ref").?.module, familyModule(.ref));
    try std.testing.expectEqualStrings("option", methodModule(.result, "map_err").?);
    try std.testing.expect(methodModule(.option, "map_err") == null);
}

test "every bundled core module names a std replacement for R0013" {
    const BundledStd = @import("bundled_std");
    for (BundledStd.modules) |m| {
        if (!std.mem.startsWith(u8, m.path, "core/")) continue;
        const r = coreReplacement(m.path) orelse {
            std.debug.print("core module '{s}' has no std replacement in core_replacements\n", .{m.path});
            return error.TestUnexpectedResult;
        };
        try std.testing.expect(std.mem.startsWith(u8, r, "std/"));
    }
    for (core_replacements.keys()) |path| {
        const bundled = for (BundledStd.modules) |m| {
            if (std.mem.eql(u8, m.path, path)) break true;
        } else false;
        if (!bundled) {
            std.debug.print("core_replacements names '{s}', which the bundle lacks\n", .{path});
            return error.TestUnexpectedResult;
        }
    }
    try std.testing.expectEqual(@as(?[]const u8, null), coreReplacement("std/vec"));
    try std.testing.expectEqual(@as(?[]const u8, null), coreReplacement("util_selftest"));
}
