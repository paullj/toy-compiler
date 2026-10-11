//! textDocument/definition: from a USE of a symbol, the source range of its DECLARATION
//! (`Analysis.declOf`). UNRELATED resolve and type errors are tolerated (a valid symbol
//! still navigates in a partially-broken buffer); only a missing tree (`graph.err`) leaves
//! nothing to navigate.
//!
//! The target uri is EITHER the open document's (a within-file target) OR an imported
//! module's `file://` path (a cross-file target, an open document or a file on disk). The
//! check roots sibling imports at the open document's directory, so `import a/b` resolves
//! to the real project files.
//!
//! CROSS-FILE go-to-definition is supported for imported PUB fns: a `mod.fn` field-access
//! resolves to a global `.func` carrying its owning module + decl node, which maps to that
//! module's real `file` and thus a real `file://` uri. The resolver leaves a type name
//! `.unresolved`, so a type use navigates by its name (`Analysis.typeDecl`): to the entry's
//! type of that name, else an imported module's.
//!
//! A broken import (`graph.err != null`, e.g. a typo'd import path) disables go-to-def for the
//! WHOLE buffer — there is then no sound tree to resolve against.

const std = @import("std");
const toyc = @import("toy_compiler");
const protocol = @import("protocol.zig");
const position = @import("position.zig");
const lsp_uri = @import("uri.zig");
const Analysis = @import("Analysis.zig");

const Token = toyc.Token;
const SourceMap = toyc.term.render.SourceMap;

/// The declaration of the symbol used at (`line`, `character`), allocated in `a`, or null
/// when there is nothing to navigate to (a gap, an out-of-range or unresolved position, a
/// bundled or builtin target with no openable file, or a buffer that does not parse).
pub fn at(a: std.mem.Allocator, an: *Analysis, line: u32, character: u32) !?protocol.Location {
    if (an.checked == null) return null;
    const use_tok = an.tokenAt(line, character) orelse return null;
    const site = (try an.declOf(a, use_tok)) orelse return null;

    if (site.module == an.entry()) {
        const t = an.module().tokens[site.tok];
        return .{ .uri = try a.dupe(u8, an.uri), .range = an.rangeOf(t.start, t.end) };
    }

    // `declOf` already refused a bundled module.
    const tm = &an.graph.modules[site.module];
    const uri = (try lsp_uri.pathToUri(a, tm.file)) orelse return null;
    var tsm = try SourceMap.init(a, tm.file, tm.source);
    defer tsm.deinit(a);
    return .{ .uri = uri, .range = rangeOfToken(&tsm, tm.tokens, site.tok, an.encoding) };
}

fn rangeOfToken(sm: *const SourceMap, tokens: []const Token, t: u32, enc: position.Encoding) protocol.Range {
    return position.rangeOf(sm, tokens[t].start, tokens[t].end, enc);
}

const testing = std.testing;
const Io = std.Io;
const Workspace = @import("Workspace.zig");
const Documents = @import("Documents.zig");

/// A test's definition: the target uri (owned by `gpa`) and range.
const Def = struct { uri: []u8, range: protocol.Range };

/// The definition at (`line`, `character`) in `source`, checked as the document `doc_uri`
/// in `ws`.
fn definitionAt(gpa: std.mem.Allocator, ws: Workspace, source: []const u8, line: u32, character: u32, doc_uri: []const u8) !?Def {
    const an = try @import("test_util.zig").analyzeIn(gpa, ws, doc_uri, source);
    defer an.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const loc = (try at(arena.allocator(), an, line, character)) orelse return null;
    return .{ .uri = try gpa.dupe(u8, loc.uri), .range = loc.range };
}

test "rangeOfToken maps a token's [start,end) to a 0-based exclusive range" {
    const gpa = testing.allocator;
    const src = "fn add(a: int) -> int { return a }";
    var sm = try SourceMap.init(gpa, "t", src);
    defer sm.deinit(gpa);

    // The `add` name token spans bytes [3, 6).
    const add = std.mem.indexOf(u8, src, "add").?;
    const toks = [_]Token{.{ .tag = .identifier, .start = @intCast(add), .end = @intCast(add + 3) }};
    const r = rangeOfToken(&sm, &toks, 0, .utf8);
    try testing.expectEqual(@as(u32, 0), r.start.line);
    try testing.expectEqual(@as(u32, 3), r.start.character);
    try testing.expectEqual(@as(u32, 0), r.end.line);
    try testing.expectEqual(@as(u32, 6), r.end.character); // exclusive end
}

/// The 0-based (line, character) of the first byte of `needle` in `src`.
const posOf = @import("test_util.zig").posOf;

test "definition: cross-file jump to an imported pub fn; within-file stays local; imports resolve" {
    const gpa = testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // A REAL on-disk two-file project. Only helper.toy must come from disk: the entry is
    // served from its buffer, and `doc_uri` roots imports at the real dir.
    const dir_name = ".lsp-xfile-test";
    Io.Dir.cwd().deleteTree(io, dir_name) catch {};
    try Io.Dir.cwd().createDirPath(io, dir_name);
    defer Io.Dir.cwd().deleteTree(io, dir_name) catch {};

    const helper_src = "pub fn foo() -> int { return 1 }\n"; // MUST be pub for cross-file resolution
    const main_src =
        "import helper\n" ++
        "\n" ++
        "fn main() -> int {\n" ++
        "    x := helper.foo()\n" ++
        "    return x\n" ++
        "}\n";
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir_name ++ "/helper.toy", .data = helper_src });
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = dir_name ++ "/main.toy", .data = main_src });

    // Root at the realpath so it is ABSOLUTE (definitionAt refuses a relative tm.file) and
    // canonical (macOS reaches the test dir via a /tmp -> /private/tmp symlink). tm.file is
    // built raw from the root string we pass, so want_uri derives from the SAME abs_dir.
    const abs_dir_z = try Io.Dir.cwd().realPathFileAlloc(io, dir_name, gpa);
    defer gpa.free(abs_dir_z);
    const abs_dir = try gpa.dupe(u8, abs_dir_z);
    defer gpa.free(abs_dir);
    const main_path = try std.fs.path.join(gpa, &.{ abs_dir, "main.toy" });
    defer gpa.free(main_path);
    const doc_uri = try std.fmt.allocPrint(gpa, "file://{s}", .{main_path});
    defer gpa.free(doc_uri);
    const helper_path = try std.fs.path.join(gpa, &.{ abs_dir, "helper.toy" });
    defer gpa.free(helper_path);
    const want_uri = try std.fmt.allocPrint(gpa, "file://{s}", .{helper_path});
    defer gpa.free(want_uri);

    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = io, .docs = &docs };

    // (1) Cross-file: the `foo` use in `helper.foo()` -> helper.toy's real uri + `foo` decl.
    {
        const use = posOf(main_src, "foo()");
        const d = (try definitionAt(gpa, ws, main_src, use.line, use.character, doc_uri)) orelse
            return error.CrossFileReturnedNull; // non-vacuous: pub fn MUST resolve cross-file
        defer gpa.free(d.uri);
        try testing.expectEqualStrings(want_uri, d.uri);
        try testing.expectEqual(@as(u32, 0), d.range.start.line);
        const foo_col: u32 = @intCast(std.mem.indexOf(u8, helper_src, "foo").?);
        try testing.expectEqual(foo_col, d.range.start.character);
        try testing.expectEqual(foo_col + 3, d.range.end.character);
    }

    // (2) Within-file: the `x` use in `return x` -> the OPEN doc uri.
    {
        const rx = posOf(main_src, "return x");
        const use_char: u32 = rx.character + @as(u32, @intCast("return ".len));
        const d = (try definitionAt(gpa, ws, main_src, rx.line, use_char, doc_uri)) orelse
            return error.WithinFileReturnedNull;
        defer gpa.free(d.uri);
        try testing.expectEqualStrings(doc_uri, d.uri);
    }
}

test "definition: cross-file jump into another OPEN document, disk off" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const helper_src = "pub fn foo() -> int { return 1 }\n";
    try docs.put(gpa, "file:///blocks/helper.toy", helper_src, 1);
    const main_src = "import helper\nfn main() -> int { return helper.foo() }\n";

    const ws: Workspace = .{ .io = Io.failing, .docs = &docs, .disk = false };
    const use = posOf(main_src, "foo()");
    const d = (try definitionAt(gpa, ws, main_src, use.line, use.character, "file:///blocks/main.toy")) orelse
        return error.CrossFileReturnedNull;
    defer gpa.free(d.uri);
    try testing.expectEqualStrings("file:///blocks/helper.toy", d.uri);
    const foo_col: u32 = @intCast(std.mem.indexOf(u8, helper_src, "foo").?);
    try testing.expectEqual(foo_col, d.range.start.character);
}

test "definition: a piped callee jumps to its declaration" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\fn add(a: int, b: int) -> int { return a + b }
        \\fn main() -> int {
        \\    x := 1 |> add(2)
        \\    y := x
        \\        |> add(3)
        \\    return y
        \\}
        \\
    ;
    const cases = [_]struct { use: []const u8, decl: []const u8 }{
        .{ .use = "add(2)", .decl = "add(a" },
        .{ .use = "add(3)", .decl = "add(a" },
    };
    for (cases) |c| {
        const use = posOf(src, c.use);
        const d = (try definitionAt(gpa, ws, src, use.line, use.character, "file:///p/main.toy")) orelse {
            std.debug.print("no definition for '{s}'\n", .{c.use});
            return error.NoDefinition;
        };
        defer gpa.free(d.uri);
        const want = posOf(src, c.decl);
        try testing.expectEqual(want.line, d.range.start.line);
        try testing.expectEqual(want.character, d.range.start.character);
    }
}

test "definition: a method callee jumps to its declaration" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src = @import("test_util.zig").method_fixture;
    const cases = [_]struct { use: []const u8, decl: ?[]const u8 }{
        .{ .use = "plus(1, 2)", .decl = "plus(self" },
        .{ .use = "plus(2)", .decl = "plus(self" },
        .{ .use = "get()", .decl = "get(self" },
        .{ .use = "make(3)", .decl = "make(v" },
        // The bounded receiver reaches the protocol's signature, not one impl of it.
        .{ .use = "dbl()", .decl = "dbl(self) -> int\n}" },
        .{ .use = "concat(", .decl = null },
        .{ .use = "eq(3)", .decl = null },
        .{ .use = "unwrap_or(", .decl = null },
        .{ .use = "hash()", .decl = null },
    };
    for (cases) |c| {
        const use = posOf(src, c.use);
        const got = try definitionAt(gpa, ws, src, use.line, use.character, "file:///p/main.toy");
        defer if (got) |d| gpa.free(d.uri);
        const want_decl = c.decl orelse {
            if (got != null) {
                std.debug.print("unexpected definition for '{s}'\n", .{c.use});
                return error.UnexpectedDefinition;
            }
            continue;
        };
        const d = got orelse {
            std.debug.print("no definition for '{s}'\n", .{c.use});
            return error.NoDefinition;
        };
        try testing.expectEqualStrings("file:///p/main.toy", d.uri);
        const want = posOf(src, want_decl);
        try testing.expectEqual(want.line, d.range.start.line);
        try testing.expectEqual(want.character, d.range.start.character);
    }
}

test "definition: a local fn and its params jump to their bindings; the enclosing fn's params still resolve" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\fn main(arg: int) -> int {
        \\    fn step(k: int) -> int { k + arg }
        \\    count := step(arg)
        \\    return count
        \\}
        \\
    ;
    const cases = [_]struct { use: []const u8, skip: u32, decl: []const u8 }{
        .{ .use = "step(arg)", .skip = 0, .decl = "step(k" },
        .{ .use = "k + arg", .skip = 0, .decl = "k: int" },
        .{ .use = "k + arg", .skip = 4, .decl = "arg: int" },
        .{ .use = "return count", .skip = 7, .decl = "count :=" },
    };
    for (cases) |c| {
        const use = posOf(src, c.use);
        const d = (try definitionAt(gpa, ws, src, use.line, use.character + c.skip, "file:///p/main.toy")) orelse {
            std.debug.print("no definition for '{s}'+{d}\n", .{ c.use, c.skip });
            return error.NoDefinition;
        };
        defer gpa.free(d.uri);
        const want = posOf(src, c.decl);
        try testing.expectEqual(want.line, d.range.start.line);
        try testing.expectEqual(want.character, d.range.start.character);
    }
}

test "definition: a bound method value jumps to its method; a protocol name to its declaration" {
    const gpa = testing.allocator;
    var docs: Documents = .{};
    defer docs.deinit(gpa);
    const ws: Workspace = .{ .io = std.Io.failing, .docs = &docs, .disk = false };
    const src =
        \\protocol Area { fn area(self) -> int }
        \\struct Sq { side: int }
        \\impl Sq has Area { fn area(self) -> int { self.side * self.side } }
        \\struct Acc { n: int }
        \\impl Acc { fn plus(self, k: int) -> int { self.n + k } }
        \\fn total[T has Area](v: T) -> int { v.area() }
        \\fn main() -> int {
        \\    acc := Acc { n: 1 }
        \\    g := acc.plus
        \\    return g(1) + total(Sq { side: 2 })
        \\}
        \\
    ;
    const cases = [_]struct { use: []const u8, decl: []const u8 }{
        .{ .use = "plus\n", .decl = "plus(self" },
        .{ .use = "Area { fn area(self) -> int { self", .decl = "Area {" },
        .{ .use = "Area](v", .decl = "Area {" },
    };
    for (cases) |c| {
        const use = posOf(src, c.use);
        const d = (try definitionAt(gpa, ws, src, use.line, use.character, "file:///p/main.toy")) orelse {
            std.debug.print("no definition for '{s}'\n", .{c.use});
            return error.NoDefinition;
        };
        defer gpa.free(d.uri);
        const want = posOf(src, c.decl);
        try testing.expectEqual(want.line, d.range.start.line);
        try testing.expectEqual(want.character, d.range.start.character);
    }
}
