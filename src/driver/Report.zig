//! The inspection-mode status output: the per-file summary table + row/cell
//! formatters, the `--dump` artifact printers, and the cache-note helper. Statuses
//! colourise via DiagRender's shared palette; per-diagnostic detail routes through
//! its Renderer. Only the last cell on a row is ever styled (width-safe).

const std = @import("std");
const Io = std.Io;
const toyc = @import("toy_compiler");
const Ast = toyc.Ast;
const Driver = toyc.Driver;
const version = toyc.version;
const term = toyc.term;
const Style = term.Style;
const Rr = term.render;
const DiagRender = @import("DiagRender.zig");
const sty_err = DiagRender.sty_err;
const sty_ok = DiagRender.sty_ok;
const sty_head = DiagRender.sty_head;
const sty_faint = DiagRender.sty_faint;

/// Print the per-file summary; returns the number of files that failed (for the
/// caller's exit status). Per-diagnostic detail lines route through the Renderer
/// via `printFailure`.
pub fn report(out: *Io.Writer, gpa: std.mem.Allocator, level: Style.ColorLevel, results: []const Driver.FileResult, emit: Driver.Emit, target: []const u8, dump: bool) !usize {
    var stamp_buf: [version.stamp_max]u8 = undefined;
    // Header: the version stamp bold, labels plain.
    try out.writeAll("compiler ");
    try sty_head.styled(out, level, version.stamp(&stamp_buf));
    try out.print("  target {s}  emit {t}\n", .{ target, emit });

    var total_tokens: usize = 0;
    var total_bytes: usize = 0;
    var lex_hits: usize = 0;
    var parse_hits: usize = 0;
    var failures: usize = 0;

    // Table header row: bold (the whole line is a header, no data cells to keep plain).
    {
        var hbuf: [80]u8 = undefined;
        const hdr = std.fmt.bufPrint(&hbuf, "{s: <28} {s: >8} {s: >7} {s: >6}  {s}", .{ "file", "bytes", "tokens", "nodes", "status" }) catch unreachable;
        try sty_head.styled(out, level, hdr);
        try out.writeByte('\n');
    }
    // Separator: dim.
    try sty_faint.styled(out, level, "-" ** 72);
    try out.writeByte('\n');

    for (results) |r| {
        total_bytes += r.source.len;
        if (r.tokens_cached) lex_hits += 1;
        if (r.nodes_cached) parse_hits += 1;

        if (r.err) |err| {
            failures += 1;
            try printFailure(out, gpa, level, r, err);
            continue;
        }

        // Every token stream ends in a synthetic .eof we don't count as "real".
        const real_tokens = r.tokens.len -| 1;
        total_tokens += real_tokens;

        // Data cells (file/bytes/tokens/nodes) stay PLAIN; only the STATUS cell is
        // coloured. The status cell is LAST (no trailing padding), so styling its
        // whole content is width-safe.
        try out.print("{s: <28} {d: >8} {d: >7} ", .{ r.path, r.source.len, real_tokens });
        if (r.parsed) try out.print("{d: >6}", .{r.nodes.len}) else try out.print("{s: >6}", .{"-"});
        try out.writeAll("  ");
        const note = cacheNote(r, emit);
        // `cached` recedes (dim); a fresh phase reads green.
        const nstyle: Style.Style = if (std.mem.eql(u8, note, "cached")) sty_faint else sty_ok;
        try nstyle.styled(out, level, note);
        try out.writeByte('\n');

        if (dump) try dumpArtifact(out, r, emit);
    }

    try sty_faint.styled(out, level, "-" ** 72);
    try out.writeByte('\n');
    // Totals: the `F failure(s)` fragment err-red bold when failures>0, else the
    // whole line dim. Pad-plain (the numbers) then style only the coloured fragment.
    if (failures > 0) {
        try out.print(
            "{d} file(s): {d} bytes, {d} tokens, cache hits lex={d} parse={d}, ",
            .{ results.len, total_bytes, total_tokens, lex_hits, parse_hits },
        );
        var fbuf: [32]u8 = undefined;
        const frag = std.fmt.bufPrint(&fbuf, "{d} failure(s)", .{failures}) catch unreachable;
        try sty_err.styled(out, level, frag);
        try out.writeByte('\n');
    } else {
        var lbuf: [160]u8 = undefined;
        const line = std.fmt.bufPrint(&lbuf, "{d} file(s): {d} bytes, {d} tokens, cache hits lex={d} parse={d}, {d} failure(s)", .{ results.len, total_bytes, total_tokens, lex_hits, parse_hits, failures }) catch unreachable;
        try sty_faint.styled(out, level, line);
        try out.writeByte('\n');
    }
    return failures;
}

/// Short per-file status: which phases were fresh vs. served from cache.
fn cacheNote(r: Driver.FileResult, emit: Driver.Emit) []const u8 {
    const lex = if (r.tokens_cached) "cached" else "lexed";
    if (emit == .lex) return lex;
    // Resolution runs in-memory (uncached); report it explicitly when reached.
    if (emit == .check and r.checked) return "checked";
    const parse = if (r.nodes_cached) "cached" else "parsed";
    if (r.tokens_cached and r.nodes_cached) return "cached";
    if (!r.tokens_cached and !r.nodes_cached) return "lexed+parsed";
    // Mixed: name the phase that actually ran.
    return if (r.tokens_cached) parse else lex;
}

/// Emit a failing file's summary ROW (kept, now styled — the `error: ...` cell
/// err-red bold, the path plain), then route each per-file diagnostic through the
/// Renderer over ONE SourceMap of the file (single-file => scope == NO_SCOPE).
fn printFailure(out: *Io.Writer, gpa: std.mem.Allocator, level: Style.ColorLevel, r: Driver.FileResult, err: anyerror) !void {
    if (r.diag) |d| {
        // Parse error: a single diagnostic. The row's `error:` cell is styled; the
        // pretty snippet follows via the Renderer.
        try rowThenErr(out, level, r.path, r.source.len, r.tokens.len -| 1, "-", "parse error", 0);
        var sm = try Rr.SourceMap.init(gpa, r.path, r.source);
        defer sm.deinit(gpa);
        try DiagRender.renderSinkDiag(out, level, &sm, d);
    } else if (err == error.ResolveError) {
        const n = if (r.resolve) |res| res.diags.len else 0;
        var msgbuf: [48]u8 = undefined;
        try rowThenErr(out, level, r.path, r.source.len, r.tokens.len -| 1, "-", std.fmt.bufPrint(&msgbuf, "{d} resolve diagnostic(s)", .{n}) catch "resolve diagnostics", 0);
        if (r.resolve) |res| {
            var sm = try Rr.SourceMap.init(gpa, r.path, r.source);
            defer sm.deinit(gpa);
            for (res.diags) |d| try DiagRender.renderSinkDiag(out, level, &sm, d);
        }
    } else if (err == error.TypeError) {
        const n = if (r.typecheck) |tc| tc.diags.len else 0;
        var msgbuf: [48]u8 = undefined;
        try rowThenErr(out, level, r.path, r.source.len, r.tokens.len -| 1, null, std.fmt.bufPrint(&msgbuf, "{d} type diagnostic(s)", .{n}) catch "type diagnostics", r.nodes.len);
        if (r.typecheck) |tc| {
            var sm = try Rr.SourceMap.init(gpa, r.path, r.source);
            defer sm.deinit(gpa);
            for (tc.diags) |d| try DiagRender.renderSinkDiag(out, level, &sm, d);
        }
    } else {
        var msgbuf: [48]u8 = undefined;
        try rowDashThenErr(out, level, r.path, std.fmt.bufPrint(&msgbuf, "{t}", .{err}) catch "error");
    }
}

/// A failing-file summary row `path bytes tokens nodes  error: <msg>` where the data
/// cells (path/bytes/tokens/nodes) stay PLAIN and only the trailing `error: <msg>`
/// is styled (err-red bold, last on the line so width-safe). `nodes` is either the
/// literal "-" (pass `"-"` as `nodes_str`, `nodes_num` unused) or a count (pass
/// `null` as `nodes_str` and the count as `nodes_num`).
fn rowThenErr(out: *Io.Writer, level: Style.ColorLevel, path: []const u8, bytes: usize, tokens: usize, nodes_str: ?[]const u8, msg: []const u8, nodes_num: usize) !void {
    try out.print("{s: <28} {d: >8} {d: >7} ", .{ path, bytes, tokens });
    if (nodes_str) |s| try out.print("{s: >6}", .{s}) else try out.print("{d: >6}", .{nodes_num});
    try out.writeAll("  ");
    try errWord(out, level, msg);
}

/// Variant of `rowThenErr` for the catch-all error row: every data cell is "-".
fn rowDashThenErr(out: *Io.Writer, level: Style.ColorLevel, path: []const u8, msg: []const u8) !void {
    try out.print("{s: <28} {s: >8} {s: >7} {s: >6}  ", .{ path, "-", "-", "-" });
    try errWord(out, level, msg);
}

/// Emit a styled `error: <msg>` fragment (the `error:` prefix + message together in
/// err-red bold) ending the line. Gate-safe at `.none`.
fn errWord(out: *Io.Writer, level: Style.ColorLevel, msg: []const u8) !void {
    var buf: [80]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "error: {s}", .{msg}) catch "error";
    try sty_err.styled(out, level, s);
    try out.writeByte('\n');
}

fn dumpArtifact(out: *Io.Writer, r: Driver.FileResult, emit: Driver.Emit) !void {
    switch (emit) {
        .lex => for (r.tokens, 0..) |tok, i| {
            try out.print("    [{d: >4}] {s: <12} {d: >5}..{d: <5} {s}\n", .{
                i, @tagName(tok.tag), tok.start, tok.end,
                if (tok.tag == .eof) "" else tok.text(r.source),
            });
        },
        .parse => {
            try out.writeAll("    ");
            try Ast.render(out, .{ .nodes = r.nodes, .extra = r.extra, .pub_bits = r.pub_bits }, r.tokens, r.source);
            try out.writeByte('\n');
        },
        // Check dump: the AST plus a per-function signature summary with the
        // inferred return type, then a tally of diagnostics (0 when clean).
        // `ir` shares the front-end with `check`; its artifact (the IR text) is
        // printed by `emitIr`, not the table dumper.
        .check, .ir => {
            try out.writeAll("    ");
            try Ast.render(out, .{ .nodes = r.nodes, .extra = r.extra, .pub_bits = r.pub_bits }, r.tokens, r.source);
            try out.writeByte('\n');
            try dumpCheck(out, r);
        },
    }
}

/// Per-function summary under `--emit check --dump`: each function's resolved
/// signature with its inferred return type, then a diagnostic count.
fn dumpCheck(out: *Io.Writer, r: Driver.FileResult) !void {
    if (r.nodes.len == 0) return;
    const tree: Ast.Tree = .{ .nodes = r.nodes, .extra = r.extra };
    const prog = r.nodes[Ast.root(r.nodes)];
    if (prog.tag != .program) return;

    const tc = r.typecheck;
    for (Ast.rangeSlice(tree, prog.lhs)) |fn_idx| {
        const decl = r.nodes[fn_idx];
        if (decl.tag != .fn_decl) continue;
        const name = r.tokens[decl.main_token].text(r.source);
        const proto = Ast.protoAt(tree, decl.lhs);

        try out.print("    fn {s}(", .{name});
        for (proto.params, 0..) |p_idx, i| {
            if (i != 0) try out.writeAll(", ");
            const p = r.nodes[p_idx];
            const p_name = r.tokens[p.main_token].text(r.source);
            const p_type = r.tokens[r.nodes[p.lhs].main_token].text(r.source);
            try out.print("{s}: {s}", .{ p_name, p_type });
        }
        const ret = if (proto.ret_type == Ast.none)
            "()"
        else if (r.nodes[proto.ret_type].tag == .literal_unit)
            "()"
        else
            r.tokens[r.nodes[proto.ret_type].main_token].text(r.source);
        // `rhs` is the body block; its inferred type isn't tracked, so report the
        // declared return spelling — the typechecker has already verified it.
        try out.print(") -> {s}\n", .{ret});
    }

    const n_diags = if (tc) |t| t.diags.len else 0;
    try out.print("    checked: {d} diagnostic(s)\n", .{n_diags});
}
