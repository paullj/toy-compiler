//! Library surface of the toy compiler. Re-exports the pieces so they can be
//! imported by consumers and exercised by the test runner.

pub const token = @import("ast/Token.zig");
pub const Token = token.Token;
pub const Tag = token.Tag;
pub const Lexer = @import("lex.zig");
pub const Ast = @import("ast/Ast.zig");
pub const Parser = @import("parse.zig");
pub const Cache = @import("query/Cache.zig");
pub const QueryEngine = @import("query/Engine.zig");
pub const QueryKey = @import("query/Key.zig");
pub const StageGraph = @import("query/StageGraph.zig");
pub const Driver = @import("driver/Driver.zig");
pub const Graph = @import("driver/Graph.zig");
pub const DriverCodegen = @import("driver/Codegen.zig");
pub const DiagnosticSink = @import("diagnostics/Sink.zig");
pub const Resolve = @import("resolve.zig");
pub const ResolveGraph = @import("resolve_graph.zig");
pub const Typecheck = @import("types.zig");
pub const TypecheckGraph = @import("types_graph.zig");
pub const LayoutEngine = @import("layout/Engine.zig");
pub const Aarch64 = @import("codegen/Aarch64.zig");
pub const CodegenIr = @import("codegen/CodegenIr.zig");
pub const Abi = @import("codegen/abi/Abi.zig");
pub const FrameLayout = @import("codegen/frame/FrameLayout.zig");
pub const Ir = @import("ir/Ir.zig");
pub const Opt = @import("opt/Opt.zig");
pub const lower = @import("lower.zig");
pub const Fingerprint = @import("query/Fingerprint.zig");
pub const AstWalk = @import("query/AstWalk.zig");
pub const Walks = @import("query/Walks.zig");
pub const Link = @import("link/Link.zig");
pub const MachO = @import("link/MachO.zig");
pub const CodeSign = @import("link/CodeSign.zig");
pub const link = @import("link/emit.zig");
pub const version = @import("version.zig");
pub const cli = struct {
    pub const Spec = @import("cli/Spec.zig");
    pub const Parsed = @import("cli/Parsed.zig");
    pub const Parser = @import("cli/Parser.zig");
    pub const Sink = @import("cli/Sink.zig");
    pub const Help = @import("cli/Help.zig");
    pub const Cli = @import("cli/Cli.zig");
};
pub const term = struct {
    pub const ansi = @import("term/ansi.zig");
    pub const Style = @import("term/Style.zig");
    pub const Terminal = @import("term/Terminal.zig");
    pub const Progress = @import("term/Progress.zig");
    pub const render = struct {
        pub const SourceMap = @import("term/render/SourceMap.zig");
        pub const Diagnostic = @import("term/render/Diagnostic.zig");
        pub const Theme = @import("term/render/Theme.zig");
        pub const Renderer = @import("term/render/Renderer.zig");
    };
};

test {
    // Pull in tests from every module that has them.
    _ = Lexer;
    _ = Parser;
    _ = Cache;
    _ = QueryEngine;
    _ = QueryKey;
    _ = StageGraph;
    _ = Driver;
    _ = Graph;
    _ = DiagnosticSink;
    _ = Resolve;
    _ = ResolveGraph;
    _ = Typecheck;
    _ = TypecheckGraph;
    _ = LayoutEngine;
    _ = Aarch64;
    _ = CodegenIr;
    _ = Abi;
    _ = FrameLayout;
    _ = Ir;
    _ = Opt;
    _ = lower;
    _ = Fingerprint;
    _ = AstWalk;
    _ = Walks;
    _ = Link;
    _ = MachO;
    _ = CodeSign;
    _ = link; // discovers emit.zig's boundary test
    _ = cli.Spec;
    _ = cli.Parsed;
    _ = cli.Parser;
    _ = cli.Sink;
    _ = cli.Help;
    _ = cli.Cli;
    _ = term.ansi;
    _ = term.Style;
    _ = term.Terminal;
    _ = term.Progress;
    _ = term.render.SourceMap;
    _ = term.render.Diagnostic;
    _ = term.render.Theme;
    _ = term.render.Renderer;
    // Inline unit tests live in their modules (pulled via the `_ = X` refs above —
    // e.g. Cache's digest tests in query/Cache.zig). The engine BOUNDARY suite
    // (integration: temp-dir cache + threaded runtime) is NOT pulled here — it lives
    // in the repo-root tests/ and runs in its own `toy-integration-test` artifact
    // (see build.zig), consuming the compiler as a black box via @import("toy_compiler").
}
