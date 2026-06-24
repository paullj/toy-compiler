//! Library surface of the toy compiler. Re-exports the pieces so they can be
//! imported by consumers and exercised by the test runner.

pub const token = @import("ast/Token.zig");
pub const Token = token.Token;
pub const Tag = token.Tag;
pub const Lexer = @import("lex.zig");
pub const Ast = @import("ast/Ast.zig");
pub const Parser = @import("parse.zig");
pub const Cache = @import("driver/Cache.zig");
pub const Driver = @import("driver/Driver.zig");
pub const Resolve = @import("resolve.zig");
pub const Typecheck = @import("types.zig");
pub const Aarch64 = @import("codegen/Aarch64.zig");
pub const Codegen = @import("codegen/Codegen.zig");
pub const Fingerprint = @import("driver/Fingerprint.zig");
pub const Link = @import("link/Link.zig");
pub const MachO = @import("link/MachO.zig");
pub const CodeSign = @import("link/CodeSign.zig");
pub const link = @import("link/emit.zig");
pub const version = @import("version.zig");

test {
    // Pull in tests from every module that has them.
    _ = Lexer;
    _ = Parser;
    _ = Cache;
    _ = Driver;
    _ = Resolve;
    _ = Typecheck;
    _ = Aarch64;
    _ = Codegen;
    _ = Fingerprint;
    _ = Link;
    _ = MachO;
    _ = CodeSign;
    _ = link; // discovers emit.zig's boundary test
    _ = @import("driver/Cache_test.zig");
}
