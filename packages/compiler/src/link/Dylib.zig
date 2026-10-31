//! Find a user dynamic library and read what a binary must record to load it.
//!
//! The writer emits one LC_LOAD_DYLIB per library and binds each import to its
//! library's ordinal, so the linker must know, for every library, the install name
//! dyld will look up and the versions dyld checks. A library on disk supplies both
//! through its LC_ID_DYLIB. A system library under /usr/lib has no file on disk on
//! current macOS (it lives in the dyld shared cache), so it is accepted when the
//! cache holds it and recorded by its path.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const MachO = @import("MachO.zig");

/// Searched after every `-L` directory, as the system linker does. /usr/lib is
/// also where the shared-cache fallback looks.
pub const default_dirs = [_][]const u8{ "/usr/local/lib", "/usr/lib" };

const libsystem_path = "/usr/lib/libSystem.B.dylib";

/// Names whose library is libSystem itself: libc, libm, libpthread and libdl are
/// part of it, so naming one adds no load command.
const libsystem_names = [_][]const u8{ "System", "c", "m", "pthread", "dl" };

/// What a program links beyond libSystem: the libraries to load, the LC_RPATH
/// entries their `@rpath/` install names need, and the C symbols bound to a library
/// other than libSystem. Every slice is borrowed from whoever built it.
pub const Linking = struct {
    dylibs: []const MachO.Dylib = &.{},
    rpaths: []const []const u8 = &.{},
    /// Sorted by name, one entry per symbol.
    bindings: []const Binding = &.{},

    pub fn load(self: Linking) MachO.Libraries {
        return .{ .dylibs = self.dylibs, .rpaths = self.rpaths };
    }

    /// The import ordinal of the bare C symbol `name`; a symbol no library claims
    /// binds to libSystem.
    pub fn ordinalOf(self: Linking, name: []const u8) u8 {
        var lo: usize = 0;
        var hi: usize = self.bindings.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            switch (std.mem.order(u8, self.bindings[mid].name, name)) {
                .eq => return self.bindings[mid].ordinal,
                .lt => lo = mid + 1,
                .gt => hi = mid,
            }
        }
        return 1;
    }
};

pub const Binding = struct { name: []const u8, ordinal: u8 };

/// Where a library name led. Strings are allocated in the allocator `find` got.
pub const Found = union(enum) {
    libsystem,
    dylib: struct { dylib: MachO.Dylib, rpath: ?[]const u8 },
    /// Only a static archive or object matched: the path of the first one.
    static: []const u8,
    not_found,
    not_dylib: struct { path: []const u8, why: []const u8 },
};

/// True for a name that is a static archive or an object file, which cannot be
/// loaded at run time.
pub fn isStatic(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".a") or std.mem.endsWith(u8, name, ".o");
}

/// True when `name` is a path to use as given rather than a library to search for.
pub fn isPath(name: []const u8) bool {
    return std.mem.indexOfScalar(u8, name, '/') != null or std.mem.endsWith(u8, name, ".dylib");
}

/// Look `name` up: a path (`isPath`) as given, relative to the working directory;
/// otherwise `lib<name>.dylib` in each of `dirs`, then `default_dirs`, then the
/// shared cache. A directory holding only `lib<name>.a` does not stop the search,
/// but if no dylib turns up the archive is reported, so the user learns why.
pub fn find(arena: std.mem.Allocator, io: Io, name: []const u8, dirs: []const []const u8) !Found {
    if (isStatic(name)) return .{ .static = name };
    for (libsystem_names) |n| if (std.mem.eql(u8, n, name)) return .libsystem;
    if (isPath(name)) {
        if (!exists(io, name)) return if (std.fs.path.isAbsolute(name) and inSharedCache(name)) cached(name) else .not_found;
        return open(arena, io, name);
    }
    var static: ?[]const u8 = null;
    for ([2][]const []const u8{ dirs, &default_dirs }) |group| for (group) |dir| {
        const dylib = try std.fmt.allocPrint(arena, "{s}/lib{s}.dylib", .{ dir, name });
        if (exists(io, dylib)) return open(arena, io, dylib);
        if (static == null) {
            const archive = try std.fmt.allocPrint(arena, "{s}/lib{s}.a", .{ dir, name });
            if (exists(io, archive)) static = archive;
        }
    };
    const system = try std.fmt.allocPrint(arena, "/usr/lib/lib{s}.dylib", .{name});
    if (inSharedCache(system)) return cached(system);
    return if (static) |s| .{ .static = s } else .not_found;
}

fn cached(path: []const u8) Found {
    if (std.mem.eql(u8, path, libsystem_path)) return .libsystem;
    return .{ .dylib = .{ .dylib = .{ .install_name = path }, .rpath = null } };
}

fn exists(io: Io, path: []const u8) bool {
    Io.Dir.cwd().access(io, path, .{}) catch return false;
    return true;
}

/// Read the library at `path` and choose the install name the binary records.
fn open(arena: std.mem.Allocator, io: Io, path: []const u8) !Found {
    const bytes = Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 30)) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return .{ .not_dylib = .{ .path = path, .why = "it cannot be read" } },
    };
    const id = switch (readId(bytes)) {
        .ok => |id| id,
        .bad => |why| return .{ .not_dylib = .{ .path = path, .why = why } },
    };
    const real = Io.Dir.cwd().realPathFileAlloc(io, path, arena) catch try arena.dupe(u8, path);
    const choice = try record(arena, id.install_name, real);
    if (std.mem.eql(u8, choice.install_name, libsystem_path)) return .libsystem;
    return .{ .dylib = .{
        .dylib = .{ .install_name = choice.install_name, .current_version = id.current_version, .compat_version = id.compat_version },
        .rpath = choice.rpath,
    } };
}

/// The install name to record for a library whose LC_ID_DYLIB says `install_name`
/// and which was found at the absolute path `real`. An absolute or `@`-relative
/// name is what the library's author chose and dyld resolves it at load time;
/// `@rpath/` also needs the found folder as an LC_RPATH so the binary runs where it
/// was built. A bare or relative name (what `cc -shared` writes by default) would be
/// looked up relative to the working directory at run time, so the found path is
/// recorded instead.
pub fn record(arena: std.mem.Allocator, install_name: []const u8, real: []const u8) !struct { install_name: []const u8, rpath: ?[]const u8 } {
    if (std.mem.startsWith(u8, install_name, "@rpath/"))
        return .{ .install_name = try arena.dupe(u8, install_name), .rpath = std.fs.path.dirname(real) orelse "/" };
    if (install_name.len > 0 and (install_name[0] == '@' or install_name[0] == '/'))
        return .{ .install_name = try arena.dupe(u8, install_name), .rpath = null };
    return .{ .install_name = real, .rpath = null };
}

/// A library's LC_ID_DYLIB: its install name (borrowed from the bytes read) and
/// the versions dyld checks.
pub const Id = struct { install_name: []const u8, current_version: u32, compat_version: u32 };

const MH_MAGIC_64: u32 = 0xFEEDFACF;
const FAT_MAGIC: u32 = 0xCAFEBABE;
const FAT_MAGIC_64: u32 = 0xCAFEBABF;
const CPU_TYPE_ARM64: u32 = 0x0100000C;
const MH_DYLIB: u32 = 0x6;
const LC_ID_DYLIB: u32 = 0xd;

/// Parse the arm64 Mach-O dylib in `bytes` (thin, or the arm64 slice of a
/// universal file) for its LC_ID_DYLIB, or say why it is not one.
pub const Read = union(enum) { ok: Id, bad: []const u8 };

pub fn readId(bytes: []const u8) Read {
    const not_macho = "it is not a Mach-O file";
    if (bytes.len < 8) return .{ .bad = not_macho };
    const be = std.mem.readInt(u32, bytes[0..4], .big);
    if (be == FAT_MAGIC or be == FAT_MAGIC_64) {
        const wide = be == FAT_MAGIC_64;
        const n = std.mem.readInt(u32, bytes[4..8], .big);
        const entry: usize = if (wide) 32 else 20;
        for (0..n) |i| {
            const at = 8 + i * entry;
            if (at + entry > bytes.len) return .{ .bad = not_macho };
            if (std.mem.readInt(u32, bytes[at..][0..4], .big) != CPU_TYPE_ARM64) continue;
            const off: u64 = if (wide) std.mem.readInt(u64, bytes[at + 8 ..][0..8], .big) else std.mem.readInt(u32, bytes[at + 8 ..][0..4], .big);
            const size: u64 = if (wide) std.mem.readInt(u64, bytes[at + 16 ..][0..8], .big) else std.mem.readInt(u32, bytes[at + 12 ..][0..4], .big);
            if (off > bytes.len or size > bytes.len - off) return .{ .bad = not_macho };
            return readThin(bytes[@intCast(off)..][0..@intCast(size)]);
        }
        return .{ .bad = "it has no arm64 slice" };
    }
    return readThin(bytes);
}

fn readThin(bytes: []const u8) Read {
    const not_macho = "it is not a Mach-O file";
    if (bytes.len < 32 or std.mem.readInt(u32, bytes[0..4], .little) != MH_MAGIC_64) return .{ .bad = not_macho };
    if (std.mem.readInt(u32, bytes[4..8], .little) != CPU_TYPE_ARM64) return .{ .bad = "it is not built for arm64" };
    if (std.mem.readInt(u32, bytes[12..16], .little) != MH_DYLIB) return .{ .bad = "it is not a dynamic library" };
    const ncmds = std.mem.readInt(u32, bytes[16..20], .little);
    var off: usize = 32;
    for (0..ncmds) |_| {
        if (off + 8 > bytes.len) break;
        const cmd = std.mem.readInt(u32, bytes[off..][0..4], .little);
        const size = std.mem.readInt(u32, bytes[off + 4 ..][0..4], .little);
        if (size < 8 or off + size > bytes.len) break;
        if (cmd == LC_ID_DYLIB and size >= 24) {
            const name_off = std.mem.readInt(u32, bytes[off + 8 ..][0..4], .little);
            if (name_off >= size) break;
            return .{ .ok = .{
                .install_name = std.mem.sliceTo(bytes[off + name_off .. off + size], 0),
                .current_version = std.mem.readInt(u32, bytes[off + 16 ..][0..4], .little),
                .compat_version = std.mem.readInt(u32, bytes[off + 20 ..][0..4], .little),
            } };
        }
        off += size;
    }
    return .{ .bad = "it has no install name (LC_ID_DYLIB)" };
}

/// True when the dyld shared cache holds `path`. Only macOS has one; elsewhere (the
/// wasm language server) nothing is cached.
fn inSharedCache(path: []const u8) bool {
    if (comptime builtin.os.tag != .macos) return false;
    var buf: [std.fs.max_path_bytes:0]u8 = undefined;
    if (path.len >= buf.len) return false;
    @memcpy(buf[0..path.len], path);
    buf[path.len] = 0;
    const dyld = struct {
        extern "c" fn _dyld_shared_cache_contains_path(p: [*:0]const u8) bool;
    };
    return dyld._dyld_shared_cache_contains_path(&buf);
}

const testing = std.testing;

/// A minimal thin arm64 Mach-O header with one LC_ID_DYLIB naming `name`, for tests.
pub fn fakeDylib(buf: []u8, name: []const u8, filetype: u32) []u8 {
    @memset(buf, 0);
    const cmdsize: u32 = @intCast(std.mem.alignForward(usize, 24 + name.len + 1, 8));
    std.mem.writeInt(u32, buf[0..4], MH_MAGIC_64, .little);
    std.mem.writeInt(u32, buf[4..8], CPU_TYPE_ARM64, .little);
    std.mem.writeInt(u32, buf[12..16], filetype, .little);
    std.mem.writeInt(u32, buf[16..20], 1, .little);
    std.mem.writeInt(u32, buf[20..24], cmdsize, .little);
    std.mem.writeInt(u32, buf[32..36], LC_ID_DYLIB, .little);
    std.mem.writeInt(u32, buf[36..40], cmdsize, .little);
    std.mem.writeInt(u32, buf[40..44], 24, .little);
    std.mem.writeInt(u32, buf[48..52], 0x10203, .little);
    std.mem.writeInt(u32, buf[52..56], 0x10000, .little);
    @memcpy(buf[56..][0..name.len], name);
    return buf[0 .. 32 + cmdsize];
}

test "readId reads a thin arm64 dylib's install name and versions" {
    var buf: [256]u8 = undefined;
    const id = readId(fakeDylib(&buf, "@rpath/libffi.dylib", MH_DYLIB)).ok;
    try testing.expectEqualStrings("@rpath/libffi.dylib", id.install_name);
    try testing.expectEqual(@as(u32, 0x10203), id.current_version);
    try testing.expectEqual(@as(u32, 0x10000), id.compat_version);
}

test "readId finds the arm64 slice of a universal file" {
    var buf: [512]u8 = undefined;
    @memset(&buf, 0);
    std.mem.writeInt(u32, buf[0..4], FAT_MAGIC, .big);
    std.mem.writeInt(u32, buf[4..8], 2, .big);
    std.mem.writeInt(u32, buf[8..12], 0x01000007, .big); // x86_64 first
    std.mem.writeInt(u32, buf[28..32], CPU_TYPE_ARM64, .big);
    std.mem.writeInt(u32, buf[36..40], 128, .big);
    var thin: [256]u8 = undefined;
    const t = fakeDylib(&thin, "/opt/lib/libz.dylib", MH_DYLIB);
    std.mem.writeInt(u32, buf[40..44], @intCast(t.len), .big);
    @memcpy(buf[128..][0..t.len], t);
    try testing.expectEqualStrings("/opt/lib/libz.dylib", readId(&buf).ok.install_name);
}

test "readId explains a file that is not an arm64 dylib" {
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("it is not a dynamic library", readId(fakeDylib(&buf, "x", 0x2)).bad);
    try testing.expectEqualStrings("it is not a Mach-O file", readId("!<arch>\nfoo.o").bad);
    const t = fakeDylib(&buf, "x", MH_DYLIB);
    std.mem.writeInt(u32, t[4..8], 0x01000007, .little);
    try testing.expectEqualStrings("it is not built for arm64", readId(t).bad);
}

test "record keeps an absolute or @ install name, adds the found folder for @rpath, and pins a bare one" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try record(a, "@rpath/libffi.dylib", "/w/zig-out/lib/libffi.dylib");
    try testing.expectEqualStrings("@rpath/libffi.dylib", r.install_name);
    try testing.expectEqualStrings("/w/zig-out/lib", r.rpath.?);
    const abs = try record(a, "/opt/z/libz.1.dylib", "/opt/z/libz.dylib");
    try testing.expectEqualStrings("/opt/z/libz.1.dylib", abs.install_name);
    try testing.expect(abs.rpath == null);
    const loader = try record(a, "@loader_path/libq.dylib", "/q/libq.dylib");
    try testing.expectEqualStrings("@loader_path/libq.dylib", loader.install_name);
    try testing.expect(loader.rpath == null);
    const bare = try record(a, "libf.dylib", "/w/libf.dylib");
    try testing.expectEqualStrings("/w/libf.dylib", bare.install_name);
}

test "names: static inputs, paths and libSystem aliases" {
    try testing.expect(isStatic("libz.a") and isStatic("x.o") and !isStatic("z"));
    try testing.expect(isPath("./libz.dylib") and isPath("libz.dylib") and isPath("/x/y") and !isPath("z"));
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    try testing.expect(try find(arena.allocator(), threaded.io(), "m", &.{}) == .libsystem);
    try testing.expectEqualStrings("libz.a", (try find(arena.allocator(), threaded.io(), "libz.a", &.{})).static);
}

test "find searches -L folders for lib<name>.dylib and reports a folder that holds only an archive" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var threaded: std.Io.Threaded = .init(testing.allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [256]u8 = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyq.dylib", .data = fakeDylib(&buf, "@rpath/libtoyq.dylib", MH_DYLIB) });
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyarc.a", .data = "!<arch>\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "libtoyexe.dylib", .data = fakeDylib(&buf, "x", 0x2) });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);

    const q = (try find(a, io, "toyq", &.{dir})).dylib;
    try testing.expectEqualStrings("@rpath/libtoyq.dylib", q.dylib.install_name);
    try testing.expectEqualStrings(dir, q.rpath.?);
    try testing.expect(std.mem.endsWith(u8, (try find(a, io, "toyarc", &.{dir})).static, "/libtoyarc.a"));
    try testing.expectEqualStrings("it is not a dynamic library", (try find(a, io, "toyexe", &.{dir})).not_dylib.why);
    try testing.expect(try find(a, io, "toy-no-such-lib", &.{dir}) == .not_found);
}

test "ordinalOf finds a bound symbol and defaults to libSystem" {
    const l: Linking = .{ .bindings = &.{ .{ .name = "a_fn", .ordinal = 2 }, .{ .name = "m_fn", .ordinal = 3 }, .{ .name = "z_fn", .ordinal = 2 } } };
    try testing.expectEqual(@as(u8, 3), l.ordinalOf("m_fn"));
    try testing.expectEqual(@as(u8, 2), l.ordinalOf("z_fn"));
    try testing.expectEqual(@as(u8, 1), l.ordinalOf("write"));
}
