const std = @import("std");

pub const Binding = enum {
    c,
    @"c++",
    python,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const linkage = b.option(std.builtin.LinkMode, "linkage", "Linkage for the libraries emitted for the bindings") orelse std.builtin.LinkMode.static;

    const python_include = b.option([]const u8, "python-include", "CPython include dir");
    const nanobind_root = b.option([]const u8, "nanobind-root", "nanobind package root");

    const bindings = b.option([]const Binding, "bindings", "Bindings to build") orelse @as([]const Binding, &[_]Binding{
        .c,
        .@"c++",
    });

    const loom_mod = b.addModule("loom", .{
        .root_source_file = b.path("lib/loom.zig"),
        .target = target,
        .optimize = optimize,
    });

    const test_step = b.step("test", "Run unit tests");

    const want_c = std.mem.find(Binding, bindings, &.{.c}) != null;
    const want_cpp = std.mem.find(Binding, bindings, &.{.@"c++"}) != null;
    const want_py = std.mem.find(Binding, bindings, &.{.python}) != null;
    if (want_c or want_cpp or want_py) {
        const clib = b.addLibrary(.{
            .linkage = linkage,
            .name = "loom",
            .root_module = b.createModule(.{
                .root_source_file = b.path("bindings/c/lib/loom.zig"),
                .target = target,
                .optimize = optimize,
                // The python ext links this static lib into a shared object, so it
                // must be position-independent.
                .pic = if (want_py) true else null,
                .imports = &.{.{ .name = "loom", .module = loom_mod }},
            }),
        });
        // The C static lib + headers are for C/C++ consumers only; the python ext
        // statically links the lib into `pyloom.so`, so a python-only build
        // installs just the python package.
        if (want_c or want_cpp) {
            b.installArtifact(clib);
            b.installFile("bindings/c/include/loom.h", "include/loom.h");
        }

        const bind_t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path("bindings/c/lib/loom.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "loom", .module = loom_mod }},
            }),
        });
        test_step.dependOn(&b.addRunArtifact(bind_t).step);

        if (want_cpp) {
            b.installFile("bindings/c++/include/loom-cpp.hpp", "include/loom-cpp.hpp");

            // Prove the header actually compiles and links against the C ABI, and
            // that the error path throws. No hardware needed.
            const smoke_mod = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libcpp = true,
            });
            smoke_mod.addCSourceFile(.{ .file = b.path("bindings/c++/test/smoke.cpp"), .flags = &.{"-std=c++17"} });
            smoke_mod.addIncludePath(b.path("bindings/c/include"));
            smoke_mod.addIncludePath(b.path("bindings/c++/include"));
            smoke_mod.linkLibrary(clib);

            const smoke = b.addExecutable(.{ .name = "loom-cpp-smoke", .root_module = smoke_mod });
            test_step.dependOn(&b.addRunArtifact(smoke).step);
        }

        if (want_py) {
            // nanobind extension `pyloom`, built the CMake-free way its
            // src/nb_combined.cpp documents: compile that unity TU + our module,
            // link the C ABI lib. Paths come from the Nix-provided python env.
            // Default the paths by asking python3 on PATH (as in `nix develop
            // .#rt`); the -D options override. b.run panics with a clear error if
            // python3 is absent, which is the right signal for this binding.
            const nb_root = nanobind_root orelse std.mem.trim(u8, b.run(&.{
                "python3",
                "-c",
                "import nanobind,os;print(os.path.dirname(nanobind.__file__))",
            }), " \r\n\t");
            const py_inc = python_include orelse std.mem.trim(u8, b.run(&.{
                "python3",
                "-c",
                "import sysconfig;print(sysconfig.get_path('include'))",
            }), " \r\n\t");

            const ext_mod = b.createModule(.{
                .target = target,
                .optimize = optimize,
                .link_libcpp = true,
            });
            const nb_flags = &[_][]const u8{
                "-std=c++17",
                "-fvisibility=hidden",
                "-fPIC",
                "-DNDEBUG",
                "-DNB_COMPACT_ASSERTIONS",
                "-fno-strict-aliasing",
            };
            const ext_flags = &[_][]const u8{
                "-std=c++17",
                "-fvisibility=hidden",
                "-fPIC",
                "-DNDEBUG",
            };
            ext_mod.addCSourceFile(.{
                .file = .{ .cwd_relative = b.pathJoin(&.{ nb_root, "src", "nb_combined.cpp" }) },
                .flags = nb_flags,
            });
            ext_mod.addCSourceFile(.{ .file = b.path("bindings/python/src/loom_ext.cpp"), .flags = ext_flags });
            ext_mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ nb_root, "include" }) });
            ext_mod.addIncludePath(.{ .cwd_relative = b.pathJoin(&.{ nb_root, "ext", "robin_map", "include" }) });
            ext_mod.addIncludePath(.{ .cwd_relative = py_inc });
            ext_mod.addIncludePath(b.path("bindings/c/include"));
            ext_mod.addIncludePath(b.path("bindings/c++/include"));
            ext_mod.linkLibrary(clib);

            const ext = b.addLibrary(.{ .linkage = .dynamic, .name = "pyloom", .root_module = ext_mod });

            // Install as a real python package: <prefix>/lib/pythonX.Y/site-packages/
            // loom/{__init__.py, pyloom.so}. `pyloom` is nested (imported as
            // loom.pyloom); its NB_MODULE name matches the last path component, and
            // the bare `.so` is importable via CPython's suffix fallback.
            const py_ver = std.mem.trim(u8, b.run(&.{
                "python3",
                "-c",
                "import sys;print(f'python{sys.version_info[0]}.{sys.version_info[1]}')",
            }), " \r\n\t");
            const site = b.pathJoin(&.{ "lib", py_ver, "site-packages" });
            const pkg = b.pathJoin(&.{ site, "loom" });

            const install_pkg = b.addInstallDirectory(.{
                .source_dir = b.path("bindings/python/loom"),
                .install_dir = .{ .custom = site },
                .install_subdir = "loom",
            });
            b.getInstallStep().dependOn(&install_pkg.step);

            const install_ext = b.addInstallFileWithDir(ext.getEmittedBin(), .{ .custom = pkg }, "pyloom.so");
            b.getInstallStep().dependOn(&install_ext.step);

            // For `zig build test`: stage the package (loom/__init__.py + pyloom.so)
            // in a WriteFiles dir and run pytest against it, so testing needs no
            // install into any prefix. The staged dir is passed as argv[1].
            const staged = b.addWriteFiles();
            _ = staged.addCopyDirectory(b.path("bindings/python/loom"), "loom", .{});
            _ = staged.addCopyFile(ext.getEmittedBin(), "loom/pyloom.so");
            const pytest = b.addSystemCommand(&.{
                "python3", "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import pytest; sys.exit(pytest.main([sys.argv[2], '-q']))",
            });
            pytest.addDirectoryArg(staged.getDirectory());
            pytest.addDirectoryArg(b.path("bindings/python/tests"));
            // Don't scatter __pycache__ into the source tree during the test run.
            pytest.setEnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1");
            test_step.dependOn(&pytest.step);
        }
    }

    const exe = b.addExecutable(.{
        .name = "loom-cli",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "loom", .module = loom_mod }},
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);

    const run_step = b.step("run", "Run the Loom CLI");
    run_step.dependOn(&run_cmd.step);

    for ([_][]const u8{
        "lib/loom/quant.zig",
        "lib/loom/protocol.zig",
        "lib/loom/fp.zig",
        "lib/loom/model.zig",
        "lib/loom/sim.zig",
        "lib/loom/linear.zig",
        "lib/loom/ops.zig",
        "lib/loom/forward.zig",
        "lib/loom/tokenizer.zig",
        "lib/loom/bpe.zig",
        "lib/loom/usb.zig",
        "lib/loom/serve.zig",
    }) |src| {
        const t = b.addTest(.{
            .root_module = b.createModule(.{
                .root_source_file = b.path(src),
                .target = target,
                .optimize = optimize,
            }),
        });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
