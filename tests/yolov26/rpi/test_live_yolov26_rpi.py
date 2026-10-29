import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS = os.path.join(REPO, "scripts")
CORE = os.path.join(REPO, "inference", "yolov26", "core")
INFERENCE = os.path.join(REPO, "inference", "yolov26", "rpi")
TESTS = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SCRIPTS)

import run_live_yolov26_rpi as launcher


def have_pkg_config(*modules):
    if shutil.which("pkg-config") is None:
        return False
    for module in modules:
        rc = subprocess.call(
            ["pkg-config", "--exists", module],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if rc != 0:
            return False
    return True


def pkg_config_flags(*modules):
    output = subprocess.check_output(
        ["pkg-config", "--cflags", "--libs", *modules],
        text=True,
    )
    return output.split()


def compile_binary(sources, output, extra_args=()):
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        *sources, "-I", CORE, "-I", INFERENCE, *extra_args, "-pthread",
        "-o", output,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "compile failed:\n%s\n%s" % (result.stdout, result.stderr)
        )


@unittest.skipUnless(shutil.which("g++"), "g++ unavailable")
class ControlFixtureTest(unittest.TestCase):
    def test_control_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "test_control")
            compile_binary(
                [
                    os.path.join(TESTS, "test_yolov26_rpi_control.cpp"),
                    os.path.join(INFERENCE, "yolov26_rpi_control.cpp"),
                ],
                binary,
            )
            result = subprocess.run(
                [binary], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(
                0, result.returncode,
                "%s\n%s" % (result.stdout, result.stderr),
            )


@unittest.skipUnless(
    shutil.which("g++")
    and have_pkg_config(
        "gstreamer-1.0", "gstreamer-app-1.0",
        "gstreamer-video-1.0", "hiredis"
    ),
    "native transport deps unavailable",
)
class TransportFixtureTest(unittest.TestCase):
    def test_transport_fixture(self):
        flags = pkg_config_flags(
            "gstreamer-1.0", "gstreamer-app-1.0",
            "gstreamer-video-1.0", "hiredis"
        )
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "test_transport")
            compile_binary(
                [
                    os.path.join(
                        TESTS, "test_yolov26_rpi_transport.cpp"
                    ),
                    os.path.join(
                        INFERENCE, "yolov26_rpi_transport.cpp"
                    ),
                    os.path.join(
                        INFERENCE, "yolov26_rpi_control.cpp"
                    ),
                ],
                binary,
                extra_args=flags,
            )
            result = subprocess.run(
                [binary],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                0, result.returncode,
                "%s\n%s" % (result.stdout, result.stderr),
            )


class LauncherTest(unittest.TestCase):
    def test_missing_binary(self):
        rc = launcher.main(
            ["--binary", "/nonexistent/path",
             "--", "--pi-host", "127.0.0.1"]
        )
        self.assertEqual(2, rc)

    def test_bad_deps_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = launcher.main(
                ["--deps-prefix", tmp, "--binary", "/bin/true"]
            )
        self.assertEqual(2, rc)

    def test_execvpe_env_and_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugins = os.path.join(tmp, "lib", "gstreamer-1.0")
            os.makedirs(plugins)
            binary = os.path.join(tmp, "sender")
            with open(binary, "w") as handle:
                handle.write("")
            captured = {}

            def fake_execvpe(path, argv, env):
                captured["path"] = path
                captured["argv"] = argv
                captured["env"] = env

            with mock.patch.object(
                launcher.os, "execvpe", fake_execvpe
            ):
                rc = launcher.main(
                    ["--deps-prefix", tmp, "--binary", binary,
                     "--", "--pi-host", "192.168.1.2",
                     "--video-port", "5002"]
                )
            self.assertEqual(0, rc)
            self.assertEqual(binary, captured["path"])
            self.assertEqual(
                [binary, "--pi-host", "192.168.1.2",
                 "--video-port", "5002"],
                captured["argv"],
            )
            env = captured["env"]
            self.assertTrue(
                env["LD_LIBRARY_PATH"].startswith(
                    os.path.join(tmp, "lib")
                )
            )
            self.assertIn(
                plugins, env["GST_PLUGIN_PATH_1_0"].split(os.pathsep)
            )
            self.assertTrue(
                env["GST_REGISTRY_1_0"].startswith(tmp)
            )

    def _fixture_prefix_and_binary(self, tmp):
        os.makedirs(os.path.join(tmp, "lib", "gstreamer-1.0"))
        binary = os.path.join(tmp, "sender")
        with open(binary, "w") as handle:
            handle.write("")
        return binary

    def test_no_delimiter_forwarding(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._fixture_prefix_and_binary(tmp)
            captured = {}

            def fake_execvpe(path, argv, env):
                captured["argv"] = argv

            with mock.patch.object(
                launcher.os, "execvpe", fake_execvpe
            ):
                rc = launcher.main(
                    ["--deps-prefix", tmp, "--binary", binary,
                     "--pi-host", "192.168.1.2",
                     "--encoder", "x264",
                     "--video-port", "5002"]
                )
            self.assertEqual(0, rc)
            self.assertEqual(
                [binary, "--pi-host", "192.168.1.2",
                 "--encoder", "x264", "--video-port", "5002"],
                captured["argv"],
            )

    def test_help_forwards_to_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._fixture_prefix_and_binary(tmp)
            captured = {}

            def fake_execvpe(path, argv, env):
                captured["argv"] = argv

            with mock.patch.object(
                launcher.os, "execvpe", fake_execvpe
            ):
                rc = launcher.main(
                    ["--deps-prefix", tmp, "--binary", binary,
                     "--help"]
                )
            self.assertEqual(0, rc)
            self.assertEqual([binary, "--help"], captured["argv"])

    def test_list_profiles_forwards(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._fixture_prefix_and_binary(tmp)
            captured = {}

            def fake_execvpe(path, argv, env):
                captured["argv"] = argv

            with mock.patch.object(
                launcher.os, "execvpe", fake_execvpe
            ):
                rc = launcher.main(
                    ["--deps-prefix", tmp, "--binary", binary,
                     "--list-profiles"]
                )
            self.assertEqual(0, rc)
            self.assertEqual(
                [binary, "--list-profiles"], captured["argv"]
            )

    def test_exec_oserror_is_friendly(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._fixture_prefix_and_binary(tmp)

            def bad_execvpe(path, argv, env):
                raise OSError("bad format")

            import io
            from contextlib import redirect_stderr
            err = io.StringIO()
            with mock.patch.object(
                launcher.os, "execvpe", bad_execvpe
            ), redirect_stderr(err):
                rc = launcher.main(
                    ["--deps-prefix", tmp, "--binary", binary,
                     "--pi-host", "h"]
                )
            self.assertEqual(2, rc)
            self.assertIn("cannot exec", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_caller_env_unmutated(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._fixture_prefix_and_binary(tmp)
            before = dict(os.environ)

            def fake_execvpe(path, argv, env):
                pass

            with mock.patch.object(
                launcher.os, "execvpe", fake_execvpe
            ):
                launcher.main(
                    ["--deps-prefix", tmp, "--binary", binary,
                     "--pi-host", "h"]
                )
            self.assertEqual(before, dict(os.environ))

    def test_nonexecutable_binary_friendly(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._fixture_prefix_and_binary(tmp)
            import io
            from contextlib import redirect_stderr
            err = io.StringIO()
            with redirect_stderr(err):
                rc = launcher.main(
                    ["--deps-prefix", tmp, "--binary", binary,
                     "--pi-host", "h"]
                )
            self.assertEqual(2, rc)
            self.assertNotIn("Traceback", err.getvalue())


class BuildScriptTest(unittest.TestCase):
    def _fake_env(self, tmp, modules, capture):
        bindir = os.path.join(tmp, "bin")
        os.makedirs(bindir)
        log = os.path.join(tmp, "invocations.log")

        pkg_config = os.path.join(bindir, "pkg-config")
        with open(pkg_config, "w") as handle:
            handle.write(
                "#!/bin/sh\n"
                "echo \"pkg-config $@\" >> \"%s\"\n"
                "if [ \"$1\" = \"--exists\" ]; then\n"
                "  shift\n"
                "  for m in \"$@\"; do\n"
                "    grep -qxF \"$m\" \"%s\" || exit 1\n"
                "  done\n"
                "  exit 0\n"
                "fi\n"
                "echo -n \" -Ifake -lfake \"\n"
                % (log, os.path.join(tmp, "modules"))
            )
        with open(os.path.join(tmp, "modules"), "w") as handle:
            handle.write("\n".join(modules) + "\n")

        gxx = os.path.join(bindir, "g++")
        with open(gxx, "w") as handle:
            handle.write(
                "#!/bin/sh\n"
                "echo \"GXX $@\" >> \"%s\"\n" % log
            )
        for path in (pkg_config, gxx):
            os.chmod(path, 0o755)
        capture["log"] = log
        capture["bindir"] = bindir
        return bindir

    def _run_build(self, tmp, args, env_extra=None):
        env = dict(os.environ)
        env["HOME"] = tmp
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            ["sh", os.path.join(SCRIPTS, "build_live_yolov26_rpi.sh"),
             *args],
            capture_output=True, text=True, env=env,
            cwd=REPO, timeout=60,
        )

    def test_auto_prefix_and_compiler_invoked(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = os.path.join(tmp, "kv260-network-deps")
            os.makedirs(os.path.join(deps, "lib", "gstreamer-1.0"))
            os.makedirs(os.path.join(deps, "lib", "pkgconfig"))
            rs = os.path.join(tmp, "rs")
            os.makedirs(os.path.join(rs, "include"))
            os.makedirs(os.path.join(rs, "lib64"))
            open(os.path.join(rs, "lib64", "librealsense2.so"),
                 "w").close()
            capture = {}
            bindir = self._fake_env(
                tmp,
                ["opencv4", "gstreamer-1.0", "gstreamer-app-1.0",
                 "gstreamer-video-1.0", "hiredis"],
                capture,
            )
            env_extra = {"PATH": bindir + ":" + os.environ["PATH"]}
            result = self._run_build(
                tmp, ["--rs-prefix", rs], env_extra
            )
            self.assertEqual(
                0, result.returncode,
                result.stdout + result.stderr,
            )
            with open(capture["log"]) as handle:
                log = handle.read()
            self.assertIn("GXX", log)
            self.assertIn(deps, result.stdout)
            self.assertIn("rpath", log)

    def test_missing_module_never_compiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            deps = os.path.join(tmp, "kv260-network-deps")
            os.makedirs(os.path.join(deps, "lib", "gstreamer-1.0"))
            rs = os.path.join(tmp, "rs")
            os.makedirs(os.path.join(rs, "include"))
            os.makedirs(os.path.join(rs, "lib"))
            open(os.path.join(rs, "lib", "librealsense2.so"),
                 "w").close()
            capture = {}
            bindir = self._fake_env(tmp, ["opencv4"], capture)
            env_extra = {"PATH": bindir + ":" + os.environ["PATH"]}
            result = self._run_build(
                tmp, ["--rs-prefix", rs], env_extra
            )
            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing pkg-config modules",
                          result.stderr)
            with open(capture["log"]) as handle:
                self.assertNotIn("GXX", handle.read())

    def test_no_deps_prefix_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_build(tmp, [])
            self.assertEqual(1, result.returncode)
            self.assertIn("no dependency prefix", result.stderr)


class ScriptSyntaxTest(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for name in (
            "build_live_yolov26.sh",
            "build_live_yolov26_rpi.sh",
        ):
            result = subprocess.run(
                ["sh", "-n", os.path.join(SCRIPTS, name)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                0, result.returncode, result.stderr
            )


if __name__ == "__main__":
    unittest.main()
