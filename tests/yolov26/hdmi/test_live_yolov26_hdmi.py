import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SCRIPTS = os.path.join(REPO, "scripts")
CORE = os.path.join(REPO, "inference", "yolov26", "core")
INFERENCE = os.path.join(REPO, "inference", "yolov26", "hdmi")
TESTS = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SCRIPTS)

import run_live_yolov26_hdmi as launcher

CORE_HEADER = os.path.join(CORE, "yolov26_realsense_core.hpp")
HDMI_SOURCE = os.path.join(INFERENCE, "live_yolov26_hdmi.cpp")


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


def extract_handoff():
    with open(CORE_HEADER) as handle:
        text = handle.read()
    start = text.index("enum class PublishResult")
    end = text.index("\n};", text.index("class LatestFrameHandoff"))
    return text[start:end + 3]


def extract_worker_seam():
    with open(HDMI_SOURCE) as handle:
        text = handle.read()
    start = text.index("struct WorkerGuard")
    end = text.index("\n}\n", text.index("start_capture_worker"))
    return text[start:end + 3]


DISPLAY_SOURCE = os.path.join(INFERENCE, "yolov26_hdmi_display.cpp")


def extract_window_visible():
    with open(DISPLAY_SOURCE) as handle:
        text = handle.read()
    start = text.index("bool window_visible(")
    end = text.index("\n}\n", start)
    return text[start:end + 3]


@unittest.skipUnless(shutil.which("g++"), "g++ unavailable")
class ControlFixtureTest(unittest.TestCase):
    def test_control_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "test_hdmi_control")
            compile_binary(
                [
                    os.path.join(
                        TESTS, "test_yolov26_hdmi_control.cpp"
                    ),
                    os.path.join(
                        INFERENCE, "yolov26_hdmi_control.cpp"
                    ),
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


def _pkg_config_exists(*modules):
    return bool(shutil.which("pkg-config")) and subprocess.run(
        ["pkg-config", "--exists", *modules]
    ).returncode == 0


@unittest.skipUnless(
    shutil.which("g++") and _pkg_config_exists("opencv4", "libdrm"),
    "opencv4/libdrm development files unavailable",
)
class KmsFixtureTest(unittest.TestCase):
    def test_kms_composition_fixture(self):
        flags = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "opencv4", "libdrm"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "test_kms_display")
            compile_binary(
                [
                    os.path.join(
                        TESTS, "test_yolov26_kms_display.cpp"
                    ),
                    os.path.join(
                        INFERENCE, "yolov26_kms_display.cpp"
                    ),
                    os.path.join(
                        INFERENCE, "yolov26_hdmi_control.cpp"
                    ),
                ],
                binary,
                flags,
            )
            result = subprocess.run(
                [binary], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(
                0, result.returncode,
                "%s\n%s" % (result.stdout, result.stderr),
            )


class LauncherActionTest(unittest.TestCase):
    def _patched_run(self, returncode=0, stdout=""):
        completed = subprocess.CompletedProcess(
            args=[], returncode=returncode,
            stdout=stdout, stderr="",
        )
        return mock.patch.object(
            launcher.subprocess, "run", return_value=completed
        )

    def test_check_display_never_execs_vision(self):
        executed = []

        def fake_execvpe(path, argv, env):
            executed.append(argv)

        with self._patched_run() as run, mock.patch.object(
            launcher.os, "execvpe", fake_execvpe
        ):
            rc = launcher.main(["--check-display"])
        self.assertEqual(0, rc)
        self.assertEqual([], executed)
        commands = [call.args[0] for call in run.call_args_list]
        joined = " ".join(
            arg for cmd in commands for arg in cmd
        )
        self.assertIn("xrandr", joined)
        self.assertIn("xset", joined)

    def test_wake_display_never_execs_vision(self):
        executed = []

        def fake_execvpe(path, argv, env):
            executed.append(argv)

        with self._patched_run() as run, mock.patch.object(
            launcher.os, "execvpe", fake_execvpe
        ):
            rc = launcher.main(["--wake-display"])
        self.assertEqual(0, rc)
        self.assertEqual([], executed)
        commands = [call.args[0] for call in run.call_args_list]
        joined = " ".join(
            arg for cmd in commands for arg in cmd
        )
        self.assertIn("dpms", joined)
        self.assertIn("force", joined)
        self.assertIn("on", joined)

    def test_check_display_failure_propagates(self):
        with self._patched_run(returncode=1) as run:
            run.return_value.stderr = "can't open display"
            rc = launcher.main(["--check-display"])
        self.assertNotEqual(0, rc)

    def test_missing_x_tool_friendly(self):
        import io
        from contextlib import redirect_stderr

        err = io.StringIO()
        with mock.patch.object(
            launcher.subprocess, "run",
            side_effect=FileNotFoundError("xrandr"),
        ), redirect_stderr(err):
            rc = launcher.main(["--check-display"])
        self.assertNotEqual(0, rc)
        self.assertNotIn("Traceback", err.getvalue())

    def test_display_override_env(self):
        envs = []

        def fake_run(argv, **kwargs):
            envs.append(kwargs.get("env"))
            return subprocess.CompletedProcess(
                args=argv, returncode=0, stdout="", stderr=""
            )

        with mock.patch.object(
            launcher.subprocess, "run", fake_run
        ):
            rc = launcher.main(
                ["--display", ":1", "--check-display"]
            )
        self.assertEqual(0, rc)
        self.assertTrue(envs)
        for env in envs:
            self.assertEqual(":1", env["DISPLAY"])

    def test_exec_forwards_args_and_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "hdmi")
            with open(binary, "w") as handle:
                handle.write("")
            captured = {}

            def fake_execvpe(path, argv, env):
                captured["path"] = path
                captured["argv"] = argv
                captured["env"] = env

            base_env = dict(os.environ)
            base_env["XAUTHORITY"] = "/run/user/0/xauth"

            with mock.patch.object(
                launcher.os, "execvpe", fake_execvpe
            ), mock.patch.dict(os.environ, base_env, clear=True):
                rc = launcher.main(
                    ["--binary", binary, "--display", ":0",
                     "--", "--test-pattern",
                     "--duration-seconds", "3"]
                )
            self.assertEqual(0, rc)
            self.assertEqual(binary, captured["path"])
            self.assertEqual(
                [binary, "--display", ":0", "--test-pattern",
                 "--duration-seconds", "3"],
                captured["argv"],
            )
            env = captured["env"]
            self.assertEqual(":0", env["DISPLAY"])
            self.assertEqual(
                "/run/user/0/xauth", env["XAUTHORITY"]
            )

    def test_exec_default_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "hdmi")
            with open(binary, "w") as handle:
                handle.write("")
            captured = {}

            def fake_execvpe(path, argv, env):
                captured["env"] = env

            with mock.patch.object(
                launcher.os, "execvpe", fake_execvpe
            ), mock.patch.dict(os.environ, {}, clear=True):
                rc = launcher.main(["--binary", binary])
            self.assertEqual(0, rc)
            self.assertEqual(":0", captured["env"]["DISPLAY"])

    def test_missing_binary(self):
        rc = launcher.main(
            ["--binary", "/nonexistent/hdmi"]
        )
        self.assertEqual(2, rc)

    def test_exec_oserror_is_friendly(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "hdmi")
            with open(binary, "w") as handle:
                handle.write("")

            def bad_execvpe(path, argv, env):
                raise OSError("bad format")

            import io
            from contextlib import redirect_stderr
            err = io.StringIO()
            with mock.patch.object(
                launcher.os, "execvpe", bad_execvpe
            ), redirect_stderr(err):
                rc = launcher.main(["--binary", binary])
            self.assertEqual(2, rc)
            self.assertIn("cannot exec", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    def test_display_tool_timeout_friendly(self):
        import io
        from contextlib import redirect_stderr

        err = io.StringIO()
        with mock.patch.object(
            launcher.subprocess, "run",
            side_effect=subprocess.TimeoutExpired("xset", 30),
        ), redirect_stderr(err):
            rc = launcher.main(["--check-display"])
        self.assertNotEqual(0, rc)
        self.assertNotIn("Traceback", err.getvalue())

    def test_actions_mutually_exclusive(self):
        rc = launcher.main(
            ["--check-display", "--wake-display"]
        )
        self.assertEqual(2, rc)

    def test_action_rejects_forwarded_args(self):
        rc = launcher.main(
            ["--check-display", "--", "--test-pattern"]
        )
        self.assertEqual(2, rc)

    def test_launcher_help_no_vision(self):
        executed = []

        def fake_execvpe(path, argv, env):
            executed.append(argv)

        with mock.patch.object(
            launcher.os, "execvpe", fake_execvpe
        ):
            rc = launcher.main(["--help"])
        self.assertEqual(0, rc)
        self.assertEqual([], executed)


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
        return bindir

    def _run_build(self, tmp, args, env_extra=None):
        env = dict(os.environ)
        env["HOME"] = tmp
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            ["sh", os.path.join(
                SCRIPTS, "build_live_yolov26_hdmi.sh"),
             *args],
            capture_output=True, text=True, env=env,
            cwd=REPO, timeout=60,
        )

    def _rs_prefix(self, tmp):
        rs = os.path.join(tmp, "rs")
        os.makedirs(os.path.join(rs, "include"))
        os.makedirs(os.path.join(rs, "lib64"))
        open(os.path.join(rs, "lib64", "librealsense2.so"),
             "w").close()
        return rs

    def test_compiler_invoked_with_expected_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs = self._rs_prefix(tmp)
            capture = {}
            bindir = self._fake_env(
                tmp, ["opencv4", "x11", "gtk+-3.0", "libdrm"], capture
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
            self.assertIn("live_yolov26_hdmi.cpp", log)
            self.assertIn("yolov26_hdmi_control.cpp", log)
            self.assertIn("yolov26_hdmi_display.cpp", log)
            self.assertIn("yolov26_kms_display.cpp", log)
            self.assertIn(
                "pkg-config --cflags --libs "
                "opencv4 x11 gtk+-3.0 libdrm",
                log,
            )
            self.assertIn("yolov26_realsense_core.cpp", log)
            self.assertIn("rpath", log)

    def test_missing_module_never_compiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            rs = self._rs_prefix(tmp)
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

    def test_missing_rs_prefix_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = {}
            bindir = self._fake_env(
                tmp, ["opencv4", "x11"], capture
            )
            env_extra = {"PATH": bindir + ":" + os.environ["PATH"]}
            result = self._run_build(
                tmp,
                ["--rs-prefix", os.path.join(tmp, "absent")],
                env_extra,
            )
            self.assertNotEqual(0, result.returncode)
            if os.path.exists(capture["log"]):
                with open(capture["log"]) as handle:
                    self.assertNotIn("GXX", handle.read())


class ScriptSyntaxTest(unittest.TestCase):
    def test_shell_scripts_parse(self):
        result = subprocess.run(
            ["sh", "-n", os.path.join(
                SCRIPTS, "build_live_yolov26_hdmi.sh")],
            capture_output=True, text=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_launcher_py39_stdlib(self):
        import ast
        path = os.path.join(SCRIPTS, "run_live_yolov26_hdmi.py")
        with open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
        banned = {"numpy", "cv2", "gi", "PyQt5"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(
                        alias.name.split(".")[0], banned
                    )
            elif isinstance(node, ast.ImportFrom):
                self.assertIsNotNone(node.module)
                self.assertNotIn(
                    node.module.split(".")[0], banned
                )


HANDOFF_DRIVER = r"""
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace yolov26 {

struct FramePacket {
    int sequence = 0;
};

%s

struct Options {
    bool test_pattern = false;
};

%s

}

using namespace yolov26;

static int failures = 0;

static void check(bool condition, const char* name)
{
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %%s\n", name);
    }
}

int main()
{
    {
        LatestFrameHandoff handoff;
        FramePacket packet;
        check(!handoff.try_pop(packet),
              "empty try_pop returns immediately");

        FramePacket first;
        first.sequence = 1;
        check(handoff.publish(std::move(first))
                  == PublishResult::published,
              "first publish");
        FramePacket second;
        second.sequence = 2;
        check(handoff.publish(std::move(second))
                  == PublishResult::replaced,
              "second publish replaces pending");
        check(handoff.try_pop(packet) && packet.sequence == 2,
              "try_pop returns latest frame");
        check(!handoff.try_pop(packet),
              "handoff empty after pop");
    }

    {
        LatestFrameHandoff handoff;
        FramePacket packet;
        handoff.stop();
        check(!handoff.wait_pop(packet),
              "wait_pop returns after stop");
        FramePacket late;
        late.sequence = 3;
        check(handoff.publish(std::move(late))
                  == PublishResult::stopped,
              "publish after stop rejected");
        check(!handoff.try_pop(packet),
              "try_pop after stop empty");
    }

    {
        Options options;
        options.test_pattern = true;
        std::atomic<bool> running{true};
        std::exception_ptr worker_error;
        bool synthetic_ran = false;
        bool real_ran = false;

        std::thread worker = start_capture_worker(
            options,
            running,
            worker_error,
            [&](std::atomic<bool>&) { synthetic_ran = true; },
            [&](std::atomic<bool>&) { real_ran = true; }
        );
        worker.join();
        check(synthetic_ran, "synthetic body invoked");
        check(!real_ran, "real body never invoked");
        check(!worker_error, "no worker error");
        check(!running.load(), "running cleared on exit");
    }

    {
        Options options;
        options.test_pattern = false;
        std::atomic<bool> running{true};
        std::exception_ptr worker_error;
        bool synthetic_ran = false;
        bool real_ran = false;

        std::thread worker = start_capture_worker(
            options,
            running,
            worker_error,
            [&](std::atomic<bool>&) { synthetic_ran = true; },
            [&](std::atomic<bool>& active) {
                check(active.load(), "real body sees running");
                real_ran = true;
            }
        );
        worker.join();
        check(real_ran, "real body invoked");
        check(!synthetic_ran, "synthetic never invoked in real");
    }

    {
        Options options;
        options.test_pattern = true;
        std::atomic<bool> running{true};
        std::exception_ptr worker_error;

        std::thread worker = start_capture_worker(
            options,
            running,
            worker_error,
            [&](std::atomic<bool>&) {
                throw std::runtime_error("producer failed");
            },
            [&](std::atomic<bool>&) {}
        );
        worker.join();
        check(worker_error != nullptr,
              "producer error propagated");
        check(!running.load(), "running cleared on error");
    }

    {
        Options options;
        options.test_pattern = true;
        std::atomic<bool> running{true};
        std::exception_ptr worker_error;
        LatestFrameHandoff handoff;
        std::thread worker;
        std::atomic<bool> producer_ticks{false};
        bool gui_threw = false;

        try {
            WorkerGuard guard{running, handoff, worker};
            worker = start_capture_worker(
                options,
                running,
                worker_error,
                [&](std::atomic<bool>& active) {
                    while (active.load()) {
                        FramePacket frame;
                        handoff.publish(std::move(frame));
                        producer_ticks.store(true);
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(1)
                        );
                    }
                },
                [&](std::atomic<bool>&) {}
            );
            const auto deadline =
                std::chrono::steady_clock::now()
                + std::chrono::seconds(5);
            while (!producer_ticks.load()
                && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1)
                );
            }
            throw std::runtime_error("gui failed");
        } catch (const std::runtime_error&) {
            gui_threw = true;
        }
        check(gui_threw, "gui exception caught");
        check(producer_ticks.load(), "producer ran before throw");
        check(!worker.joinable(), "guard joined worker on throw");
        check(!running.load(), "running cleared after throw");
    }

    {
        Options options;
        options.test_pattern = true;
        std::atomic<bool> running{true};
        std::exception_ptr worker_error;
        LatestFrameHandoff handoff;
        std::thread worker;
        std::atomic<bool> entered{false};

        {
            WorkerGuard guard{running, handoff, worker};
            worker = start_capture_worker(
                options,
                running,
                worker_error,
                [&](std::atomic<bool>& active) {
                    entered.store(true);
                    while (active.load()) {
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(1)
                        );
                    }
                },
                [&](std::atomic<bool>&) {}
            );
            const auto deadline =
                std::chrono::steady_clock::now()
                + std::chrono::seconds(5);
            while (!entered.load()
                && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(1)
                );
            }
        }
        check(entered.load(), "idle producer body entered");
        check(!running.load(), "running cleared by guard");
        check(!worker.joinable(), "guard joined idle worker");
    }

    {
        Options options;
        options.test_pattern = true;
        std::atomic<bool> running{true};
        std::exception_ptr worker_error;
        LatestFrameHandoff handoff;
        std::thread worker;
        std::atomic<bool> finished{false};

        {
            WorkerGuard guard{running, handoff, worker};
            worker = start_capture_worker(
                options,
                running,
                worker_error,
                [&](std::atomic<bool>&) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50)
                    );
                    throw std::runtime_error("late failure");
                },
                [&](std::atomic<bool>&) {}
            );
            guard.stop_and_join();
            finished.store(true);
        }
        check(finished.load(), "stop_and_join completed");
        check(worker_error != nullptr,
              "worker_error visible after stop_and_join");
        check(!running.load(), "running cleared after join");
    }

    {
        Options options;
        options.test_pattern = true;
        std::atomic<bool> running{true};
        std::exception_ptr worker_error;
        auto flag = std::make_shared<std::atomic<bool>>(false);
        std::weak_ptr<std::atomic<bool>> weak(flag);
        std::atomic<bool> release{false};

        std::thread worker = start_capture_worker(
            options,
            running,
            worker_error,
            [flag, &release](std::atomic<bool>& active) {
                while (active.load() && !release.load()) {
                    std::this_thread::yield();
                }
            },
            WorkerBody()
        );
        flag.reset();
        check(!weak.expired(), "copied body survives temporaries");
        release.store(true);
        worker.join();
        check(weak.expired(), "body released after join");
    }

    std::printf("handoff fixture done, %%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
"""


WINDOW_VISIBLE_DRIVER = r"""
#include <cstdio>

static void* g_handle = nullptr;
static bool g_throw = false;

void* cvGetWindowHandle(const char*)
{
    if (g_throw) {
        throw 1;
    }
    return g_handle;
}

%s

static int failures = 0;

static void check(bool condition, const char* name)
{
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %%s\n", name);
    }
}

int main()
{
    int marker = 0;
    g_handle = &marker;
    check(window_visible("w"), "live handle visible");
    g_handle = nullptr;
    check(!window_visible("w"), "null handle hidden");
    g_handle = &marker;
    g_throw = true;
    check(!window_visible("w"), "exception hidden");
    std::printf("window_visible fixture done, %%d failures\n",
                failures);
    return failures == 0 ? 0 : 1;
}
"""


@unittest.skipUnless(shutil.which("g++"), "g++ unavailable")
class WindowVisibleFixtureTest(unittest.TestCase):
    def test_window_visible_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "visible_fixture.cpp")
            with open(source, "w") as handle:
                handle.write(
                    WINDOW_VISIBLE_DRIVER % extract_window_visible()
                )
            binary = os.path.join(tmp, "visible_fixture")
            compile_binary([source], binary)
            result = subprocess.run(
                [binary], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(
                0, result.returncode,
                "%s\n%s" % (result.stdout, result.stderr),
            )


@unittest.skipUnless(shutil.which("g++"), "g++ unavailable")
class HandoffFixtureTest(unittest.TestCase):
    def test_handoff_and_source_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "handoff_fixture.cpp")
            with open(source, "w") as handle:
                handle.write(
                    HANDOFF_DRIVER
                    % (extract_handoff(), extract_worker_seam())
                )
            binary = os.path.join(tmp, "handoff_fixture")
            compile_binary([source], binary)
            result = subprocess.run(
                [binary], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(
                0, result.returncode,
                "%s\n%s" % (result.stdout, result.stderr),
            )


class SourceBoundaryTest(unittest.TestCase):
    def test_acquire_window_covers_camera_wait(self):
        with open(HDMI_SOURCE) as handle:
            text = handle.read()
        i_start = text.index(
            "const auto frame_start = Clock::now();"
        )
        i_wait = text.index("try_wait_for_frames", i_start)
        i_acquire = text.index(
            "const auto acquire_end = Clock::now();", i_wait
        )
        i_infer = text.index("infer_frame(", i_acquire)
        self.assertLess(i_start, i_wait)
        self.assertLess(i_wait, i_acquire)
        self.assertLess(i_acquire, i_infer)
        call = text[i_infer:text.index(");", i_infer)]
        self.assertIn("frame_start", call)
        self.assertIn("acquire_end", call.rsplit(",", 1)[-1])

    def test_synthetic_worker_never_stats_model(self):
        with open(HDMI_SOURCE) as handle:
            text = handle.read()
        i_synth = text.index("const WorkerBody synthetic_body")
        i_real = text.index("const WorkerBody real_body")
        i_stat = text.index("::stat(")
        i_dispatch = text.index("worker = start_capture_worker(")
        self.assertLess(i_synth, i_real)
        self.assertLess(i_real, i_stat)
        self.assertLess(i_stat, i_dispatch)


if __name__ == "__main__":
    unittest.main()
