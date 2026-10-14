import ast
import os
import socket
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import mock_yolov26_sender as sender


def parse(argv):
    return sender.build_arg_parser().parse_args(argv)


def args_for(**overrides):
    argv = ["--pi-host", "192.0.2.10"]
    for key, value in overrides.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    parsed = parse(argv)
    if parsed.redis_host is None:
        parsed.redis_host = parsed.pi_host
    return parsed


class FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, delta):
        self.now += delta


class FakeStop:
    def __init__(self, clock=None):
        self._set = False
        self.clock = clock
        self.on_wait = None

    def is_set(self):
        return self._set

    def set(self):
        self._set = True

    def wait(self, timeout=None):
        if self.clock is not None and timeout:
            self.clock.advance(timeout)
        if self.on_wait is not None:
            self.on_wait(timeout)
        return self._set


class FakeSocket:
    def __init__(self, chunks=(), send_error=None, recv_error=None,
                 clock=None):
        self.chunks = list(chunks)
        self.sent = b""
        self.sent_times = []
        self.closed = False
        self.send_error = send_error
        self.recv_error = recv_error
        self.timeout = None
        self.clock = clock

    def settimeout(self, value):
        self.timeout = value

    def sendall(self, data):
        if self.send_error:
            raise self.send_error
        if self.clock is not None:
            self.sent_times.append(self.clock.now)
        self.sent += data

    def recv(self, count):
        if self.recv_error:
            raise self.recv_error
        if not self.chunks:
            return b""
        return self.chunks.pop(0)

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self, exit_after_polls=None, code=0, hang_on_wait=False):
        self.code = code
        self.exit_after_polls = exit_after_polls
        self.polls = 0
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.waited = False
        self.hang_on_wait = hang_on_wait

    def poll(self):
        self.polls += 1
        if self.exit_after_polls is not None and \
                self.polls >= self.exit_after_polls:
            self.returncode = self.code
            return self.code
        return None

    def terminate(self):
        self.terminated = True
        if not self.hang_on_wait:
            self.code = -15
            self.exit_after_polls = 0
            self.returncode = self.code

    def kill(self):
        self.killed = True
        self.code = -9
        self.exit_after_polls = 0
        self.returncode = self.code

    def wait(self, timeout=None):
        self.waited = True
        if self.hang_on_wait and self.exit_after_polls is None:
            raise subprocess.TimeoutExpired("x", timeout)
        self.returncode = self.code
        return self.code

    def communicate(self, timeout=None):
        return b"out", None


def fresh_state(clock, detected=True):
    state = sender.FaceState()
    real_snapshot = state.snapshot

    def snapshot():
        snap = real_snapshot()
        if snap is None:
            return snap
        return snap[0], clock.now

    state.update(detected, clock.now)
    state.snapshot = snapshot
    return state


def run_publisher(state, a, clock, connect, max_waits,
                  replies=None):
    stop = FakeStop(clock)
    abort = threading.Event()
    logs = []
    count = {"n": 0}

    def on_wait(timeout):
        count["n"] += 1
        if count["n"] >= max_waits:
            stop.set()

    stop.on_wait = on_wait
    sender.publisher_loop(stop, abort, state, a, clock.now, logs.append,
                          clock=clock, connect=connect)
    return logs, abort, count["n"]


def redis_args(**overrides):
    base = dict(redis_interval_seconds=0.5)
    base.update(overrides)
    a = args_for(**base)
    a.redis_host = "127.0.0.1"
    return a


class ArgParseTests(unittest.TestCase):
    def test_defaults(self):
        a = args_for()
        self.assertEqual(a.video_port, 5000)
        self.assertEqual(a.encoder, "auto")
        self.assertEqual(a.bitrate_kbps, 4000)
        self.assertEqual(a.redis_host, "192.0.2.10")
        self.assertEqual(a.redis_port, 6379)
        self.assertEqual(a.redis_channel, "face_detection")
        self.assertEqual(a.redis_interval_seconds, 1.0)
        self.assertEqual(a.face_stale_seconds, 2.0)
        self.assertEqual(a.redis_timeout_ms, 200)
        self.assertEqual(a.pattern, "snow")
        self.assertEqual(a.face_mode, "random")
        self.assertEqual(a.width, 848)
        self.assertEqual(a.height, 480)
        self.assertEqual(a.fps, 30)
        self.assertIsNone(a.seed)
        self.assertIsNone(a.duration_seconds)

    def test_overrides(self):
        a = args_for(video_port=5001, encoder="x264", bitrate_kbps=800,
                     redis_host="::1", redis_port=6380,
                     redis_channel="k\u00e4nal",
                     redis_interval_seconds=0.5,
                     face_stale_seconds=3, redis_timeout_ms=500,
                     pattern="ball", face_mode="alternate", seed=7,
                     duration_seconds=2.5, width=640, height=360,
                     fps=15)
        self.assertEqual(a.encoder, "x264")
        self.assertEqual(a.redis_host, "::1")
        self.assertEqual(a.redis_channel, "k\u00e4nal")
        self.assertEqual(a.width, 640)
        self.assertEqual(a.height, 360)
        self.assertEqual(a.fps, 15)

    def test_no_subprocess_on_bad_args_or_help(self):
        with mock.patch.object(
            subprocess, "Popen",
            side_effect=AssertionError("spawned"),
        ):
            bad = [
                ["--pi-host"], [],
                ["--pi-host", "h", "--video-port", "0"],
                ["--pi-host", "h", "--video-port", "65536"],
                ["--pi-host", "h", "--video-port", "abc"],
                ["--pi-host", "h", "--bitrate-kbps", "0"],
                ["--pi-host", "h", "--bitrate-kbps", "-1"],
                ["--pi-host", "h", "--redis-timeout-ms", "0"],
                ["--pi-host", "h", "--redis-timeout-ms", "10001"],
                ["--pi-host", "h", "--encoder", "jpeg"],
                ["--pi-host", "h", "--pattern", "smpte"],
                ["--pi-host", "h", "--face-mode", "maybe"],
                ["--pi-host", "h", "--redis-interval-seconds", "nan"],
                ["--pi-host", "h", "--redis-interval-seconds", "inf"],
                ["--pi-host", "h", "--redis-interval-seconds", "0"],
                ["--pi-host", "h", "--redis-interval-seconds", "0.05"],
                ["--pi-host", "h", "--face-stale-seconds", "0"],
                ["--pi-host", "h", "--face-stale-seconds", "-2"],
                ["--pi-host", "h", "--redis-channel", ""],
                ["--pi-host", ""],
                ["--pi-host", "h", "--duration-seconds", "0"],
                ["--pi-host", "h", "--duration-seconds", "nan"],
                ["--pi-host", "h", "--width", "0"],
                ["--pi-host", "h", "--width", "641"],
                ["--pi-host", "h", "--width", "8194"],
                ["--pi-host", "h", "--height", "0"],
                ["--pi-host", "h", "--height", "481"],
                ["--pi-host", "h", "--height", "8194"],
                ["--pi-host", "h", "--fps", "0"],
                ["--pi-host", "h", "--fps", "241"],
                ["--pi-host", "h", "--fps", "29.5"],
                ["--pi-host", "h", "--fps", "nan"],
            ]
            for argv in bad:
                with self.assertRaises(SystemExit, msg=str(argv)):
                    parse(argv)
            with self.assertRaises(SystemExit) as ctx:
                parse(["--help"])
            self.assertEqual(ctx.exception.code, 0)

    def test_host_validation(self):
        good = ["127.0.0.1", "192.168.1.50", "::1", "fe80::1",
                "fe80::1%eth0", "pi.local", "cam-01.example.com",
                "host_name", "a.b.c."]
        for host in good:
            self.assertEqual(parse(["--pi-host", host]).pi_host, host)
        bad = ["bad host", "host|id", "host;id", "host`id`", "host'id'",
               "host$(id)", "host\nid", "host id", "pi.local\n",
               "pi.local\r", "pi.local\t", "pi.local\r\n",
               "pi.local\n\t", "999.1.1.1", "-bad-.com", "bad..name",
               "x" * 300, "fe80::1%bad;id", "fe80::1%two scopes",
               "fe80::1%", "fe80::1%a%b", "fe80::1%eth0\n",
               "\u00e4host", "10.0.0.1,"]
        for host in bad:
            with self.assertRaises(SystemExit, msg=repr(host)):
                parse(["--pi-host", host])


class DepsPrefixTests(unittest.TestCase):
    def make_prefix(self, tmp):
        p = Path(tmp) / "deps"
        (p / "lib" / "gstreamer-1.0").mkdir(parents=True)
        return p

    def test_explicit_valid_and_invalid(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self.make_prefix(tmp)
            self.assertEqual(sender.resolve_deps_prefix(str(p)), p)
            with self.assertRaises(sender.DependencyError):
                sender.resolve_deps_prefix(tmp)
            with self.assertRaises(sender.DependencyError):
                sender.resolve_deps_prefix(str(Path(tmp) / "nope"))

    def test_auto_discovery(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            with mock.patch.object(Path, "home", return_value=home):
                self.assertIsNone(sender.resolve_deps_prefix(None))
                (home / "kv260-network-deps" / "lib"
                 / "gstreamer-1.0").mkdir(parents=True)
                self.assertEqual(
                    sender.resolve_deps_prefix(None),
                    home / "kv260-network-deps",
                )

    def test_child_env(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self.make_prefix(tmp)
            plug = str(p / "lib" / "gstreamer-1.0")
            lib = str(p / "lib")
            base = {
                "PATH": "/usr/bin",
                "LD_LIBRARY_PATH": lib + os.pathsep + "/other"
                                   + os.pathsep + lib,
                "GST_PLUGIN_PATH": "/sys/plugins",
                "PKG_CONFIG_PATH": "/keep/me",
            }
            env = sender.build_child_env(base, p)
            self.assertEqual(env["LD_LIBRARY_PATH"],
                             lib + os.pathsep + "/other")
            self.assertEqual(env["GST_PLUGIN_PATH"],
                             plug + os.pathsep + "/sys/plugins")
            self.assertEqual(env["GST_PLUGIN_PATH_1_0"],
                             plug + os.pathsep + "/sys/plugins")
            self.assertEqual(env["PKG_CONFIG_PATH"], "/keep/me")
            self.assertIn("registry-", env["GST_REGISTRY"])
            self.assertEqual(env["GST_REGISTRY"],
                             env["GST_REGISTRY_1_0"])
            self.assertTrue(env["GST_REGISTRY"].startswith(str(p)))
            self.assertEqual(
                base["LD_LIBRARY_PATH"].count(lib), 2)
            self.assertNotIn("GST_PLUGIN_PATH_1_0", base)
            self.assertNotIn("GST_REGISTRY", base)

    def test_child_env_versioned_and_registry_preserved(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self.make_prefix(tmp)
            plug = str(p / "lib" / "gstreamer-1.0")
            base = {
                "GST_PLUGIN_PATH_1_0": "/v/plugin",
                "GST_REGISTRY": "/caller/reg.bin",
            }
            env = sender.build_child_env(base, p)
            expected = plug + os.pathsep + "/v/plugin"
            self.assertEqual(env["GST_PLUGIN_PATH"], expected)
            self.assertEqual(env["GST_PLUGIN_PATH_1_0"], expected)
            self.assertEqual(env["GST_REGISTRY"], "/caller/reg.bin")
            self.assertEqual(env["GST_REGISTRY_1_0"], "/caller/reg.bin")

    def test_child_env_no_prefix_is_copy(self):
        base = {"A": "1"}
        env = sender.build_child_env(base, None)
        self.assertEqual(env, base)
        self.assertIsNot(env, base)

    def test_env_passed_to_all_subprocesses(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p = self.make_prefix(tmp)
            env = sender.build_child_env({}, p)
            seen = []

            def fake_runner(argv, env_, timeout, stop):
                seen.append((argv, env_))
                return 0, ""

            a = args_for(encoder="x264")
            stop = threading.Event()

            def exists(env_, el, st):
                return True

            def smoker(env_, enc, args_, st):
                fake_runner(["smoke"], env_, 5, st)

            enc = sender.select_encoder(env, a, stop, lambda m: None,
                                        exists=exists, smoker=smoker)
            self.assertEqual(enc, "x264")
            self.assertEqual(seen[0][1], env)


class PipelineTests(unittest.TestCase):
    def test_exact_main_pipeline(self):
        a = args_for(bitrate_kbps=4000, video_port=5000)
        argv = sender.build_pipeline_argv(a, "x264", main_sink=True)
        joined = " ".join(argv)
        for needle in [
            "videotestsrc is-live=true pattern=snow",
            "video/x-raw,format=BGR,width=848,height=480,"
            "framerate=30/1",
            "queue max-size-buffers=1 max-size-bytes=0 "
            "max-size-time=0 leaky=downstream",
            "videoconvert", "video/x-raw,format=NV12",
            "x264enc tune=zerolatency speed-preset=ultrafast "
            "bitrate=4000 key-int-max=30 bframes=0 rc-lookahead=0",
            "video/x-h264,profile=baseline", "h264parse",
            "video/x-h264,alignment=au,stream-format=byte-stream",
            "rtph264pay pt=96 config-interval=1 mtu=1200",
            "udpsink host=192.0.2.10 port=5000 sync=false "
            "async=false",
        ]:
            self.assertIn(needle, joined)
        self.assertNotIn("num-buffers", joined)
        self.assertNotIn("appsrc", joined)
        self.assertNotIn("latency-mode", joined)

    def test_custom_dimensions_pipeline(self):
        a = args_for(width=640, height=360, fps=15)
        joined = " ".join(
            sender.build_pipeline_argv(a, "x264", main_sink=True))
        self.assertIn(
            "video/x-raw,format=BGR,width=640,height=360,"
            "framerate=15/1", joined)
        self.assertIn("rtph264pay pt=96", joined)
        self.assertIn("video/x-h264,profile=baseline", joined)

    def test_warmup_pipeline_fakesink_not_udp(self):
        a = args_for()
        argv = sender.build_pipeline_argv(a, "x264", main_sink=False)
        joined = " ".join(argv)
        self.assertIn("num-buffers=1", joined)
        self.assertNotIn("is-live", joined)
        self.assertIn("fakesink sync=false", joined)
        self.assertNotIn("udpsink", joined)

    def test_omx_pipeline(self):
        a = args_for(bitrate_kbps=2000)
        joined = " ".join(
            sender.build_pipeline_argv(a, "omx", main_sink=True))
        self.assertIn(
            "omxh264enc target-bitrate=2000 control-rate=low-latency "
            "b-frames=0 look-ahead=0 gop-mode=basic gop-length=30 "
            "periodicity-idr=30", joined)
        self.assertNotIn("latency-mode", joined)

    def test_ball_pattern(self):
        a = args_for(pattern="ball")
        joined = " ".join(
            sender.build_pipeline_argv(a, "x264", main_sink=True))
        self.assertIn("pattern=ball", joined)


class EncoderSelectTests(unittest.TestCase):
    def all_common(self, *encoders):
        m = {f: True for f in sender.COMMON_FACTORIES}
        for e in encoders:
            m[e] = True
        return m

    def run_select(self, exists_map, smoke_fail=(), encoder="auto",
                   stop=None):
        a = args_for(encoder=encoder)
        stop = stop or threading.Event()
        logs = []
        exists = lambda env, el, st: exists_map.get(el, False)

        def smoker(env, enc, args_, st):
            if enc in smoke_fail:
                raise sender.DependencyError("boom %s" % enc)

        return sender.select_encoder({}, a, stop, logs.append,
                                     exists=exists, smoker=smoker), logs

    def test_auto_prefers_omx(self):
        enc, _ = self.run_select(
            self.all_common("omxh264enc", "x264enc"))
        self.assertEqual(enc, "omx")

    def test_auto_omx_fail_falls_back_x264(self):
        enc, logs = self.run_select(
            self.all_common("omxh264enc", "x264enc"),
            smoke_fail=("omx",))
        self.assertEqual(enc, "x264")
        self.assertTrue(any("omxh264enc" in line for line in logs))

    def test_auto_x264_when_no_omx(self):
        enc, _ = self.run_select(self.all_common("x264enc"))
        self.assertEqual(enc, "x264")

    def test_auto_none_available(self):
        with self.assertRaises(sender.DependencyError):
            self.run_select(self.all_common())

    def test_missing_common_factory(self):
        m = self.all_common("x264enc")
        del m["queue"]
        with self.assertRaises(sender.DependencyError):
            self.run_select(m)

    def test_forced_missing(self):
        m = self.all_common("x264enc")
        with self.assertRaises(sender.DependencyError):
            self.run_select(m, encoder="omx")

    def test_forced_no_switch(self):
        m = self.all_common("x264enc", "omxh264enc")
        with self.assertRaises(sender.DependencyError):
            self.run_select(m, smoke_fail=("omx",), encoder="omx")

    def test_forced_x264_ok(self):
        enc, _ = self.run_select(self.all_common("x264enc"),
                                 encoder="x264")
        self.assertEqual(enc, "x264")

    def test_stop_during_select_cancels(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(sender.DependencyError):
            self.run_select(self.all_common("x264enc"), stop=stop)


class RunManagedTests(unittest.TestCase):
    def test_no_spawn_when_already_stopped(self):
        stop = threading.Event()
        stop.set()
        with mock.patch.object(
            subprocess, "Popen",
            side_effect=AssertionError("spawned"),
        ):
            with self.assertRaises(sender.DependencyError):
                sender.run_managed(["x"], {}, 5.0, stop)

    def test_real_subprocess_large_output(self):
        stop = threading.Event()
        code, tail = sender.run_managed(
            [sys.executable, "-c",
             "import sys; sys.stdout.write('x' * 200000); "
             "sys.stdout.flush()"],
            {}, 2.0, stop)
        self.assertEqual(code, 0)
        self.assertLessEqual(len(tail), 2000)
        self.assertEqual(set(tail), {"x"})

    def test_success_fake(self):
        proc = FakeProc(exit_after_polls=1, code=0)
        seen = {}
        real_popen = subprocess.Popen

        def fake_popen(argv, **kwargs):
            seen["kwargs"] = kwargs
            return proc

        with mock.patch.object(subprocess, "Popen", fake_popen):
            code, tail = sender.run_managed(
                ["x"], {}, 5.0, threading.Event())
        self.assertEqual(code, 0)

    def test_timeout_kills_and_waits(self):
        proc = FakeProc(exit_after_polls=None, hang_on_wait=True)
        with mock.patch.object(subprocess, "Popen",
                               return_value=proc):
            clock = FakeClock()
            with self.assertRaises(sender.DependencyError) as ctx:
                sender.run_managed(["x"], {}, -1.0,
                                   threading.Event(), clock=clock)
        self.assertIn("timeout", str(ctx.exception))
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        self.assertTrue(proc.waited)

    def test_cancelled_terminates(self):
        proc = FakeProc(exit_after_polls=None)
        stop = threading.Event()

        def poll_and_stop():
            stop.set()
            return None

        with mock.patch.object(subprocess, "Popen",
                               return_value=proc):
            proc_poll = proc.poll
            proc.poll = lambda: (stop.set(), proc_poll())[1]
            with self.assertRaises(sender.DependencyError) as ctx:
                sender.run_managed(["x"], {}, 5.0, stop)
        self.assertIn("cancelled", str(ctx.exception))
        self.assertTrue(proc.terminated or proc.killed)


class RespTests(unittest.TestCase):
    def test_request_bytes(self):
        req = sender.encode_publish_request(
            "face_detection", b'{"face_detected":true}')
        expected = (
            b"*3\r\n$7\r\nPUBLISH\r\n$14\r\nface_detection\r\n"
            b"$22\r\n{\"face_detected\":true}\r\n"
        )
        self.assertEqual(req, expected)

    def test_unicode_channel_framing(self):
        req = sender.encode_publish_request(
            "k\u00e4na\r\nl", b'{"face_detected":false}')
        chan = "k\u00e4na\r\nl".encode("utf-8")
        self.assertIn(b"$%d\r\n%s\r\n" % (len(chan), chan), req)
        head, _, _ = req.partition(chan)
        self.assertTrue(head.endswith(b"\r\n"))

    def test_exact_json(self):
        self.assertEqual(sender.face_payload(True),
                         b'{"face_detected":true}')
        self.assertEqual(sender.face_payload(False),
                         b'{"face_detected":false}')
        for bad in (1, 0, "true", None):
            with self.assertRaises(TypeError):
                sender.face_payload(bad)

    def make_conn(self, sock):
        conn = sender.RedisConnection("127.0.0.1", 6379, 0.2,
                                    connect=lambda *a, **k: sock)
        conn.connect()
        return conn

    def test_integer_replies_fragmented(self):
        sock = FakeSocket([b":", b"0", b"\r", b"\n"])
        conn = self.make_conn(sock)
        self.assertEqual(conn.publish("c", True), 0)
        sock2 = FakeSocket([b":123", b"\r\n"])
        conn2 = self.make_conn(sock2)
        self.assertEqual(conn2.publish("c", False), 123)

    def test_persistent_reuse(self):
        sock = FakeSocket([b":1\r\n", b":2\r\n"])
        conn = self.make_conn(sock)
        conn.publish("c", True)
        conn.publish("c", True)
        self.assertEqual(sock.sent.count(b"PUBLISH"), 2)

    def test_error_replies_fail(self):
        for reply in (b"-DENIED nope\r\n", b"-NOAUTH auth\r\n",
                      b"-ERR bad\r\n"):
            sock = FakeSocket([reply])
            conn = self.make_conn(sock)
            with self.assertRaises(sender.RedisError):
                conn.publish("c", True)

    def test_malformed_replies(self):
        for reply in (b":-1\r\n", b":1_0\r\n", b": 1\r\n", b"+OK\r\n",
                      b":\r\n", b":1x\r\n", b":1\n\r\n", b":1\r\r\n",
                      b":+1\r\n"):
            sock = FakeSocket([reply])
            conn = self.make_conn(sock)
            with self.assertRaises(sender.RedisError, msg=repr(reply)):
                conn.publish("c", True)

    def test_eof_and_oversize(self):
        sock = FakeSocket([b":1"])
        conn = self.make_conn(sock)
        with self.assertRaises(sender.RedisError):
            conn.publish("c", True)
        big = FakeSocket([b":" + b"9" * 5000])
        conn2 = self.make_conn(big)
        with self.assertRaises(sender.RedisError):
            conn2.publish("c", True)

    def test_timeouts_and_reset(self):
        sock = FakeSocket(recv_error=socket.timeout())
        conn = self.make_conn(sock)
        with self.assertRaises(sender.RedisError):
            conn.publish("c", True)
        sock2 = FakeSocket(send_error=socket.timeout())
        conn2 = self.make_conn(sock2)
        with self.assertRaises(sender.RedisError):
            conn2.publish("c", True)
        sock3 = FakeSocket(recv_error=ConnectionResetError())
        conn3 = self.make_conn(sock3)
        with self.assertRaises(sender.RedisError):
            conn3.publish("c", True)


class AlternateTests(unittest.TestCase):
    def test_formula(self):
        self.assertFalse(sender.alternate_value(0.0, 0.0, 1.0))
        self.assertFalse(sender.alternate_value(0.4, 0.0, 1.0))
        self.assertTrue(sender.alternate_value(0.6, 0.0, 1.0))
        self.assertTrue(sender.alternate_value(1.4, 0.0, 1.0))
        self.assertFalse(sender.alternate_value(1.6, 0.0, 1.0))
        self.assertFalse(sender.alternate_value(10.49, 0.0, 1.0))
        self.assertTrue(sender.alternate_value(10.51, 0.0, 1.0))
        self.assertTrue(sender.alternate_value(0.5, 0.0, 1.0))
        self.assertFalse(sender.alternate_value(1.5, 0.0, 1.0))
        self.assertTrue(sender.alternate_value(2.5, 0.0, 1.0))


class ProducerTests(unittest.TestCase):
    def test_seeded_random_sequence(self):
        seen1, seen2 = [], []
        for seen in (seen1, seen2):
            clock = FakeClock()
            state = sender.FaceState()
            stop = FakeStop(clock)
            count = {"n": 0}

            def w(timeout=None, s=stop, c=count, v=seen):
                c["n"] += 1
                s.clock.advance(timeout or 0)
                if c["n"] >= 20:
                    s.set()
                return s._set

            stop.wait = w
            orig_update = state.update
            state.update = lambda d, t, v=seen: (
                v.append(d), orig_update(d, t))[0]
            sender.producer_loop(stop, state, "random", 42,
                                 clock.now, 1.0, clock=clock)
        self.assertEqual(seen1, seen2)
        self.assertGreater(len(seen1), 10)
        self.assertIn(True, seen1)
        self.assertIn(False, seen1)

    def test_forced_modes(self):
        for mode, want in (("true", True), ("false", False)):
            clock = FakeClock()
            state = sender.FaceState()
            stop = FakeStop(clock)
            count = {"n": 0}

            def w(timeout=None, s=stop, c=count):
                c["n"] += 1
                if c["n"] >= 3:
                    s.set()
                return s._set

            stop.wait = w
            sender.producer_loop(stop, state, mode, None,
                                 clock.now, 1.0, clock=clock)
            snap = state.snapshot()
            self.assertEqual(snap[0], want)

    def test_skips_missed_frames(self):
        clock = FakeClock()
        times = []

        class RecState(sender.FaceState):
            def update(self, d, t):
                times.append(t)
                super().update(d, t)

        state = RecState()
        stop = FakeStop(clock)
        count = {"n": 0}

        def w(timeout=None):
            count["n"] += 1
            clock.advance(1.0)
            if count["n"] >= 3:
                stop.set()
            return stop._set

        stop.wait = w
        sender.producer_loop(stop, state, "true", None,
                             clock.now, 1.0, fps=30, clock=clock)
        self.assertEqual(len(times), 3)
        self.assertEqual(times, sorted(set(times)))
        for i in range(1, len(times)):
            self.assertAlmostEqual(times[i] - times[i - 1], 1.0)


class SchedulingTests(unittest.TestCase):
    def test_next_deadline_skips_missed(self):
        self.assertAlmostEqual(
            sender.next_report_deadline(0.0, 1.0, 0.0), 1.0)
        self.assertAlmostEqual(
            sender.next_report_deadline(0.0, 1.0, 2.5), 3.0)
        self.assertAlmostEqual(
            sender.next_report_deadline(0.0, 0.3, 1.0), 1.2)
        self.assertAlmostEqual(
            sender.next_report_deadline(0.0, 1.0, 3.0), 4.0)

    def test_report_times_and_reuse(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        sock = FakeSocket([b":0\r\n", b":3\r\n", b":0\r\n"],
                          clock=clock)
        a = redis_args()
        logs, abort, _ = run_publisher(
            state, a, clock, lambda *a_, **k: sock, max_waits=4)
        pubs = [l for l in logs if "published" in l]
        self.assertEqual(len(pubs), 3)
        self.assertIn("subscribers=3", pubs[1])
        self.assertEqual(sock.sent.count(b"PUBLISH"), 3)
        self.assertAlmostEqual(sock.sent_times[0], 0.5)
        self.assertAlmostEqual(sock.sent_times[1], 1.0)
        self.assertAlmostEqual(sock.sent_times[2], 1.5)
        self.assertFalse(abort.is_set())

    def test_no_initial_or_stale_report(self):
        clock = FakeClock(0.0)
        state = sender.FaceState()
        sock = FakeSocket([b":0\r\n"])
        a = redis_args()
        logs, abort, _ = run_publisher(
            state, a, clock, lambda *a_, **k: sock, max_waits=5)
        self.assertEqual(sock.sent, b"")

        clock2 = FakeClock(0.0)
        state2 = sender.FaceState()
        state2.update(True, -100.0)
        sock2 = FakeSocket([b":0\r\n"])
        logs2, abort2, _ = run_publisher(
            state2, a, clock2, lambda *a_, **k: sock2, max_waits=3)
        self.assertEqual(sock2.sent, b"")

    def test_exact_boundary_is_stale(self):
        clock = FakeClock(0.0)
        state = sender.FaceState()
        a = redis_args(face_stale_seconds=2.0)
        state.update(True, -1.5)
        sock = FakeSocket([b":0\r\n"])
        logs, abort, _ = run_publisher(
            state, a, clock, lambda *a_, **k: sock, max_waits=2)
        self.assertEqual(sock.sent, b"")

    def test_fresh_resumption_after_stale(self):
        clock = FakeClock(0.0)
        state = sender.FaceState()
        state.update(True, -100.0)
        sock = FakeSocket([b":1\r\n"] * 10)
        a = redis_args()
        stop = FakeStop(clock)
        abort = threading.Event()
        logs = []
        count = {"n": 0}

        def on_wait(timeout):
            count["n"] += 1
            if count["n"] == 3:
                state.update(True, clock.now)
            if count["n"] >= 5:
                stop.set()

        stop.on_wait = on_wait
        sender.publisher_loop(stop, abort, state, a, 0.0,
                              logs.append, clock=clock,
                              connect=lambda *a_, **k: sock)
        self.assertGreaterEqual(sock.sent.count(b"PUBLISH"), 1)

    def test_resample_after_connect(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, False)
        sock = FakeSocket([b":1\r\n"])
        a = redis_args()
        stop = FakeStop(clock)
        abort = threading.Event()
        logs = []

        def connect(addr, timeout=None):
            state.update(True, clock.now)
            return sock

        count = {"n": 0}

        def on_wait(timeout):
            count["n"] += 1
            if count["n"] >= 2:
                stop.set()

        stop.on_wait = on_wait
        sender.publisher_loop(stop, abort, state, a, 0.0,
                              logs.append, clock=clock,
                              connect=connect)
        self.assertIn(b'{"face_detected":true}', sock.sent)

    def test_stale_after_slow_connect_no_publish(self):
        clock = FakeClock(0.0)
        state = sender.FaceState()
        state.update(True, 0.0)
        sock = FakeSocket([b":1\r\n"])
        a = redis_args(face_stale_seconds=1.0)
        stop = FakeStop(clock)
        abort = threading.Event()
        logs = []

        def connect(addr, timeout=None):
            clock.advance(5.0)
            return sock

        count = {"n": 0}

        def on_wait(timeout):
            count["n"] += 1
            if count["n"] >= 2:
                stop.set()

        stop.on_wait = on_wait
        sender.publisher_loop(stop, abort, state, a, 0.0,
                              logs.append, clock=clock,
                              connect=connect)
        self.assertEqual(sock.sent, b"")

    def test_stop_after_connect_no_publish(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        sock = FakeSocket([b":1\r\n"])
        a = redis_args()
        stop = FakeStop(clock)
        abort = threading.Event()
        logs = []

        def connect(addr, timeout=None):
            stop.set()
            return sock

        sender.publisher_loop(stop, abort, state, a, 0.0,
                              logs.append, clock=clock,
                              connect=connect)
        self.assertEqual(sock.sent, b"")

    def test_stop_on_persistent_connection_no_publish(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        sock = FakeSocket([b":1\r\n"] * 4)
        a = redis_args()
        stop = FakeStop(clock)
        abort = threading.Event()
        logs = []

        def on_wait(timeout):
            if sock.sent.count(b"PUBLISH") >= 1:
                stop.set()

        stop.on_wait = on_wait
        sender.publisher_loop(stop, abort, state, a, 0.0,
                              logs.append, clock=clock,
                              connect=lambda *a_, **k: sock)
        self.assertEqual(sock.sent.count(b"PUBLISH"), 1)

    def test_backoff_gaps_and_cap(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        attempts = []
        a = redis_args()
        logs, abort, _ = run_publisher(
            state, a, clock,
            lambda *a_, **k: (_ for _ in ()).throw(OSError("x"))
            if not attempts.append(clock.now) else None,
            max_waits=40)
        self.assertFalse(abort.is_set())
        self.assertGreaterEqual(len(attempts), 6)
        gaps = [attempts[i + 1] - attempts[i]
                for i in range(len(attempts) - 1)]
        self.assertAlmostEqual(gaps[0], 1.0)
        self.assertTrue(any(4.9 <= g <= 5.6 for g in gaps))
        self.assertTrue(all(g <= 5.6 for g in gaps))

    def test_reconnect_after_failure_success(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        sock = FakeSocket([b":2\r\n"] * 4)
        calls = {"n": 0}

        def connect(addr, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("refused")
            return sock

        a = redis_args()
        logs, abort, _ = run_publisher(state, a, clock, connect,
                                       max_waits=8)
        pubs = [l for l in logs if "published" in l]
        self.assertGreaterEqual(len(pubs), 1)
        self.assertIn("subscribers=2", pubs[0])
        self.assertFalse(abort.is_set())
        self.assertTrue(sock.closed)

    def test_error_log_throttle(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        error_times = []

        def log(msg):
            if "published" not in msg:
                error_times.append(clock.now)

        stop = FakeStop(clock)
        abort = threading.Event()
        count = {"n": 0}

        def on_wait(timeout):
            count["n"] += 1
            if count["n"] >= 30:
                stop.set()

        stop.on_wait = on_wait
        a = redis_args(redis_interval_seconds=0.5)
        sender.publisher_loop(
            stop, abort, state, a, 0.0, log, clock=clock,
            connect=lambda *a_, **k: (_ for _ in ()).throw(
                OSError("down")))
        self.assertAlmostEqual(error_times[0], 0.5)
        for i in range(1, len(error_times)):
            self.assertGreaterEqual(
                error_times[i] - error_times[i - 1], 5.0)
        self.assertGreaterEqual(clock.now, 10.0)

    def test_uncertain_write_no_immediate_replay(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        sock = FakeSocket(recv_error=socket.timeout())
        calls = {"n": 0}

        def connect(addr, timeout=None):
            calls["n"] += 1
            if calls["n"] > 1:
                raise OSError("still down")
            return sock

        a = redis_args()
        logs, abort, _ = run_publisher(state, a, clock, connect,
                                       max_waits=4)
        self.assertEqual(sock.sent.count(b"PUBLISH"), 1)
        self.assertTrue(sock.closed)

    def test_no_burst_after_gap(self):
        clock = FakeClock(0.0)
        state = fresh_state(clock, True)
        sock = FakeSocket([b":1\r\n"] * 20)
        a = redis_args()
        stop = FakeStop(clock)
        abort = threading.Event()
        logs = []
        count = {"n": 0}

        def on_wait(timeout):
            clock.advance(2.0)
            count["n"] += 1
            if count["n"] >= 5:
                stop.set()

        stop.on_wait = on_wait
        sender.publisher_loop(stop, abort, state, a, 0.0,
                              logs.append, clock=clock,
                              connect=lambda *a_, **k: sock)
        pubs = sock.sent.count(b"PUBLISH")
        self.assertGreaterEqual(pubs, 1)
        self.assertLessEqual(pubs, 5)


class MainFlowTests(unittest.TestCase):
    def test_main_bad_prefix(self):
        code = sender.main([
            "--pi-host", "127.0.0.1",
            "--deps-prefix", "/nonexistent-dir-xyz",
        ])
        self.assertEqual(code, 2)

    def test_main_missing_tools(self):
        with mock.patch.object(sender, "check_tools",
                               side_effect=sender.DependencyError(
                                   "no gst")):
            code = sender.main(["--pi-host", "127.0.0.1"])
        self.assertEqual(code, 2)

    def test_main_video_unexpected_exit_nonzero(self):
        for code in (0, 1):
            proc = FakeProc(exit_after_polls=2, code=code)
            with mock.patch.object(sender, "check_tools"), \
                 mock.patch.object(sender, "select_encoder",
                                   return_value="x264"), \
                 mock.patch.object(sender, "launch_video",
                                   return_value=proc), \
                 mock.patch.object(sender.socket,
                                   "create_connection",
                                   side_effect=OSError("no redis")):
                rc = sender.main([
                    "--pi-host", "127.0.0.1", "--encoder", "x264",
                    "--redis-interval-seconds", "0.1",
                ])
            self.assertEqual(rc, 1)

    def test_main_worker_failure_nonzero(self):
        def dead_publisher(*a, **k):
            a[1].set()

        proc2 = FakeProc(exit_after_polls=None)
        with mock.patch.object(sender, "check_tools"), \
             mock.patch.object(sender, "select_encoder",
                               return_value="x264"), \
             mock.patch.object(sender, "launch_video",
                               return_value=proc2), \
             mock.patch.object(sender, "publisher_loop",
                               dead_publisher), \
             mock.patch.object(sender.socket,
                               "create_connection",
                               side_effect=OSError("x")):
            rc = sender.main([
                "--pi-host", "127.0.0.1", "--encoder", "x264",
                "--duration-seconds", "5",
                "--redis-interval-seconds", "0.1",
            ])
        self.assertEqual(rc, 1)

    def test_main_launch_oserror_friendly(self):
        with mock.patch.object(sender, "check_tools"), \
             mock.patch.object(sender, "select_encoder",
                               return_value="x264"), \
             mock.patch.object(sender, "launch_video",
                               side_effect=OSError("ENOENT")):
            rc = sender.main(["--pi-host", "127.0.0.1"])
        self.assertEqual(rc, 2)

    def test_main_duration_graceful(self):
        proc = FakeProc(exit_after_polls=None)
        with mock.patch.object(sender, "check_tools"), \
             mock.patch.object(sender, "select_encoder",
                               return_value="x264"), \
             mock.patch.object(sender, "launch_video",
                               return_value=proc), \
             mock.patch.object(sender.socket,
                               "create_connection",
                               side_effect=OSError("no redis")):
            code = sender.main([
                "--pi-host", "127.0.0.1", "--encoder", "x264",
                "--duration-seconds", "0.3",
                "--redis-interval-seconds", "0.1",
            ])
        self.assertEqual(code, 0)
        self.assertTrue(proc.terminated or proc.killed)

    def test_main_stop_during_preflight_graceful(self):
        real_select = sender.select_encoder

        def select_with_stop(env, args, stop, log):
            return real_select(
                env, args, stop, log,
                exists=lambda e, el, st: (st.set(), True)[1],
                smoker=lambda *a_, **k: None)

        with mock.patch.object(sender, "check_tools"), \
             mock.patch.object(sender, "select_encoder",
                               select_with_stop), \
             mock.patch.object(sender, "launch_video",
                               side_effect=AssertionError(
                                   "launched")):
            rc = sender.main([
                "--pi-host", "127.0.0.1", "--encoder", "x264"])
        self.assertEqual(rc, 0)

    def test_main_signal_handlers_restored(self):
        import signal as sig
        before_int = sig.getsignal(sig.SIGINT)
        before_term = sig.getsignal(sig.SIGTERM)
        proc = FakeProc(exit_after_polls=None)
        with mock.patch.object(sender, "check_tools"), \
             mock.patch.object(sender, "select_encoder",
                               return_value="x264"), \
             mock.patch.object(sender, "launch_video",
                               return_value=proc), \
             mock.patch.object(sender.socket,
                               "create_connection",
                               side_effect=OSError("x")):
            sender.main([
                "--pi-host", "127.0.0.1", "--encoder", "x264",
                "--duration-seconds", "0.2",
                "--redis-interval-seconds", "0.1",
            ])
        self.assertIs(sig.getsignal(sig.SIGINT), before_int)
        self.assertIs(sig.getsignal(sig.SIGTERM), before_term)


class SourceSyntaxTests(unittest.TestCase):
    def test_py39_syntax_and_stdlib_only(self):
        source = (REPO_ROOT / "scripts" / "mock_yolov26_sender.py"
                  ).read_text()
        tree = ast.parse(source, feature_version=(3, 9))
        allowed = {
            "argparse", "ipaddress", "math", "os", "platform",
            "random", "re", "shutil", "signal", "socket",
            "subprocess", "sys", "tempfile", "threading", "time",
            "pathlib",
        }
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        self.assertLessEqual(names, allowed)


if __name__ == "__main__":
    unittest.main()
