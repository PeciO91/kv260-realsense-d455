import argparse
import ipaddress
import math
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

RTP_PAYLOAD_TYPE = 96
RTP_MTU = 1200
INSPECT_TIMEOUT_SECONDS = 5.0
SMOKE_TIMEOUT_SECONDS = 5.0
PROCESS_STOP_GRACE_SECONDS = 2.0
SUPERVISION_SECONDS = 0.05
WAIT_CLAMP_SECONDS = 60.0
RECONNECT_INITIAL_SECONDS = 1.0
RECONNECT_MAX_SECONDS = 5.0
ERROR_LOG_THROTTLE_SECONDS = 5.0
REDIS_REPLY_LIMIT = 4096
REDIS_ERROR_TEXT_LIMIT = 200
OUTPUT_TAIL_BYTES = 2000
DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30

HOST_CHARS = re.compile(r"[A-Za-z0-9.:%_-]+")
DNS_LABEL = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?")
SCOPE_ID = re.compile(r"[A-Za-z0-9._-]+")
REPLY_INTEGER = re.compile(rb"[0-9]+")
IPV4_SHAPE = re.compile(r"[0-9.]+")

COMMON_FACTORIES = (
    "videotestsrc",
    "capsfilter",
    "queue",
    "videoconvert",
    "h264parse",
    "rtph264pay",
    "fakesink",
    "udpsink",
)
ENCODER_ELEMENTS = {"omx": "omxh264enc", "x264": "x264enc"}


class DependencyError(Exception):
    pass


class RedisError(Exception):
    pass


def validate_host(text):
    if not text or not HOST_CHARS.fullmatch(text):
        raise argparse.ArgumentTypeError("invalid host %r" % (text,))
    if ":" in text:
        if text.count("%") > 1:
            raise argparse.ArgumentTypeError("invalid IPv6 host %r" % text)
        address, _, scope = text.partition("%")
        try:
            ipaddress.IPv6Address(address)
        except ValueError:
            raise argparse.ArgumentTypeError("invalid IPv6 host %r" % text)
        if "%" in text and not SCOPE_ID.fullmatch(scope):
            raise argparse.ArgumentTypeError(
                "invalid IPv6 scope in %r" % text
            )
        return text
    if IPV4_SHAPE.fullmatch(text):
        try:
            ipaddress.IPv4Address(text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "invalid IPv4 host %r" % text
            )
        return text
    name = text[:-1] if text.endswith(".") else text
    labels = name.split(".")
    if (
        len(name) > 253
        or not labels
        or any(not DNS_LABEL.fullmatch(label) for label in labels)
    ):
        raise argparse.ArgumentTypeError("invalid DNS host %r" % text)
    return text


def port_number(text):
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid port %r" % text)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError("port out of range %r" % text)
    return value


def positive_int(text):
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid integer %r" % text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive %r" % text)
    return value


def even_dimension(text):
    value = positive_int(text)
    if value % 2 != 0 or value > 8192:
        raise argparse.ArgumentTypeError(
            "dimension must be an even integer in [2, 8192], got %r" % text
        )
    return value


def fps_value(text):
    value = positive_int(text)
    if value > 240:
        raise argparse.ArgumentTypeError(
            "fps must be an integer in [1, 240], got %r" % text
        )
    return value


def bounded_timeout_ms(text):
    value = positive_int(text)
    if value > 10000:
        raise argparse.ArgumentTypeError("timeout too large %r" % text)
    return value


def finite_float(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid number %r" % text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be finite %r" % text)
    return value


def report_interval(text):
    value = finite_float(text)
    if value < 0.1:
        raise argparse.ArgumentTypeError("interval too small %r" % text)
    return value


def positive_float(text):
    value = finite_float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive %r" % text)
    return value


def nonempty_text(text):
    if not text:
        raise argparse.ArgumentTypeError("value must not be empty")
    return text


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "MOCK YOLOv26 sender: synthetic H.264/RTP video plus Redis "
            "face-status messages. Not real inference."
        )
    )
    parser.add_argument("--pi-host", required=True, type=validate_host)
    parser.add_argument("--video-port", type=port_number, default=5000)
    parser.add_argument(
        "--encoder", choices=("auto", "omx", "x264"), default="auto"
    )
    parser.add_argument("--bitrate-kbps", type=positive_int, default=4000)
    parser.add_argument("--redis-host", type=validate_host, default=None)
    parser.add_argument("--redis-port", type=port_number, default=6379)
    parser.add_argument(
        "--redis-channel", type=nonempty_text, default="face_detection"
    )
    parser.add_argument(
        "--redis-interval-seconds", type=report_interval, default=1.0
    )
    parser.add_argument(
        "--face-stale-seconds", type=positive_float, default=2.0
    )
    parser.add_argument(
        "--redis-timeout-ms", type=bounded_timeout_ms, default=200
    )
    parser.add_argument(
        "--pattern", choices=("snow", "ball"), default="snow"
    )
    parser.add_argument(
        "--face-mode",
        choices=("random", "true", "false", "alternate"),
        default="random",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration-seconds", type=positive_float, default=None)
    parser.add_argument(
        "--width",
        type=even_dimension,
        default=DEFAULT_WIDTH,
        help="even integer in [2, 8192] (default 848)",
    )
    parser.add_argument(
        "--height",
        type=even_dimension,
        default=DEFAULT_HEIGHT,
        help="even integer in [2, 8192] (default 480)",
    )
    parser.add_argument(
        "--fps",
        type=fps_value,
        default=DEFAULT_FPS,
        help="integer in [1, 240] (default 30)",
    )
    parser.add_argument("--deps-prefix", type=str, default=None)
    return parser


def resolve_deps_prefix(explicit):
    if explicit is not None:
        prefix = Path(explicit)
        if not (prefix / "lib" / "gstreamer-1.0").is_dir():
            raise DependencyError(
                "--deps-prefix %s lacks lib/gstreamer-1.0" % prefix
            )
        return prefix
    auto = Path.home() / "kv260-network-deps"
    if (auto / "lib" / "gstreamer-1.0").is_dir():
        return auto
    return None


def _prepend_unique(value, existing):
    entries = [
        e for e in (existing or "").split(os.pathsep) if e and e != value
    ]
    return os.pathsep.join([value] + entries)


def build_child_env(base, prefix):
    env = dict(base)
    if prefix is None:
        return env
    lib = str(prefix / "lib")
    plugins = str(prefix / "lib" / "gstreamer-1.0")
    env["LD_LIBRARY_PATH"] = _prepend_unique(
        lib, base.get("LD_LIBRARY_PATH")
    )
    effective_plugins = base.get("GST_PLUGIN_PATH_1_0")
    if effective_plugins is None:
        effective_plugins = base.get("GST_PLUGIN_PATH")
    merged = _prepend_unique(plugins, effective_plugins)
    env["GST_PLUGIN_PATH"] = merged
    env["GST_PLUGIN_PATH_1_0"] = merged
    registry = base.get("GST_REGISTRY_1_0") or base.get("GST_REGISTRY")
    if registry is None:
        registry = str(
            prefix / ("registry-%s.bin" % platform.machine())
        )
    env["GST_REGISTRY"] = registry
    env["GST_REGISTRY_1_0"] = registry
    return env


def run_managed(argv, env, timeout, stop, clock=time.monotonic):
    if stop.is_set():
        raise DependencyError("cancelled: %s" % argv[0])
    with tempfile.TemporaryFile() as capture:
        proc = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=capture,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = clock() + timeout
        timed_out = False
        cancelled = False
        try:
            while proc.poll() is None:
                if stop.is_set():
                    cancelled = True
                    break
                if clock() >= deadline:
                    timed_out = True
                    break
                stop.wait(
                    min(SUPERVISION_SECONDS,
                        max(0.0, deadline - clock()))
                )
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(PROCESS_STOP_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        capture.seek(max(0, _capture_size(capture) - OUTPUT_TAIL_BYTES))
        tail = capture.read(OUTPUT_TAIL_BYTES).decode("utf-8", "replace")
    if cancelled:
        raise DependencyError("cancelled: %s" % argv[0])
    if timed_out:
        raise DependencyError(
            "timeout after %.1fs: %s\n%s" % (timeout, argv[0], tail)
        )
    return proc.returncode, tail


def _capture_size(capture):
    pos = capture.tell()
    capture.seek(0, 2)
    size = capture.tell()
    capture.seek(pos)
    return size


def element_exists(env, element, stop, runner=run_managed):
    try:
        code, _ = runner(
            ["gst-inspect-1.0", "--exists", element],
            env,
            INSPECT_TIMEOUT_SECONDS,
            stop,
        )
    except DependencyError:
        return False
    return code == 0


def encoder_spec(encoder, bitrate_kbps):
    if encoder == "x264":
        return [
            "x264enc",
            "tune=zerolatency",
            "speed-preset=ultrafast",
            "bitrate=%d" % bitrate_kbps,
            "key-int-max=30",
            "bframes=0",
            "rc-lookahead=0",
        ]
    return [
        "omxh264enc",
        "target-bitrate=%d" % bitrate_kbps,
        "control-rate=low-latency",
        "b-frames=0",
        "look-ahead=0",
        "gop-mode=basic",
        "gop-length=30",
        "periodicity-idr=30",
    ]


def build_pipeline_argv(args, encoder, main_sink):
    source = ["gst-launch-1.0", "-q", "-e", "videotestsrc"]
    if main_sink:
        source += ["is-live=true"]
    else:
        source += ["num-buffers=1"]
    source += ["pattern=%s" % args.pattern]
    pipeline = source + [
        "!",
        "video/x-raw,format=BGR,width=%d,height=%d,framerate=%d/1"
        % (args.width, args.height, args.fps),
        "!",
        "queue",
        "max-size-buffers=1",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=NV12",
        "!",
    ]
    pipeline += encoder_spec(encoder, args.bitrate_kbps)
    pipeline += [
        "!",
        "video/x-h264,profile=baseline",
        "!",
        "h264parse",
        "!",
        "video/x-h264,alignment=au,stream-format=byte-stream",
        "!",
        "rtph264pay",
        "pt=%d" % RTP_PAYLOAD_TYPE,
        "config-interval=1",
        "mtu=%d" % RTP_MTU,
        "!",
    ]
    if main_sink:
        pipeline += [
            "udpsink",
            "host=%s" % args.pi_host,
            "port=%d" % args.video_port,
            "sync=false",
            "async=false",
        ]
    else:
        pipeline += ["fakesink", "sync=false"]
    return pipeline


def smoke_encoder(env, encoder, args, stop, runner=run_managed):
    argv = build_pipeline_argv(args, encoder, main_sink=False)
    code, tail = runner(argv, env, SMOKE_TIMEOUT_SECONDS, stop)
    if code != 0:
        raise DependencyError(
            "%s smoke encode failed (exit %s):\n%s"
            % (ENCODER_ELEMENTS[encoder], code, tail)
        )


def check_tools():
    for tool in ("gst-launch-1.0", "gst-inspect-1.0"):
        if shutil.which(tool) is None:
            raise DependencyError("missing tool: %s" % tool)


def select_encoder(env, args, stop, log, exists=element_exists,
                   smoker=smoke_encoder):
    for factory in COMMON_FACTORIES:
        if stop.is_set():
            raise DependencyError("cancelled")
        if not exists(env, factory, stop):
            raise DependencyError(
                "missing GStreamer factory: %s" % factory
            )
    if args.encoder != "auto":
        element = ENCODER_ELEMENTS[args.encoder]
        if not exists(env, element, stop):
            raise DependencyError(
                "missing encoder element: %s" % element
            )
        if stop.is_set():
            raise DependencyError("cancelled")
        smoker(env, args.encoder, args, stop)
        return args.encoder
    for candidate in ("omx", "x264"):
        if stop.is_set():
            raise DependencyError("cancelled")
        element = ENCODER_ELEMENTS[candidate]
        if not exists(env, element, stop):
            log("encoder %s not present" % element)
            continue
        try:
            smoker(env, candidate, args, stop)
        except DependencyError as exc:
            log("encoder %s unusable: %s" % (element, exc))
            continue
        return candidate
    raise DependencyError("no usable H.264 encoder (omx/x264)")


def face_payload(detected):
    if type(detected) is not bool:
        raise TypeError("face_detected must be a bool")
    return b'{"face_detected":%s}' % (b"true" if detected else b"false")


def encode_publish_request(channel, payload):
    parts = [b"PUBLISH", channel.encode("utf-8"), payload]
    out = b"*3\r\n"
    for part in parts:
        out += b"$%d\r\n%s\r\n" % (len(part), part)
    return out


class RedisConnection:
    def __init__(self, host, port, timeout, connect=socket.create_connection):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._connect = connect
        self.sock = None
        self.buffer = b""

    def connect(self):
        self.close()
        self.sock = self._connect(
            (self.host, self.port), timeout=self.timeout
        )
        self.sock.settimeout(self.timeout)
        self.buffer = b""

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.buffer = b""

    def _read_line(self):
        while b"\r\n" not in self.buffer:
            if len(self.buffer) >= REDIS_REPLY_LIMIT:
                raise RedisError("reply exceeds %d bytes" % REDIS_REPLY_LIMIT)
            try:
                chunk = self.sock.recv(REDIS_REPLY_LIMIT)
            except socket.timeout:
                raise RedisError("reply read timed out")
            except OSError as exc:
                raise RedisError("reply read failed: %s" % exc)
            if not chunk:
                raise RedisError("connection closed by server")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\r\n", 1)
        if len(line) + 2 > REDIS_REPLY_LIMIT:
            raise RedisError("reply exceeds %d bytes" % REDIS_REPLY_LIMIT)
        return line

    def publish(self, channel, detected):
        if self.sock is None:
            raise RedisError("not connected")
        request = encode_publish_request(channel, face_payload(detected))
        try:
            self.sock.sendall(request)
        except socket.timeout:
            raise RedisError("publish write timed out")
        except OSError as exc:
            raise RedisError("publish write failed: %s" % exc)
        line = self._read_line()
        tag, body = line[:1], line[1:]
        if tag == b":":
            if not REPLY_INTEGER.fullmatch(body):
                raise RedisError("malformed integer reply")
            return int(body)
        if tag == b"-":
            text = body.decode("utf-8", "replace")[:REDIS_ERROR_TEXT_LIMIT]
            raise RedisError("server error: %s" % text)
        raise RedisError("unexpected reply type %r" % tag)


class FaceState:
    def __init__(self):
        self._lock = threading.Lock()
        self._available = False
        self._detected = False
        self._inferred_at = 0.0

    def update(self, detected, inferred_at):
        with self._lock:
            self._available = True
            self._detected = detected
            self._inferred_at = inferred_at

    def snapshot(self):
        with self._lock:
            if not self._available:
                return None
            return self._detected, self._inferred_at


def alternate_value(now, origin, interval):
    index = int((now - origin + interval / 2.0) / interval)
    return bool(index % 2)


def next_report_deadline(origin, interval, now):
    index = int((now - origin) / interval) + 1
    deadline = origin + index * interval
    while deadline <= now:
        index += 1
        deadline = origin + index * interval
    return deadline


def wait_until(stop, deadline, clock=time.monotonic):
    while not stop.is_set():
        remaining = deadline - clock()
        if remaining <= 0:
            return True
        stop.wait(min(remaining, WAIT_CLAMP_SECONDS))
    return False


def producer_loop(stop, state, mode, seed, origin, report_interval,
                  fps=DEFAULT_FPS, clock=time.monotonic):
    rng = random.Random(seed)
    frame_interval = 1.0 / fps
    while not stop.is_set():
        now = clock()
        if mode == "random":
            detected = bool(rng.getrandbits(1))
        elif mode == "true":
            detected = True
        elif mode == "false":
            detected = False
        else:
            detected = alternate_value(now, origin, report_interval)
        state.update(detected, now)
        deadline = next_report_deadline(origin, frame_interval, clock())
        if not wait_until(stop, deadline, clock):
            return


def publisher_loop(stop, abort, state, args, origin, log,
                   clock=time.monotonic, connect=socket.create_connection):
    interval = args.redis_interval_seconds
    stale = args.face_stale_seconds
    timeout = args.redis_timeout_ms / 1000.0
    conn = RedisConnection(
        args.redis_host, args.redis_port, timeout, connect=connect
    )
    deadline = origin + interval
    retry_at = None
    backoff = RECONNECT_INITIAL_SECONDS
    last_error_log = None

    def log_error(message):
        nonlocal last_error_log
        now = clock()
        if (
            last_error_log is None
            or now - last_error_log >= ERROR_LOG_THROTTLE_SECONDS
        ):
            log("redis: %s" % message)
            last_error_log = now

    def fresh_snapshot():
        snap = state.snapshot()
        now = clock()
        if snap is None or now - snap[1] >= stale:
            return None
        return snap

    try:
        while not stop.is_set():
            if not wait_until(stop, deadline, clock):
                break
            now = clock()
            if retry_at is not None and now < retry_at:
                deadline = next_report_deadline(origin, interval, now)
                continue
            snap = fresh_snapshot()
            if snap is None:
                deadline = next_report_deadline(origin, interval, now)
                continue
            try:
                if conn.sock is None:
                    conn.connect()
                snap = fresh_snapshot()
                if stop.is_set():
                    break
                if snap is None:
                    deadline = next_report_deadline(
                        origin, interval, clock()
                    )
                    continue
                detected, _ = snap
                subscribers = conn.publish(args.redis_channel, detected)
            except (RedisError, OSError) as exc:
                conn.close()
                log_error(str(exc))
                retry_at = clock() + backoff
                backoff = min(backoff * 2.0, RECONNECT_MAX_SECONDS)
                deadline = next_report_deadline(origin, interval, clock())
                continue
            retry_at = None
            backoff = RECONNECT_INITIAL_SECONDS
            log("redis: published %r subscribers=%d"
                % (face_payload(detected).decode("utf-8"), subscribers))
            deadline = next_report_deadline(origin, interval, clock())
    except Exception as exc:
        log("redis worker failed: %r" % exc)
        abort.set()
    finally:
        conn.close()


def print_banner(args, prefix, encoder):
    print("=" * 60)
    print("MOCK YOLOv26 SENDER - synthetic data, not real inference")
    print("=" * 60)
    print("deps prefix       : %s" % (prefix if prefix else "(system)"))
    print("video             : rtp/udp %s:%d pt=%d %dx%d@%d %s %d kbps"
          % (args.pi_host, args.video_port, RTP_PAYLOAD_TYPE,
             args.width, args.height, args.fps, encoder,
             args.bitrate_kbps))
    print("redis             : %s:%d channel=%r interval=%.3fs stale=%.3fs"
          % (args.redis_host, args.redis_port, args.redis_channel,
             args.redis_interval_seconds, args.face_stale_seconds))
    print("face mode         : %s seed=%s" % (args.face_mode, args.seed))
    print("=" * 60, flush=True)


def launch_video(argv, env):
    return subprocess.Popen(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=None,
        start_new_session=True,
    )


def stop_process(proc):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(PROCESS_STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _guarded_producer(stop, abort, state, args, origin, log,
                      clock=time.monotonic):
    try:
        producer_loop(
            stop, state, args.face_mode, args.seed, origin,
            args.redis_interval_seconds, fps=args.fps, clock=clock,
        )
    except Exception as exc:
        log("producer failed: %r" % exc)
        abort.set()


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.redis_host is None:
        args.redis_host = args.pi_host

    def log(message):
        print("mock-sender: %s" % message, flush=True)

    try:
        prefix = resolve_deps_prefix(args.deps_prefix)
    except DependencyError as exc:
        log("error: %s" % exc)
        return 2
    child_env = build_child_env(os.environ, prefix)
    log("dependency prefix: %s" % (prefix if prefix else "system"))

    stop = threading.Event()
    abort = threading.Event()
    previous_handlers = {}

    def request_stop(signum, frame):
        stop.set()

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)

    proc = None
    producer = None
    publisher = None
    exit_code = 0
    try:
        check_tools()
        encoder = select_encoder(child_env, args, stop, log)
        log("using encoder: %s" % ENCODER_ELEMENTS[encoder])
        if stop.is_set():
            return 0
        print_banner(args, prefix, encoder)
        proc = launch_video(
            build_pipeline_argv(args, encoder, main_sink=True), child_env
        )
        origin = time.monotonic()
        duration_deadline = (
            origin + args.duration_seconds
            if args.duration_seconds is not None
            else None
        )
        state = FaceState()
        producer = threading.Thread(
            target=_guarded_producer,
            args=(stop, abort, state, args, origin, log),
            daemon=True,
        )
        publisher = threading.Thread(
            target=publisher_loop,
            args=(stop, abort, state, args, origin, log),
            kwargs={"connect": socket.create_connection},
            daemon=True,
        )
        producer.start()
        publisher.start()
        while not stop.is_set() and not abort.is_set():
            if proc.poll() is not None:
                if not stop.is_set():
                    log("video process exited rc=%s" % proc.returncode)
                    exit_code = 1
                break
            if (
                duration_deadline is not None
                and time.monotonic() >= duration_deadline
            ):
                break
            stop.wait(SUPERVISION_SECONDS)
        if abort.is_set() and exit_code == 0:
            exit_code = 1
    except DependencyError as exc:
        if stop.is_set():
            exit_code = 0
        else:
            log("error: %s" % exc)
            exit_code = 2
    except OSError as exc:
        log("error: %s" % exc)
        exit_code = 2
    finally:
        stop.set()
        if proc is not None:
            stop_process(proc)
        for thread in (producer, publisher):
            if thread is not None:
                thread.join(
                    timeout=args.redis_timeout_ms / 1000.0 + 2.0
                )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
