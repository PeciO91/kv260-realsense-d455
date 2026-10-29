# Scripts

Everything here is run from the **repository root**. Build scripts write
binaries to `build/` (git-ignored); launchers default to those binaries.

## Catalogue

| Script | Kind | Runs on | Purpose |
|--------|------|---------|---------|
| `build_live_yolov26.sh` | build | KV260 | Compile the YOLOv26 browser MJPEG app (`inference/yolov26/mjpeg` + `core`) → `build/live_yolov26_realsense` |
| `build_live_yolov26_rpi.sh` | build | KV260 | Compile the Pi sender (`rpi` + `core`) → `build/live_yolov26_rpi`. Options `--deps-prefix DIR`, `--rs-prefix DIR`. Checks for GStreamer/hiredis pkg-config modules and librealsense before compiling |
| `build_live_yolov26_hdmi.sh` | build | KV260 | Compile the HDMI app with both X11 and KMS backends (`hdmi` + `core`) → `build/live_yolov26_hdmi`. Option `--rs-prefix DIR`. Checks for `opencv4 x11 gtk+-3.0 libdrm` |
| `build_live_densebox.sh` | build | KV260 | Compile the live DenseBox MJPEG app → `build/live_densebox_realsense` |
| `build_densebox_realsense.sh` | build | KV260 | Compile the single-shot DenseBox diagnostic → `build/densebox_realsense` (`CXX` env override) |
| `run_live_yolov26_hdmi.py` | launcher | KV260 | Wraps `build/live_yolov26_hdmi`. Adds `--check-display` (XRandR/DPMS query) and `--wake-display` (DPMS on, screensaver reset) that never touch the camera or model. Everything after `--` goes to the binary |
| `run_live_yolov26_rpi.py` | launcher | KV260 | Execs `build/live_yolov26_rpi` with the isolated GStreamer/hiredis environment from `--deps-prefix` (plugin paths, private registry), so nothing is installed globally |
| `mock_yolov26_sender.py` | test tool | KV260 or any Linux | Sends synthetic H.264/RTP video plus a simulated Redis `face_detected` flag using the real wire contract. Needs no camera, model or DPU. Lets the Pi receiver be developed independently. See [the guide below](#mock-sender-usage-guide) |
| `summarize_telemetry.py` | analysis | any | Turns saved `[telemetry]` logs into the markdown tables used in the main README (see [`docs/performance/`](../docs/performance/README.md)) |

### Typical workflows

```sh
# Browser view
sh scripts/build_live_yolov26.sh && ./build/live_yolov26_realsense --model <xmodel>

# Local monitor, lowest latency
sh scripts/build_live_yolov26_hdmi.sh
systemctl stop xserver-nodm
./build/live_yolov26_hdmi --renderer kms --model <xmodel> --telemetry true

# Raspberry Pi
sh scripts/build_live_yolov26_rpi.sh
python3 scripts/run_live_yolov26_rpi.py --pi-host <PI_IP> --encoder x264 --model <xmodel>

# Develop the Pi receiver without hardware
python3 scripts/mock_yolov26_sender.py --pi-host <PI_IP> --encoder x264
```

The launchers and the mock share dependency-prefix helpers:
`run_live_yolov26_rpi.py` imports them from `mock_yolov26_sender.py`.

---

## Mock sender usage guide

Guide for `mock_yolov26_sender.py`.

### What it does

The mock sender transmits synthetic `videotestsrc` imagery as
H.264/RTP/UDP and, independently, publishes a simulated face-detection
boolean to a Redis Pub/Sub channel. There is no camera, model, DPU, or
inference of any kind, and the synthetic pixels have no relationship to
the published boolean. It lets the receiver integration be tested
independently of camera/DPU inference, using the same transport
contract as the real sender.

### Requirements

On the sender:

- Linux, Python >= 3.9. All imports are from the standard library; no
  requirements.txt or pip installs are needed for this mock.
- `gst-launch-1.0` and `gst-inspect-1.0` on `PATH`, plus the GStreamer
  factories `videotestsrc`, `capsfilter`, `queue`, `videoconvert`,
  `h264parse`, `rtph264pay`, `fakesink`, `udpsink`, and a working
  encoder such as `x264enc`.
- On the Kria board, a private GStreamer prefix under
  `~/kv260-network-deps` is auto-detected; no manual environment
  exports are needed.

On the receiver side: a video receiver with an H.264 decoder and a
reachable Redis server. The script does not create or start Redis and
does not supply a receiver. `redis-cli` is an optional diagnostic tool,
not a sender dependency.

### Run

From `scripts/` (or any folder containing the script):

```sh
python3 mock_yolov26_sender.py \
    --pi-host PI_IP \
    --encoder x264 \
    --pattern ball \
    --face-mode alternate \
    --video-port 5002 \
    --redis-channel face_detection_mock_test \
    --redis-interval-seconds 1 \
    --duration-seconds 30
```

Replace `PI_IP` with the actual Pi IP address. The sender stops 30
seconds after startup completes; Ctrl+C stops it earlier.
`--pattern ball` with `--face-mode alternate` produces changing video
and a changing status; after missed deadlines the boolean may repeat a
value rather than strictly alternating each delivered message.

### Video on the Pi

The example uses UDP port 5002; sender and receiver ports must match.
Nominal video is 848x480 at 30 FPS — a nominal rate, not a guaranteed
measured or displayed throughput. The RTP caps are exactly:

```
application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96
```

No receiver implementation or pipeline is supplied here.

### Redis

By default the sender publishes to the Pi host on TCP port 6379; use
`--redis-host REDIS_IP` and `--redis-port PORT` to point at a different
endpoint. The example channel is `face_detection_mock_test`; no channel
creation is needed — the subscriber only needs to subscribe to the same
name. If Redis runs on the Pi, an optional check on the Pi is:

```sh
redis-cli -h 127.0.0.1 -p 6379 SUBSCRIBE face_detection_mock_test
```

Payloads are exactly `{"face_detected":true}` or
`{"face_detected":false}`, published roughly every second
(`--redis-interval-seconds` adjusts it). Publishing is freshness-gated
and best-effort: missing messages mean unavailable, not false, nothing
is replayed, and the status stream is independent of the video. The
Redis endpoint must stay private/source-restricted; plain TCP only, no
AUTH or TLS support.

### Defaults vs. this example

The defaults are video port 5000 and channel `face_detection`; the
example deliberately uses port 5002 and a test channel so it does not
collide with live automation — coordinate test endpoints with the
receiver developer. Use `--pattern snow --face-mode random` for random
inputs, or `--face-mode true` / `--face-mode false` to force a state.

A clean exit 0 means local completion only; validate the displayed
video and the subscriber messages independently. See `--help` for all
options.
