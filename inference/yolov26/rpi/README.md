# YOLOv26 → Raspberry Pi (H.264/RTP + Redis)

`live_yolov26_rpi` runs the same D455 + DPU + depth pipeline as the other
modes, pushes the annotated frames through GStreamer `appsrc` as
H.264/RTP/UDP to a Raspberry Pi, and independently publishes a face-detected
flag on Redis Pub/Sub. The Pi-side receiver and Redis server are maintained
outside this repository. Overview of the shared pipeline:
[`../README.md`](../README.md).

Files: `live_yolov26_rpi.cpp` (lifecycle, capture, render, status worker),
`yolov26_rpi_control.{hpp,cpp}` (CLI validation, freshness, report/backoff
scheduling, telemetry), `yolov26_rpi_transport.{hpp,cpp}` (host resolution,
deadline-bounded Redis publishing, GStreamer encoder/video transport).

## Wire contract

Identical to the Python mock ([`scripts/mock_yolov26_sender.py`](../../../scripts/README.md)).

- **Video:** `application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96`,
  unicast UDP (default port 5000), MTU 1200, H.264 baseline, byte-stream,
  access-unit aligned, `config-interval=1`, 4000 kbit/s default, 848x480 nominal 30 FPS.
  Best effort: no RTCP, retransmission or negotiation.
- **Status:** RESP2 `PUBLISH` (default TCP 6379, channel `face_detection`) with payload
  exactly `{"face_detected":true}` or `{"face_detected":false}`, once per second.
  `true` means the latest successful inference frame had at least one valid
  `Human face` box. Stale (> 2 s) or unavailable state is **silence, not false**.
  No AUTH or TLS: use a private, source-restricted Redis.
- Video and status are independent and not frame-paired.

## Prerequisites on the KV260

- RSUSB librealsense (`/root/librealsense-rsusb`), VART/XIR runtime, the xmodel.
- GStreamer 1.18 runtime + plugins and `hiredis` under a dependency prefix such as
  `/home/root/kv260-network-deps` (must contain `lib/gstreamer-1.0`). The build
  script and launcher auto-discover `$HOME/kv260-network-deps` or take
  `--deps-prefix DIR`. The prefix is used only for the compile (`PKG_CONFIG_PATH`),
  baked into the binary as an `rpath`, and applied privately to the child process by
  the launcher. Nothing is installed globally.
- An H.264 encoder. `omxh264enc` needs the VCU device tree (`/dev/allegro*`),
  which is absent on this board, so use `--encoder x264`. `--encoder auto` tries OMX
  first and falls back visibly.

## Build and run

```sh
sh scripts/build_live_yolov26_rpi.sh [--deps-prefix DIR] [--rs-prefix DIR]
python3 scripts/run_live_yolov26_rpi.py \
    --pi-host <PI_IP> \
    --model /home/root/models/yolov26n_facehuman_kria.xmodel \
    --encoder x264 --video-port 5000 \
    --redis-host <REDIS_IP> --redis-port 6379 --redis-channel face_detection
```

The launcher (`--deps-prefix DIR`, `--binary PATH` before the sender arguments;
a literal `--` also works) sets the plugin environment and execs
`build/live_yolov26_rpi`. The binary can be run directly if the environment is
already set. The build requires the pkg-config modules `opencv4`, `gstreamer-1.0`,
`gstreamer-app-1.0`, `gstreamer-video-1.0`, `hiredis`, and `librealsense2` under the RS prefix.

## Options

Shared inference options: [`../README.md`](../README.md#shared-options). Mode-specific
(see `--help` for bounds):

| Option | Default | |
|--------|---------|---|
| `--pi-host HOST` | required | IPv4, IPv6 (bracketed/scoped) or DNS name |
| `--video-port N` | 5000 | UDP port, must match the receiver |
| `--encoder auto\|omx\|x264` | auto | |
| `--bitrate-kbps N` | 4000 | |
| `--redis-host HOST` | the Pi host | |
| `--redis-port N` | 6379 | |
| `--redis-channel NAME` | `face_detection` | |
| `--redis-interval-seconds S` | 1.0 | minimum 0.1 |
| `--face-stale-seconds S` | 2.0 | |
| `--redis-timeout-ms N` | 200 | 1..10000 |
| `--duration-seconds S` | run until stopped | counts after encoder preflight, camera start and warm-up |
| `--telemetry true\|false` | false | explicit value required |

## Behaviour notes

- Redis connection or publish failures never stop the video. Reconnects back off
  1/2/4/5 s in the background and missed deadlines are skipped, not replayed. A fatal
  video or inference error still terminates the whole application.
- RTP timestamps follow host acquisition time, so dropped frames leave gaps. Raw-frame
  buffering is bounded and latest-only. 30 FPS is a nominal cap, not a Pi display-rate guarantee.
- Hostnames are resolved once at startup. A Redis DNS failure marks status unavailable.
- Exit code 0 means local completion only. It does not prove UDP delivery or Redis health.
- `--help` is fully offline. `--list-profiles` skips the Pi host and model checks but does
  open the camera.
- A hung camera or DPU call inside VART cannot be cancelled, so shutdown can block there.
- Telemetry (`--telemetry true`): inference FPS, appsrc submissions/s, encoded AU/s,
  replaced frames, and mean stage times (acquire, preprocess, DPU, decode-depth,
  overlay, push admission, host start-to-enqueue).

## Testing without a camera or DPU

`scripts/mock_yolov26_sender.py` sends the same contract with synthetic video and a
random face flag. See [`scripts/README.md`](../../../scripts/README.md). Offline tests:

```sh
python3 -m unittest discover -s tests/yolov26/rpi
```

The transport fixture needs the GStreamer and hiredis development files and is
skipped without them.
