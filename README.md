# KV260 RealSense D455

Intel RealSense D455 camera pipeline for the AMD/Xilinx Kria KV260.

## Current status

Working:

- Intel RealSense D455 detection
- RGB capture at 640x480
- librealsense / pyrealsense2
- RSUSB backend on Xilinx 2022.2
- Stable pipeline start and stop
- JPEG frame capture
- Live MJPEG stream over HTTP

## Camera pipeline

RealSense D455
    |
    | USB
    v
librealsense RSUSB
    |
    v
pyrealsense2
    |
    v
RGB frame
    |
    +----> OpenCV / JPEG
    |
    +----> HTTP MJPEG stream
    |
    v
Future YOLO26 preprocessing
    |
    v
KV260 DPU
    |
    v
Person / Human Face detections

## Camera test

Run:

    python3 camera/camera_test.py

A test frame is saved as:

    d455_test.jpg

## Live preview

Run:

    python3 camera/live_stream.py

Find the KV260 IP address:

    hostname -I

Then open on another computer:

    http://KV260_IP:8080/

The default network preview runs at 15 FPS with JPEG quality 75.

Alternative:

    python3 camera/live_stream.py \
        --port 8080 \
        --stream-fps 20 \
        --jpeg-quality 80

## DenseBox live stream

Build and run the DPU-backed RGB/depth stream:

    ./scripts/build_live_densebox.sh
    ./inference/live_densebox_realsense

Open `http://KV260_IP:8081/`. The page removes browser margins and scales the
MJPEG image to the available viewport with its aspect ratio preserved.

At startup the application enumerates native D455 YUYV color and Z16 depth
profiles and verifies/selects the configured 848x480 YUYV profile at 30 FPS.
The smallest Z16 depth profile at the same FPS is selected to reduce depth
transport and mapping cost without lowering RGB resolution; this is 424x240 at
30 FPS on the tested camera. A calibrated 5x5 grid maps points from each
face's central color ROI into the raw depth image using RealSense
intrinsics/extrinsics, avoiding
full-frame alignment.
The selected RGB and depth profiles are printed at startup.

List the native profiles seen by the application:

    ./inference/live_densebox_realsense --list-profiles

JPEG quality defaults to 90.
It can be changed without rebuilding:

    ./inference/live_densebox_realsense --jpeg-quality 75

A supported depth mode at the selected RGB FPS can also be requested for
quality/performance comparisons:

    ./inference/live_densebox_realsense \
        --depth-width 640 \
        --depth-height 480

## YOLOv26 face/human live stream

Build and run the DPU-backed YOLOv26 Person/Human face stream:

    ./scripts/build_live_yolov26.sh
    ./inference/live_yolov26_realsense

The default model path is `yolov26n_facehuman_kria.xmodel` resolved
relative to the working directory, so run the binary from the folder
containing the xmodel or pass an explicit path:

    ./inference/live_yolov26_realsense \
        --model /path/to/model.xmodel \
        --confidence 0.25 \
        --max-detections 100

- `--model PATH`: xmodel file (default
  `yolov26n_facehuman_kria.xmodel`).
- `--confidence FLOAT`: detection confidence in `[0, 1]` (default 0.1).
  Thresholding happens in quantized INT8 logit space.
- `--max-detections N`: positive cap on reported boxes (default 300).
  Detections are sorted by score and truncated; no NMS is applied.

All existing live-stream options (`--port`, `--jpeg-quality`, `--depth-width`,
`--depth-height`, `--list-profiles`) work unchanged.

The app talks to the deployed VART/Vitis AI 3.0 runtime directly through
XIR/VART `RunnerExt`: it
deserializes the graph, selects the single `DPU` child subgraph, and
validates the raw split one2one output contract (one 4-channel box tensor
and one 2-channel class tensor per stride-8/16/32 level, matched by
spatial shape, not tensor order or names). Startup prints every input and
output tensor's name, shape, data type, and fix point. Preprocessing
replicates the reference letterbox (RGB, INTER_LINEAR, 114 border) and
INT8 quantization via a cached 256-entry LUT. Decoding is anchor-free
`REG_MAX = 1` LTRB distance decode with stable sigmoid scores and no NMS.

Annotated classes are `Person` (green) and `Human face` (orange); each
box also shows the calibrated 5x5 median depth of its central ROI. Open
`http://KV260_IP:8081/` to view the stream.

The app runs a two-stage pipeline: the main thread owns the camera, DPU
inference, and depth mapping, while a renderer thread owns annotation
drawing, JPEG encoding, and MJPEG sending. A bounded latest-frame handoff
with queue capacity 1 sits between them: publishing onto an occupied
slot replaces the pending frame, which avoids latency buildup when the
renderer falls behind. Terminal telemetry reports independent pipeline
and stream FPS, per-stage timings for `acquire`, `preprocess`, `dpu`,
`depth_postprocess`, `overlay`, `jpeg`, and `send`, cumulative dropped
frames plus per-window drop rate, and true end-to-end latency from frame
capture to completed send.

## Mock YOLOv26 network sender

### Quick start

The mock sender streams synthetic `videotestsrc` video as
H.264/RTP/UDP and periodically publishes a simulated, independent
face-detection boolean over Redis Pub/Sub. It needs no camera, model,
or DPU, and the synthetic pixels have no relationship to the published
boolean. It requires only Python >= 3.9 (standard library; no pip
packages) on the sender, plus the GStreamer command-line tools and
encoder plugins; the receiver needs H.264 decoding and a reachable
Redis server.

Run from the repository root (replace `PI_IP` with the Pi address):

    python3 scripts/mock_yolov26_sender.py \
        --pi-host PI_IP \
        --encoder x264 \
        --pattern ball \
        --face-mode alternate \
        --video-port 5002 \
        --redis-channel face_detection_mock_test \
        --redis-interval-seconds 1 \
        --duration-seconds 30

This example uses UDP video port 5002 and Redis TCP port 6379; the
video port must match the receiver and the Redis channel name must be
shared between sender and subscriber. The detailed usage guide is in
[Mock sender usage guide](scripts/README.md).

### Reference

`scripts/mock_yolov26_sender.py` is a standard-library-only mock of the
future network sender. It transmits synthetic `videotestsrc` imagery as
H.264/RTP/UDP and periodically publishes a randomized face boolean to
Redis, using the same wire contract planned for real inference. It does
not run the camera, the model, or the DPU, and it changes nothing about
the real streams or the receiver (owned by the Pi developer).

Prerequisites:

- Python >= 3.9, standard library only. No OpenCV, NumPy, redis-cli,
  hiredis, redis-py, or model files needed.
- `gst-launch-1.0` / `gst-inspect-1.0` on PATH plus the usual
  videotestsrc/capsfilter/queue/videoconvert/h264parse/rtph264pay/
  fakesink/udpsink factories and a working H.264 encoder
  (`x264enc` or `omxh264enc`).

Wire contract (must match on both ends):

- Video: unicast RTP/UDP to the configured host, UDP port 5000 default;
  H.264 RFC 6184, MTU 1200, 848x480 at nominal 30 FPS (configurable via
  `--width`/`--height`/`--fps`), encoder target 4000 kbit/s, no B-frames,
  keyframe every 30 frames, SPS/PPS repetition configured with
  `config-interval=1`. Receiver caps:
  `application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96`.
  Best-effort UDP: no session negotiation, RTCP, retransmission or
  acknowledgement.
- Redis: sender is a PUBLISH client only (RESP2 over TCP, no AUTH/TLS/db
  select). Server runs on the receiver side; default port 6379, default
  channel `face_detection`. Exact payload `{"face_detected":true}` or
  `{"face_detected":false}` every 1.0 s (configurable >= 0.1 s), repeated
  while the synthetic state is fresh (< 2.0 s old). No replay or catch-up
  bursts after reconnect; missing messages mean unavailable, not false.
  A successful PUBLISH to zero subscribers is valid. The Redis server
  must stay private/source-restricted; no CLI passwords are used.
- Video and Redis status are independent streams, not frame-paired.
- Future real meaning: the boolean is true when the latest successful
  inference frame contains at least one valid class-1 Human-face
  detection, else false; nothing is published before the first result or
  while the state is stale. This mock's value is random/forced and is NOT
  computed from the synthetic pixels; the real annotated stream is
  deferred work.

Options (see `--help` for bounds):

`--pi-host` (required), `--video-port` (5000),
`--encoder auto|omx|x264` (auto), `--bitrate-kbps` (4000),
`--width`/`--height` (848x480, even 2..8192), `--fps` (30, 1..240),
`--redis-host` (defaults to the Pi host), `--redis-port` (6379),
`--redis-channel` (face_detection), `--redis-interval-seconds` (1.0,
>=0.1), `--face-stale-seconds` (2.0), `--redis-timeout-ms` (200, max
10000), `--pattern snow|ball` (snow), `--face-mode
random|true|false|alternate` (random), `--seed`, `--duration-seconds`,
`--deps-prefix`.

Run on the Kria (software x264 in the private prefix is auto-detected at
`~/kv260-network-deps`, no manual exports needed):

    python3 scripts/mock_yolov26_sender.py --pi-host PI_LAN_IP --encoder x264

With an explicit dependency prefix:

    python3 scripts/mock_yolov26_sender.py --pi-host PI_LAN_IP \
        --encoder x264 --deps-prefix /home/root/kv260-network-deps

Deterministic acceptance run against a TEST channel/port only (never
flip the live `face_detection` channel during automation; coordinate the
UDP test port with the receiver developer):

    python3 scripts/mock_yolov26_sender.py --pi-host PI_LAN_IP \
        --video-port 5002 --encoder x264 --face-mode alternate \
        --redis-channel face_detection_mock_test \
        --redis-interval-seconds 0.3 --duration-seconds 3

Notes:

- `--encoder auto` (default) tries `omxh264enc` first and falls back
  visibly to `x264enc`. The OMX path currently fails on the Kria
  (`/dev/allegro` nodes absent), so `--encoder x264` is recommended.
- The dependency prefix environment is applied only to child GStreamer
  processes (LD_LIBRARY_PATH, GST_PLUGIN_PATH(_1_0), private
  GST_REGISTRY); the caller's environment and system config are not
  modified. Without a prefix the system plugins are used (works on the
  WSL/dev PC).
- `--face-mode alternate` toggles at the nominal report cadence; after
  missed deadlines it may repeat a value rather than strictly toggling
  each delivered message. `--seed` seeds the face-value generator only,
  not GStreamer `snow` pixels, and wall-clock sampling makes published
  timing not bit-for-bit deterministic.
- A local "MOCK" banner is printed at startup; nothing marks the mock on
  the wire. `snow` is random noise and compresses poorly, so it is not a
  bitrate benchmark; `--pattern ball` is a calmer diagnostic.
- `--duration-seconds` starts counting after preflight and video launch.
  DNS resolution can outlast the socket timeout; a clean exit 0 means
  local completion only - it does not prove UDP delivery to the Pi or
  Redis health (watch the published/subscriber and error logs).
- Tests (no GStreamer/Redis needed; fakes are used):

      python3 -m unittest discover -s tests -p 'test_mock_yolov26_sender.py'

  Pi-side reception/decoding, real Redis connectivity, and sustained
  combined CPU load remain to be verified with the receiver developer.

## Native YOLOv26 RTP sender

`inference/live_yolov26_rpi` is the native C++ sibling of the MJPEG
application. It runs the same D455 + VART/DPU inference and depth
annotation pipeline, then pushes annotated frames through appsrc into
a GStreamer H.264/RTP/UDP pipeline aimed at a Raspberry Pi receiver,
and publishes the face-detection state on Redis Pub/Sub.

Wire contract (same as the Python mock):

- RTP: `application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96`,
  MTU 1200 bytes.
- H.264: byte-stream, access-unit aligned, baseline profile,
  `config-interval=1`.
- Redis: RESP2 `PUBLISH` on the configured channel with payload exactly
  `{"face_detected":true}` or `{"face_detected":false}`.

Prerequisites on the KV260:

- RealSense D455 over the existing RSUSB build (`/root/librealsense-rsusb`),
  the VART/XIR toolchain, and `/home/root/yolov26n_facehuman_kria.xmodel`
  (or `--model`).
- GStreamer 1.18 runtime + plugins plus `hiredis`, staged under a
  dependency prefix such as `/home/root/kv260-network-deps`
  (`lib/gstreamer-1.0` must exist there). Both the build script and the
  launcher auto-discover `$HOME/kv260-network-deps` or take
  `--deps-prefix DIR`. The build script sets `PKG_CONFIG_PATH` only
  for the compile and bakes an `rpath` into the binary; the launcher
  additionally sets `LD_LIBRARY_PATH`, `GST_PLUGIN_PATH(_1_0)` and an
  arch-specific registry file privately for the child process only.
  Nothing is installed globally.
- A H.264 encoder: `omxh264enc` (preferred when the VCU device tree is
  present) or `x264enc`. `--encoder auto` tries OMX first and falls
  back to x264; on this board `/dev/allegroIP` is absent, so
  `--encoder x264` is recommended. Forcing a missing/broken encoder
  fails the run.

Build on the board from the repository root:

    sh scripts/build_live_yolov26.sh        # MJPEG application
    sh scripts/build_live_yolov26_rpi.sh    # native RTP sender

`build_live_yolov26_rpi.sh` also accepts `--deps-prefix DIR` and
`--rs-prefix DIR` (alternate librealsense install; default
`/root/librealsense-rsusb`) and requires the pkg-config modules
`opencv4`, `gstreamer-1.0`, `gstreamer-app-1.0`, `gstreamer-video-1.0`,
`hiredis` plus `librealsense2` under the RS prefix.

Run:

    python3 scripts/run_live_yolov26_rpi.py \
        --pi-host <PI_IP> \
        --model /home/root/yolov26n_facehuman_kria.xmodel \
        --encoder x264 \
        --video-port 5000 \
        --redis-host <REDIS_IP> \
        --redis-port 6379 \
        --redis-channel face_detection

The launcher accepts `--deps-prefix DIR` and `--binary PATH` ahead of
the sender arguments; a literal `--` before the sender arguments also
works. The same binary can be invoked directly if the dependency
environment is already set:

    ./inference/live_yolov26_rpi \
        --pi-host <PI_IP> --encoder x264 \
        --model /home/root/yolov26n_facehuman_kria.xmodel

Options (see `--help` for bounds): `--pi-host` (required; IPv4, IPv6
including bracketed/scoped literals, or DNS name), `--video-port`
(default 5000), `--encoder auto|omx|x264` (default auto),
`--bitrate-kbps` (default 4000), `--redis-host` (defaults to the Pi
host; IPv6 ok), `--redis-port` (default 6379), `--redis-channel`
(default `face_detection`), `--redis-interval-seconds` (default 1.0,
minimum 0.1), `--face-stale-seconds` (default 2.0, minimum ~1 ns),
`--redis-timeout-ms` (integer, default 200, range 1..10000),
`--duration-seconds` (optional; all second-valued options are capped at
630720000 s), `--telemetry true|false` (explicit value required,
default false), plus the shared inference options `--model`,
`--confidence` `[0,1]`, `--max-detections` (default 300),
`--depth-width`, `--depth-height`, `--list-profiles`.

Behaviour notes:

- The face state is the latest successful inference result for class 1
  ("Human face") with a valid in-bounds box; it is published on the
  report interval only while fresh (< `--face-stale-seconds`). There
  is no initial publication, stale results go silent, reconnects re-read
  the freshest state before writing, and missed report deadlines are
  skipped rather than replayed.
- Video RTP and the Redis status channel are independent channels:
  Redis connection/publish failures never stop video streaming, and
  reconnect attempts back off 1/2/4/5 s in the background. They are
  not fully fault-isolated lifecycles: a fatal video or inference
  error still terminates the application and its status reporting.
- The camera profile is color 848x480@30. By default, depth uses the
  smallest Z16 profile at the color FPS (424x240 on the tested D455);
  `--depth-width`/`--depth-height` request an explicit matching profile.
- RTP timestamps follow host acquisition time, retaining gaps when
  frames are dropped. The nominal 30 FPS caps are not a throughput or
  Pi display-rate guarantee; raw-frame buffering is bounded and
  latest-only.
- `--duration-seconds` starts counting only after encoder preflight,
  camera start, warmup, and pipeline start complete. SIGINT/SIGTERM
  stop cleanly at any phase.
- `--help` runs entirely offline: no Pi host, model, camera, or
  network initialization. `--list-profiles` skips the Pi host and
  model checks but DOES open the camera to enumerate profiles.
- Video and Redis hosts are resolved to numeric addresses once at
  startup; a slow DNS server can therefore add startup latency before
  streaming begins. A Redis DNS failure marks status unavailable
  rather than publishing a stale/false state.
- The sender targets a private endpoint: no authentication or TLS on
  Redis, no destination NAT traversal.
- Exit 0 means local completion; it does not prove UDP delivery to a
  Pi or Redis health. When telemetry is enabled the overlay/log
  reports inference FPS, appsrc submissions/s, encoded AU/s, replaced
  frames and mean stage timings (acquire, preprocess, DPU,
  decode-depth, overlay, push preparation + admission, host
  start-to-enqueue).
- Known runtime limitation: a hung camera/DPU call inside VART is not
  cancellable; shutdown can block there the same way the MJPEG
  application does.

Tests (control fixture is stdlib-only; the transport fixture needs the
native GStreamer/hiredis development files and is skipped otherwise):

    python3 -m unittest discover -s tests -p 'test_live_yolov26_rpi.py'

Known limitation: the OMX path requires the VCU/Allegro device tree;
on boards where `/dev/allegro*` is absent, use `--encoder x264` or let
`auto` fall back.

## DenseBox telemetry

The live DenseBox stream draws a compact telemetry panel over the MJPEG
video. It updates approximately once per second and shows:

- **Stream FPS**: annotated JPEG frames successfully sent to the MJPEG client
  per second. This is the observable output rate.
- **Pipeline FPS**: completed camera, preprocessing, inference, sparse depth
  mapping, and drawing iterations per second.
- **DPU latency**: average time spent only in `detector->run(model_input)`.
- **DPU rate**: `1000 / DPU latency (ms)`. This is the accelerator's
  inference-throughput equivalent, not the complete application's frame rate.
- CPU utilization from `/proc/stat` deltas and used RAM from
  `MemTotal - MemAvailable`.
- PS and PL temperatures from the Xilinx AMS hwmon device when available.
- Board power, voltage, and current from the INA260 hwmon device when
  available.

Acquisition, resize, sparse depth mapping, overlay drawing, JPEG encoding,
HTTP send, and total-frame timing averages are reported in the console.
Hardware monitoring devices are discovered by their hwmon names; unavailable
metrics display as `N/A` and do not stop streaming.

## RealSense backend

The standard V4L2/UVC backend caused a kernel crash during
pipeline shutdown on the Xilinx 2022.2 kernel.

The project therefore uses the official librealsense RSUSB backend.

See:

    docs/realsense-rsusb.md

## Future work

- Depth stream
- RGB/depth alignment
- DPU-compatible YOLO26n
- INT8 Vitis AI inference
- Person and Human Face detection
- Detection overlays
- Distance estimation from depth data

## RGB + depth test

The D455 can capture RGB and depth simultaneously.

Run:

    python3 camera/depth_test.py

The test starts:

- RGB: 640x480 @ 30 FPS
- Depth: 640x480 @ 30 FPS

It captures 30 frames and prints the depth value at the center of the
depth image in metres.

Example:

    Frame 12: RGB 640x480, Depth 640x480, center distance = 1.842 m

This test does not yet align the depth image to the RGB camera.
RGB-to-depth alignment will be added separately before YOLO integration.
