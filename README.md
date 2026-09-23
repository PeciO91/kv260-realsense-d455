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

The app talks to Vitis AI 3.5 directly through XIR/VART `RunnerExt`: it
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
