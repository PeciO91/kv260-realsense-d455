# YOLOv26 person / face detection on the KV260 DPU

Real-time INT8 detection of **Person** and **Human face** from an Intel
RealSense D455, running on the Kria KV260 DPU, with a distance in metres
attached to every box. One shared core, four output modes.

```
inference/yolov26/
├── core/    camera, letterbox, INT8 quantization, VART runner, decode, depth, overlay, telemetry
├── mjpeg/   browser MJPEG                      → mjpeg/README.md
├── rpi/     Raspberry Pi H.264/RTP + Redis     → rpi/README.md
└── hdmi/    local monitor, X11 or DRM/KMS      → hdmi/README.md
```

| Mode | Binary | Output | Use when |
|------|--------|--------|----------|
| [mjpeg](mjpeg/README.md) | `build/live_yolov26_realsense` | HTTP MJPEG on port 8081 | You want to watch from a laptop on the LAN |
| [rpi](rpi/README.md) | `build/live_yolov26_rpi` | H.264/RTP/UDP video + Redis `face_detected` | A Raspberry Pi consumes the stream and the face flag |
| [hdmi](hdmi/README.md) | `build/live_yolov26_hdmi` | Fullscreen on the board's own monitor | Standalone demo, lowest latency (`--renderer kms`) |

## Pipeline

```
RealSense D455 (USB 3, librealsense RSUSB)
  color 848x480@30 YUYV → BGR8        depth 424x240@30 Z16
        │                                   │
        ▼                                   │ (raw, not aligned)
  letterbox to 640x640 (RGB, INTER_LINEAR,  │
  pad 114) + INT8 quantize (256-entry LUT)  │
        │                                   │
        ▼                                   │
  DPU (VART RunnerExt, one DPU subgraph)    │
  6 output tensors: box + class × strides 8/16/32
        │                                   │
        ▼                                   │
  decode (REG_MAX=1 LTRB, sigmoid, no NMS)  │
        │                                   │
        ▼                                   ▼
  detections ──► 5x5 sample grid over the central 35% of each box,
                 mapped colour→depth with intrinsics/extrinsics, median Z
        │
        ▼
  capacity-1 latest-frame handoff  (a newer frame replaces a pending one)
        │
        ▼
  output thread: annotate → MJPEG | H.264 + Redis | HDMI (X11 or KMS)
```

**Two threads.** The capture thread owns the camera, DPU and depth mapping. The
output thread owns drawing and delivery. They are joined by a queue of
capacity 1, so when the output side is slower the pending frame is replaced
rather than queued. Latency stays bounded and the cost shows up as dropped
frames instead of a growing backlog. This is implemented once in
`core/yolov26_realsense_core.{hpp,cpp}` (`LatestFrameHandoff`) and shared by all modes.

## Model contract

The applications load an externally compiled xmodel through Vitis AI 3.0
XIR/VART `RunnerExt` and validate the graph at startup. Arbitrary YOLO exports
are **not** supported. The expected model is:

| Item | Value |
|------|-------|
| Input | `[1, 640, 640, 3]` INT8, fix_point 6 |
| Classes | 2: `Person` (green), `Human face` (orange) |
| Heads | split one2one: for each stride 8 / 16 / 32 (80², 40², 20² cells) one 4-channel box tensor and one 2-channel class tensor, matched by spatial shape (not by name or order) |
| Box decode | anchor-free, `REG_MAX = 1` LTRB distances |
| Score | stable sigmoid, thresholded in quantized INT8 logit space |
| Post-processing | none beyond top-N by score. The one2one head needs **no NMS** |

Default path `yolov26n_facehuman_kria.xmodel` is resolved relative to the
current directory. Pass `--model PATH` otherwise. Compiled models are produced
outside this repo and are git-ignored.

> Quantization note: the class-head output tensors are coarsely quantized
> (fix_point 2 at stride 8, 0 at strides 16 and 32), so scores sit on a few
> discrete levels. With a 0.30 threshold a person box flickers on and off; with
> 0.25 the person is present in 96.8 % of frames against 66.3 % at 0.30 in a
> 45 s stationary test. The default is currently 0.30.

## Depth

Each detection gets a distance without aligning the whole depth frame.
`core` samples a 5x5 grid over the central 35 % of the box in the colour
image, maps each point into the raw Z16 image through the camera calibration,
and reports the median of the valid readings (0.1 – 10 m). Full-frame
alignment is what the early Python tests and the DenseBox diagnostic do, and
it is the cost this scheme removes.

## Shared options

Every mode accepts these (see each mode's README for its own options):

| Option | Default | Meaning |
|--------|---------|---------|
| `--model PATH` | `yolov26n_facehuman_kria.xmodel` | compiled model |
| `--confidence F` | 0.30 | threshold in `[0, 1]` |
| `--max-detections N` | 50 | cap after sorting by score |
| `--depth-width W` / `--depth-height H` | smallest Z16 at 30 FPS (424x240) | request another depth profile |
| `--show-depth-roi true\|false` | false | draw the blue depth-sampling rectangle |
| `--show-confidence true\|false` | true | include the numeric score in labels |
| `--telemetry true\|false` | false | overlay + once-per-second `[telemetry]` log line |

## Telemetry

With `--telemetry true` the apps report, per 1-second window: output FPS,
mean stage times (`acquire`, `preprocess`, `dpu`, depth post-processing, and for
HDMI `wait`, `draw`, `show`), CPU %, RAM, and temperature. **`e2e` is host
pipeline latency** from the moment the frame is requested from the camera to
the moment it is handed to the display or socket. It is not glass-to-glass
latency and does not include camera exposure or monitor scan-out.

## Build and run

```sh
sh scripts/build_live_yolov26.sh           # mjpeg
sh scripts/build_live_yolov26_rpi.sh       # rpi
sh scripts/build_live_yolov26_hdmi.sh      # hdmi
./build/live_yolov26_realsense --model /home/root/models/yolov26n_facehuman_kria.xmodel
```

Offline tests for the control logic, option parsing and launchers:
`python3 -m unittest discover -s tests` (see [`tests/README.md`](../../tests/README.md)).
