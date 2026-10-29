# Real-time person & face detection with depth on the Kria KV260

An Intel RealSense D455 feeds an INT8 **YOLOv26n** detector running on the
KV260's DPU. Every detected **person** and **face** gets a distance in metres,
and the annotated video is delivered to a browser, a Raspberry Pi, or straight to
a 1080p monitor. The C++ pipeline reaches the camera's full **30 FPS** with ~45 ms
host latency while using a third of the CPU.

The main engineering result is the fullscreen 1920x1080 display path. Measured on
the board, it went from **8.7 FPS / 172 ms / 77 % CPU** to **29.5 FPS / 45 ms / 33 % CPU**
in five measured steps, by finding and removing one bottleneck at a time
([details below](#performance-optimisation-at-1920x1080)).

| | Start (X11, letterbox canvas) | End (DRM/KMS, word-level nearest) | Change |
|---|---|---|---|
| Output FPS | 8.7 | 29.5 | 3.4× |
| Host latency (mean) | 172 ms | 45 ms | 3.8× lower |
| CPU load | 77 % | 33 % | 2.3× lower |

## Contents

- [Hardware](#hardware) · [Repository layout](#repository-layout) · [Workflow](#workflow)
- [Architecture](#architecture) · [Output modes](#output-modes)
- [Performance optimisation at 1920x1080](#performance-optimisation-at-1920x1080)
- [Earlier work](#earlier-work) · [Tests](#tests)

## Hardware

| Part | Detail |
|------|--------|
| Board | AMD/Xilinx Kria KV260 (Zynq UltraScale+ MPSoC, 4× Cortex-A53), PetaLinux 2022.2, kernel 5.15.36 |
| Accelerator | DPU `DPUCZDX8G_ISA1_B4096`, 1 core @ 300 MHz, Vitis AI 3.0 (VART / XIR) |
| Camera | Intel RealSense D455 (stereo depth + RGB), USB 3 (5000 Mbit/s) |
| Display | Samsung S24D300, 1920x1080 over DisplayPort (zynqmp DPSUB) |
| Model | YOLOv26n, 2 classes (Person, Human face), INT8, 640x640 input; compiled outside this repo |

**D455 streams used.** The camera offers colour up to 1280x800@30 and depth up to
1280x720@30 (depth also 848x480@90 and 424x240@90; colour 848x480@60 — full list with
`--list-profiles`). The application uses:

| Stream | Profile | Why |
|--------|---------|-----|
| RGB | **848x480 @ 30 FPS**, YUYV (native) → BGR8 | enough resolution for a 640x640 model input, low USB and conversion cost |
| Depth | **424x240 @ 30 FPS**, Z16 | smallest depth profile at the colour frame rate; depth is only sampled at a few points per detection |

Depth is **not** aligned to the colour frame. For each box the app samples a 5x5 grid
over its central 35 %, maps those points into the raw depth image with the
camera intrinsics/extrinsics, and takes the median (valid range 0.1 – 10 m). The early
Python tests aligned the whole frame; the sparse mapping replaced that.

The camera runs through librealsense's **RSUSB** backend, not the kernel UVC path:
the default backend triggered a kernel Oops in `uvc_stop_streaming()` when the
pipeline stopped ([`docs/realsense-rsusb.md`](docs/realsense-rsusb.md)).

## Repository layout

```
inference/
├── yolov26/                 ← the main application
│   ├── core/                shared: camera, letterbox, INT8, DPU runner, decode, depth, overlay
│   ├── mjpeg/               browser output                           (README)
│   ├── rpi/                 Raspberry Pi H.264/RTP + Redis           (README)
│   └── hdmi/                local monitor, X11 or DRM/KMS            (README)
└── densebox/                first DPU experiment, frozen             (README)
scripts/                     build scripts, launchers, mock sender    (README: catalogue of every script)
tools/camera_bringup/        Python camera diagnostics                (README)
tests/                       host-side tests, mirrors inference/yolov26 (README)
docs/                        RSUSB backend notes, performance logs
models/                      placeholder; .xmodel files are not committed
build/                       compiled binaries (git-ignored)
```

Per-mode guides: [overview](inference/yolov26/README.md) ·
[mjpeg](inference/yolov26/mjpeg/README.md) · [rpi](inference/yolov26/rpi/README.md) ·
[hdmi (X11 + KMS)](inference/yolov26/hdmi/README.md) ·
[scripts](scripts/README.md) · [camera tools](tools/camera_bringup/README.md) ·
[tests](tests/README.md).

## Workflow

```
 train / export YOLOv26n  →  quantize + compile for the B4096 DPU  →  yolov26n_facehuman_kria.xmodel
        (outside this repo)                (Vitis AI 3.0)                       │
                                                                                ▼
 sh scripts/build_live_yolov26_hdmi.sh   →   ./build/live_yolov26_hdmi --model <xmodel> --renderer kms
   (on the KV260, g++ + pkg-config)                  (or the mjpeg / rpi build and binary)
```

1. **Camera:** install librealsense with the RSUSB backend ([`docs/realsense-rsusb.md`](docs/realsense-rsusb.md)). Check it with `python3 tools/camera_bringup/camera_test.py`.
2. **Model:** copy the compiled xmodel to the board (e.g. `/home/root/models/`). It must match the
   [model contract](inference/yolov26/README.md#model-contract); arbitrary YOLO exports are not supported.
3. **Build** the mode you want from the repository root: `sh scripts/build_live_yolov26.sh`
   (browser), `…_rpi.sh` (Pi), or `…_hdmi.sh` (monitor). Binaries land in `build/`.
4. **Run**, with `--telemetry true` to see FPS, per-stage times and CPU:

```sh
# Browser: open http://KV260_IP:8081/
./build/live_yolov26_realsense --model /home/root/models/yolov26n_facehuman_kria.xmodel

# Local monitor, lowest latency (stop X first so the app can own the display)
systemctl stop xserver-nodm
./build/live_yolov26_hdmi --renderer kms --model /home/root/models/yolov26n_facehuman_kria.xmodel
```

## Architecture

```
RealSense D455 ─ USB3 ─ librealsense (RSUSB)
   colour 848x480@30 ───────────────┐          depth 424x240@30 (raw Z16)
                                    ▼                          │
                      letterbox 640x640 (RGB, pad 114)         │
                      INT8 quantize (256-entry LUT)            │
                                    ▼                          │
                      DPU, one subgraph, 6 output tensors      │
                      (box + class at strides 8/16/32)         │
                                    ▼                          │
                      decode (REG_MAX=1 LTRB, no NMS)          │
                                    ▼                          ▼
                          boxes + classes ──► 5x5 depth grid → median distance
                                    │
            ┌───────────────────────┴───────────────────────┐
            │ capture thread                                │
            └───────────────► capacity-1 latest-frame handoff ◄── newer frame replaces a pending one
                                    ▼
            output thread: annotate →  MJPEG │ H.264/RTP + Redis │ HDMI X11 │ HDMI KMS
```

### Design choices

**Distance from a 5x5 grid, not from an aligned depth frame.** Every box needs one
number: how far away it is. The obvious way is to align the whole depth frame to the colour
frame (`rs2::align`) and read the box region, but that reprojects every depth pixel on the
A53 cores, every frame, to answer a question about ~25 pixels. The first live DenseBox app
did this and fell below the camera rate; replacing it with sparse mapping restored full
frame-rate throughput (commit `29043a5`). Now, for each detection:

1. Take the central 35 % of the box, which stays on the person or face and away from the
   background at the box edges.
2. Place a 5x5 grid of 25 points on it, in colour-image coordinates.
3. Map each point into the raw depth image with `rs2_project_color_pixel_to_depth_pixel`,
   which uses the camera's calibrated intrinsics and extrinsics.
4. Drop invalid readings (zero or out of range, 0.1 – 10 m) and take the **median**, so a few
   holes or background pixels cannot pull the distance.

Per box this is 25 projections and a selection over a fixed 25-element array, with no
per-frame allocation, instead of a full-frame warp. It is also why depth can be the small
424x240 stream.

**Two threads and a capacity-1 handoff.** Capture, DPU and depth never wait for the
display. The two sides are joined by a queue that holds exactly one frame, and a newer
frame replaces a pending one. If the output is slower than the camera, the cost shows up as
a dropped frame, not as a growing queue, so latency stays bounded instead of drifting up.

**Direct VART, own pre/post-processing.** The app loads the xmodel through XIR/VART
`RunnerExt` and validates the tensor contract at startup (pairing box and class tensors by
spatial shape, not by name or order). Letterboxing and INT8 quantization use a cached
256-entry lookup table, and decoding is plain thresholding and sorting.

**No NMS.** YOLOv26's one2one head produces non-duplicated boxes, so non-maximum
suppression is not needed.

**One core, several outputs.** All modes share `inference/yolov26/core`; each mode adds
only its own delivery code behind a small interface, which is how the display backends
(X11, KMS) could be swapped and measured against each other.

## Output modes

| Mode | Binary | Output | Guide |
|------|--------|--------|-------|
| **mjpeg** | `live_yolov26_realsense` | HTTP MJPEG for a browser (port 8081) | [mjpeg](inference/yolov26/mjpeg/README.md) |
| **rpi** | `live_yolov26_rpi` | H.264/RTP/UDP video to a Raspberry Pi + independent Redis `face_detected` status | [rpi](inference/yolov26/rpi/README.md) |
| **hdmi, X11** | `live_yolov26_hdmi` | OpenCV/GTK window on the board's X desktop | [hdmi](inference/yolov26/hdmi/README.md#x11-backend) |
| **hdmi, KMS** | `live_yolov26_hdmi --renderer kms` | Direct DRM scan-out, no X | [hdmi](inference/yolov26/hdmi/README.md#kms-backend) |

## Performance optimisation at 1920x1080

**Goal:** show the annotated stream fullscreen on a 1080p monitor at the camera's 30 FPS.

**Setup.** Everything below is the same hardware, model (640x640 INT8, DPU ≈ 11 ms per
frame, ≈ 90 FPS capacity), camera profile (848x480@30) and telemetry, measured with
`--telemetry true`. Each step was built from the commit that introduced it and run for
35 s; the first 5 s are dropped. 30 FPS is the camera's ceiling.
**Latency is host pipeline latency** (frame request → handed to the display), not
glass-to-glass. Method, caveats and raw logs: [`docs/performance/`](docs/performance/README.md).

Step numbers match the log files in `docs/performance/1080p/`.

**Reference points** (not fullscreen output; they show what the pipeline can do)

| # | Run | FPS | Latency (ms) | CPU % | What it tells us |
|---|-----|----:|-------------:|------:|------------------|
| 0 | MJPEG stream to a browser | 29.6 | 51.6 | 36 | the pipeline itself runs at camera rate |
| 3 | X11, 848x480 frame in a native-size window | 29.4 | 50.1 | 52 | X11 keeps up when it pushes 5× fewer pixels |

**Optimisation steps: fullscreen 1080p display**

| # | Step | FPS | Latency (ms) | CPU % |
|---|------|----:|-------------:|------:|
| 1 | X11, new 1920x1080 canvas every frame, then resize + copy | 8.7 | 172.4 | 77 |
| 2 | + reuse the canvas, `waitKey(1)` | 9.7 | 162.3 | 79 |
| 4 | X11, aspect-fit upscale to 1080p | 9.7 | 159.7 | 77 |
| 5 | KMS, bilinear scaling in software | 29.2 | 78.2 | 65 |
| 6 | KMS, nearest-neighbour at 16/32-bit word level | **29.5** | **45.3** | **33** |

### Bottleneck and steps

- **1 → 2, the display, not inference.** At 8.7 FPS the frame period is 115 ms, but
  preprocessing + DPU + depth take only ~37 ms. The rest is the GUI loop pushing a fresh
  6.2 MB canvas to X every frame. Reusing the canvas changed nothing, so allocation was
  not the cost.
- **4, it is the pixels.** Reference 3 shows the same X11 path keeps up with a 1.2 MB frame,
  and OpenCV's GTK backend cannot scale to fullscreen, so we upscale ourselves and are back
  to 9.7 FPS. An earlier ~30 FPS result was measured at 1024x768 (3.5× fewer pixels), so the
  numbers were re-measured at 1080p. X11 cannot reach camera rate here.
- **Dead end, hardware scaling.** The ZynqMP display has no scaler (both layers must share one
  source size), so the first KMS attempt showed a black screen. The fix was full-mode-size
  buffers with scaling in software.
- **5, drop X.** Writing straight into DRM buffers reaches camera rate, but the software
  bilinear resize costs ~30 ms per frame on the A53 (65 % CPU, 78 ms).
- **6, cheaper scaling.** Convert at native size, then upscale by copying whole 16/32-bit
  words (nearest-neighbour): display time 30.0 → 7.1 ms, CPU 65 → 33 %, latency 78 → 45 ms.
  `--kms-scaling linear` keeps the smoother option.

### Why latency dropped but FPS barely moved (steps 5 → 6)

FPS and latency measure different things. **FPS is throughput**: it is set by the slowest
*repeating* stage, and here that is the camera, which delivers a frame every 33 ms. **Latency
is the time one frame spends travelling through the pipeline**: it is the *sum* of
everything that frame goes through, including time spent waiting.

In step 5 the display thread needed ~30 ms to show a frame, still below the 33 ms frame
period, so it just kept up: 29.2 FPS, the camera limit. But every frame paid for it, and
it often arrived while the display was still busy with the previous one, so it also sat in
the handoff (`wait` 12.8 ms). Step 6 shrank the display work from 30.0 to 7.1 ms (−22.9 ms)
and the wait from 12.8 to 3.0 ms (−9.8 ms): together −32.7 ms, which is the whole
78.2 → 45.3 ms drop in latency. The camera still produced one frame per 33 ms, so FPS stayed
essentially where it was (29.2 → 29.5).

An assembly-line analogy: making the last station faster shortens the time a part takes to
get through the line, but not how many parts leave per minute, because the first station
still sets the pace. What the faster display bought instead was **headroom**: CPU load fell
from 65 % to 33 %, and the display no longer limits frame rate even if load rises.

**Where the time goes now (step 6, per frame):** the capture thread spends 18.6 ms on
preprocessing (CPU letterbox + quantize), 11.1 ms in the DPU and ~4 ms acquiring, about 34 ms
in total against the camera's 33 ms frame period. The display side (3.0 ms wait, 1.4 ms draw,
7.1 ms show) runs on the other thread and has headroom. So the camera and the capture thread
are now the limit, and the largest remaining cost is CPU preprocessing, not the DPU. Going
beyond 30 FPS (the D455 offers 848x480@60) would first need faster preprocessing.

## Earlier work

**DenseBox** was the first DPU experiment: the stock Vitis AI face detector, used to prove
camera → DPU → overlay → browser worked at all. It is kept for reference and is not developed
further ([`inference/densebox`](inference/densebox/README.md)). The **camera scripts** in
[`tools/camera_bringup`](tools/camera_bringup/README.md) are the original bring-up diagnostics.

## Tests

```sh
python3 -m unittest discover -s tests
```

Host-side tests cover option parsing, geometry, frame composition, scheduling, launchers,
build scripts and the Pi wire contract against fixtures: 110 tests, 2 skipped on a PC without
the board's libraries ([`tests/README.md`](tests/README.md)). Camera, DPU and monitor
behaviour is verified manually on the board.
