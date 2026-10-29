# DenseBox (first DPU experiment)

DenseBox was the first model run on the KV260 DPU, using the stock Vitis AI
`facedetect` library with the pre-built `densebox_320_320` model. Its purpose
was to prove the whole chain works at all: RealSense D455 → OpenCV → DPU →
overlay → browser. The project's real detector is [YOLOv26](../yolov26/README.md);
DenseBox is no longer developed.

| Application | Source | Build | Binary |
|-------------|--------|-------|--------|
| Single-shot face + depth diagnostic | `densebox_realsense.cpp` | `scripts/build_densebox_realsense.sh` | `build/densebox_realsense` |
| Live MJPEG stream with depth | `live_densebox_realsense.cpp` | `scripts/build_live_densebox.sh` | `build/live_densebox_realsense` |

The single-shot diagnostic aligns the full depth frame to color and takes the
median of valid depth pixels in the central ROI of each face. The live app
uses the sparse calibrated 5x5 sample grid instead (see below). This is the
origin of the sparse-depth design later reused by every YOLO mode.

## Live stream

```sh
sh scripts/build_live_densebox.sh
./build/live_densebox_realsense                    # http://KV260_IP:8081/
./build/live_densebox_realsense --list-profiles    # print native D455 profiles
./build/live_densebox_realsense --jpeg-quality 75  # default 90
./build/live_densebox_realsense --depth-width 640 --depth-height 480
```

At startup the app verifies the 848x480 YUYV color profile at 30 FPS and picks
the smallest Z16 depth profile at that FPS (424x240 on the tested camera). A
calibrated 5x5 grid maps points from each face's central color ROI into the raw
depth image with the RealSense intrinsics and extrinsics, avoiding full-frame
alignment.

## Telemetry panel

The MJPEG video carries a panel refreshed about once per second:

- **Stream FPS**: annotated JPEG frames sent to the client per second.
- **Pipeline FPS**: completed capture → preprocess → DPU → depth → draw loops per second.
- **DPU latency**: mean time in `detector->run()` only.
- **DPU rate**: `1000 / DPU latency`, the accelerator's throughput equivalent, not the application frame rate.
- CPU use (`/proc/stat` deltas), RAM used, PS/PL temperature (Xilinx AMS hwmon), board power (INA260 hwmon).

Unavailable metrics show `N/A` and never stop streaming. Per-stage averages
(acquire, resize, depth, overlay, JPEG, send, total) are printed to the console.
