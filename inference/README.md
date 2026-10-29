# Inference

Native C++17 RealSense + DPU applications for the Kria KV260.

| Folder | What it is | Status |
|--------|-----------|--------|
| [`yolov26/`](yolov26/README.md) | **Main work.** INT8 YOLOv26n person / face detector on the DPU with sparse calibrated depth, and four output modes (browser MJPEG, Raspberry Pi RTP + Redis, HDMI via X11, HDMI via DRM/KMS). | Active |
| [`densebox/`](densebox/README.md) | First DPU bring-up with the stock Vitis AI DenseBox face detector. Kept as a reference. | Frozen |

```
inference/
├── densebox/            first DPU experiment (stock Vitis AI facedetect library)
└── yolov26/
    ├── core/            shared by every YOLO mode: camera, letterbox, INT8, DPU runner, decode, depth, overlay
    ├── mjpeg/           annotated MJPEG over HTTP, for a browser
    ├── rpi/             annotated H.264/RTP to a Raspberry Pi + independent Redis face status
    └── hdmi/            local monitor output: --renderer x11 (GTK) or --renderer kms (DRM planes)
```

## Building

Run the build scripts from the repository root. Binaries are written to
`build/` (git-ignored). See [`scripts/README.md`](../scripts/README.md) for
every script.

| Application | Source | Build script | Binary |
|-------------|--------|--------------|--------|
| Single-shot DenseBox face/depth diagnostic | `densebox/densebox_realsense.cpp` | `scripts/build_densebox_realsense.sh` | `build/densebox_realsense` |
| Live DenseBox MJPEG stream | `densebox/live_densebox_realsense.cpp` | `scripts/build_live_densebox.sh` | `build/live_densebox_realsense` |
| YOLOv26 MJPEG stream | `yolov26/mjpeg/` | `scripts/build_live_yolov26.sh` | `build/live_yolov26_realsense` |
| YOLOv26 Pi RTP/Redis sender | `yolov26/rpi/` | `scripts/build_live_yolov26_rpi.sh` | `build/live_yolov26_rpi` |
| YOLOv26 HDMI (X11 + KMS) | `yolov26/hdmi/` | `scripts/build_live_yolov26_hdmi.sh` | `build/live_yolov26_hdmi` |

## Requirements (on the KV260)

- C++17 toolchain, OpenCV 4 (`pkg-config opencv4`).
- RSUSB librealsense install (default `/root/librealsense-rsusb`, see
  [`docs/realsense-rsusb.md`](../docs/realsense-rsusb.md)).
- DenseBox: Vitis AI `facedetect` library and the installed `densebox_320_320` model.
- YOLOv26: Vitis AI 3.0 VART/XIR runtime plus an external compiled `.xmodel`.
- Pi sender: GStreamer and hiredis under a dependency prefix.
- HDMI: X11, GTK3 and libdrm development files.

Model artifacts (`.xmodel`) are not committed; `models/` is a placeholder.
