# Inference

Native C++17 RealSense + DPU applications for the Kria KV260. The live
applications annotate detections using sparse calibrated median depth over
each box's central ROI without full-frame alignment. The single-shot
DenseBox diagnostic instead aligns the full depth frame to color and takes
the median of all valid depth pixels in the central ROI.

Run the build scripts from the repository root.

| Application | Source | Build script |
|-------------|--------|--------------|
| Single-shot DenseBox face/depth diagnostic | `densebox_realsense.cpp` | `scripts/build_densebox_realsense.sh` |
| Live DenseBox MJPEG stream | `live_densebox_realsense.cpp` | `scripts/build_live_densebox.sh` |
| Live YOLOv26 MJPEG stream | `live_yolov26_realsense.cpp` | `scripts/build_live_yolov26.sh` |
| Pi YOLOv26 RTP/Redis sender | `live_yolov26_rpi.cpp` | `scripts/build_live_yolov26_rpi.sh` |
| YOLOv26 HDMI/X11 local display | `live_yolov26_hdmi.cpp` | `scripts/build_live_yolov26_hdmi.sh` |

Requirements: C++17 toolchain on the board, the RSUSB librealsense install
(default `/root/librealsense-rsusb`), and OpenCV. DenseBox requires the
Vitis AI `facedetect` library plus the installed `densebox_320_320` model;
YOLOv26 uses the Vitis AI 3.0 VART/XIR `RunnerExt`. The
Pi sender additionally needs GStreamer and hiredis under a dependency
prefix (auto-discovered or `--deps-prefix`). The HDMI display
additionally needs the X11 development files (`pkg-config x11`) and a
running local X session.

Model artifacts (`.xmodel`) are external and not committed. The default
YOLOv26 model path `yolov26n_facehuman_kria.xmodel` is resolved relative to
the process working directory, or passed explicitly via `--model`. Only the
documented two-class split one2one INT8 output contract is supported — not
arbitrary YOLO exports.

The Pi sender publishes face-detection status on Redis independently of the
H.264/RTP video stream; the receiver is maintained outside this repository.

See the root [README](../README.md) for detailed build, run, option,
wire-contract, and telemetry documentation for each application.
