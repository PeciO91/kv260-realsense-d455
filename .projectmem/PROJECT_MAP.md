# Project Map - KV260_Camera

Reviewed against repository source on 2026-10-29, after the inference/ restructure (densebox/ + yolov26/{core,mjpeg,rpi,hdmi}).

## Project purpose

Intel RealSense D455 RGB/depth capture and INT8 person/face detection on AMD/Xilinx Kria KV260. Python utilities provide camera diagnostics and browser previews; C++17 DenseBox/YOLOv26 applications run on the board's DPU and deliver annotated MJPEG or H.264/RTP video plus independent Redis face status to a Raspberry Pi.

## Stack

- Deployment: Kria KV260, Xilinx/PetaLinux 2022.2 (documented kernel 5.15.36-xilinx-v2022.2), RealSense D455, Python 3.9.
- Camera: librealsense 2.58.4 source-built with RSUSB and pyrealsense2; NumPy/OpenCV for Python, librealsense/OpenCV for C++.
- Inference: Vitis AI 3.0 VART/XIR RunnerExt for YOLO; Vitis AI facedetect for DenseBox.
- Build: standalone POSIX shell scripts, g++ C++17, pkg-config; no project-wide CMake build.
- Network: HTTP MJPEG for browsers; native GStreamer appsrc/H.264/RTP/UDP and hiredis-backed Redis status for Pi; standard-library Python mock.

## Entry points

- Python camera diagnostics (bring-up only): tools/camera_bringup/{camera_test,depth_test,aligned_depth_test,live_stream,live_rgb_depth}.py.
- DenseBox (frozen first experiment): inference/densebox/densebox_realsense.cpp and live_densebox_realsense.cpp (MJPEG, port 8081).
- YOLO (main): inference/yolov26/mjpeg/live_yolov26_realsense.cpp (MJPEG, port 8081), inference/yolov26/rpi/live_yolov26_rpi.cpp (Pi video/status sender), inference/yolov26/hdmi/live_yolov26_hdmi.cpp (local monitor; --renderer x11|kms, synthetic --test-pattern).
- Network-only integration: scripts/mock_yolov26_sender.py; launchers scripts/run_live_yolov26_rpi.py and scripts/run_live_yolov26_hdmi.py.
- Binaries are built to build/ (git-ignored) by scripts/build_*.sh.

## Structure

- `README.md` - CV-oriented overview: hardware, architecture, modes, 1080p optimisation study (steps 0-6 table).
- `requirements.txt` - NumPy/OpenCV only, warning to retain RSUSB pyrealsense2.
- `.gitignore` - excludes build/, model artifacts, generated captures, caches.
- `AGENTS.md` and `CLAUDE.md` - projectmem bridge; registered id kv260-camera.
- `tools/camera_bringup/` - Python camera diagnostics + README (not part of the pipeline).
- `inference/` - native board inference. `inference/README.md` is the index.
  - `inference/densebox/` - frozen first DPU experiment (densebox_realsense.cpp, live_densebox_realsense.cpp, README).
  - `inference/yolov26/` - main app; README has model contract, pipeline, shared options.
    - `core/` - yolov26_common.hpp, yolov26_realsense_core.{hpp,cpp}: profiles, letterbox/INT8, VART, decode, sparse depth, annotations, LatestFrameHandoff, monitors.
    - `mjpeg/` - live_yolov26_realsense.cpp + README.
    - `rpi/` - live_yolov26_rpi.cpp, yolov26_rpi_control.{hpp,cpp}, yolov26_rpi_transport.{hpp,cpp} + README.
    - `hdmi/` - live_yolov26_hdmi.cpp, yolov26_display.hpp (Display interface), yolov26_hdmi_control.{hpp,cpp} (CLI, fit geometry, DisplayTelemetry), yolov26_hdmi_display.{hpp,cpp} (X11/GTK backend), yolov26_kms_display.{hpp,cpp} (DRM/KMS backend) + README.
- `scripts/` - build scripts, launchers, mock sender, analysis. `scripts/README.md` catalogues every script and holds the mock-sender guide.
  - `build_densebox_realsense.sh`, `build_live_densebox.sh`, `build_live_yolov26.sh`, `build_live_yolov26_rpi.sh`, `build_live_yolov26_hdmi.sh` - all write to build/.
  - `run_live_yolov26_hdmi.py` (--check-display/--wake-display), `run_live_yolov26_rpi.py` (isolated dependency env), `mock_yolov26_sender.py`, `summarize_telemetry.py` (logs -> markdown table).
- `docs/` - `realsense-rsusb.md` (RSUSB backend, USB 3), `performance/` (method README + raw 1080p telemetry logs docs/performance/1080p/step0..6).
- `models/` - external-model placeholder directory (`.gitkeep` only).
- `tests/` - host-side tests mirroring inference/yolov26: `tests/yolov26/{hdmi,rpi,mjpeg}/` (Python drivers + native .cpp fixtures); `tests/README.md`. Run `python3 -m unittest discover -s tests` (110 tests, 2 skipped without board libs).
- `.projectmem/` - project memory: authored map/intent plus tool-managed audit and derived context.

## Relationships

- Each `scripts/build_*.sh` compiles its app from inference/<densebox|yolov26/<mode>> (plus `inference/yolov26/core` for YOLO, via `-I`) into build/.
- Every YOLO mode calls `inference/yolov26/core/yolov26_realsense_core.hpp/.cpp`; `yolov26_common.hpp` supplies shared types/constants. Includes are bare names resolved by per-build `-I` dirs.
- `inference/yolov26/hdmi/live_yolov26_hdmi.cpp` picks `HdmiDisplay` (X11) or `KmsDisplay` through the `Display` interface in yolov26_display.hpp; CLI/geometry/telemetry come from yolov26_hdmi_control.
- `inference/yolov26/rpi/live_yolov26_rpi.cpp` uses yolov26_rpi_control (state/CLI) and yolov26_rpi_transport (network delivery).
- `scripts/run_live_yolov26_rpi.py` imports dependency helpers from `scripts/mock_yolov26_sender.py` and execs build/live_yolov26_rpi; `scripts/run_live_yolov26_hdmi.py` wraps build/live_yolov26_hdmi.
- Tests under tests/yolov26/<mode>/ compile the native sources of inference/yolov26/{core,<mode>} and import scripts/; REPO root is four directory levels up.
- projectmem MCP appends `.projectmem/events.jsonl` and regenerates `.projectmem/summary.md`; purpose is pulled from this authored map.

## Setup and validation

- Run build scripts from repo root on equipped KV260; default RealSense prefix /root/librealsense-rsusb, lib64. Pi build accepts --rs-prefix and lib/lib64; DenseBox diagnostic accepts CXX env override.
- Keep RSUSB bindings at /usr/lib/python3.9/site-packages/pyrealsense2; requirements.txt does not install pyrealsense2.
- External default model yolov26n_facehuman_kria.xmodel resolves relative to process cwd, or use --model. Two-class split one2one INT8 contract only; not arbitrary YOLO exports. RGB letterbox INTER_LINEAR, padding 114, REG_MAX=1, no NMS.
- Native live color 848x480@30; smallest same-FPS Z16 depth (424x240 on documented camera). Sparse calibrated central-ROI 5x5 median depth; Python alignment diagnostics and the single-shot DenseBox diagnostic use full-frame alignment.
- Pi dependency prefix ~/kv260-network-deps or --deps-prefix; build uses pkg-config and rpaths, launcher isolates plugin/library environment. OMX requires /dev/allegro*; x264 fallback.
- Pi video defaults: UDP 5000, H.264 payload 96/90kHz, MTU 1200, 4000 kbit/s. Redis defaults: TCP 6379/channel face_detection, one-second interval, two-second freshness; missing status is unavailable, not false. No Redis AUTH/TLS; private/source-restricted endpoint only; receiver is maintained outside this repo.
- Mock requires Python >=3.9 stdlib plus GStreamer CLI/plugins, not NumPy/OpenCV/redis-py/model/hardware. Synthetic pixels and face state are independent.
- Offline test commands: python3 -m unittest discover -s tests -p 'test_mock_yolov26_sender.py'; python3 -m unittest discover -s tests -p 'test_live_yolov26_rpi.py'. Native control fixture needs g++; transport fixture skips without GStreamer/hiredis dev packages. No dedicated CI/lint configuration in tracked file index.
- Hardware/DPU/live streaming are not verified by this memory setup pass; do not conflate local tests or exit zero with Pi delivery or Redis health.

## Known gaps

- Compiled model artifacts (.xmodel) are external and not committed; the Pi receiver/Redis endpoint is maintained outside this repository.
