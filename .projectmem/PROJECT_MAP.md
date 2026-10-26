# Project Map - KV260_Camera

Reviewed against repository source on 2026-10-19. Promoted from PROJECTMEM_MAP_DRAFT.md; describes the existing architecture.

## Project purpose

Intel RealSense D455 RGB/depth capture and INT8 person/face detection on AMD/Xilinx Kria KV260. Python utilities provide camera diagnostics and browser previews; C++17 DenseBox/YOLOv26 applications run on the board's DPU and deliver annotated MJPEG or H.264/RTP video plus independent Redis face status to a Raspberry Pi.

## Stack

- Deployment: Kria KV260, Xilinx/PetaLinux 2022.2 (documented kernel 5.15.36-xilinx-v2022.2), RealSense D455, Python 3.9.
- Camera: librealsense 2.58.4 source-built with RSUSB and pyrealsense2; NumPy/OpenCV for Python, librealsense/OpenCV for C++.
- Inference: Vitis AI 3.0 VART/XIR RunnerExt for YOLO; Vitis AI facedetect for DenseBox.
- Build: standalone POSIX shell scripts, g++ C++17, pkg-config; no project-wide CMake build.
- Network: HTTP MJPEG for browsers; native GStreamer appsrc/H.264/RTP/UDP and hiredis-backed Redis status for Pi; standard-library Python mock.

## Entry points

- Python camera diagnostics: camera/camera_test.py, camera/depth_test.py, camera/aligned_depth_test.py.
- Python browser previews: camera/live_stream.py (port 8080) and camera/live_rgb_depth.py (port 8080).
- DenseBox: inference/densebox_realsense.cpp (local image diagnostic) and inference/live_densebox_realsense.cpp (MJPEG, port 8081).
- YOLO: inference/live_yolov26_realsense.cpp (MJPEG, port 8081), inference/live_yolov26_rpi.cpp (Pi video/status sender), and inference/live_yolov26_hdmi.cpp (fullscreen X11 display, synthetic --test-pattern mode).
- Network-only integration: scripts/mock_yolov26_sender.py; native sender launcher scripts/run_live_yolov26_rpi.py.

## Structure

- `PROJECTMEM_MAP_DRAFT.md` - original reviewed bootstrap draft, retained for provenance; live navigation is this map.
- `README.md` - primary board build/run, options, wire contracts, telemetry guide.
- `requirements.txt` - NumPy/OpenCV only, warning to retain RSUSB pyrealsense2.
- `.gitignore` - excludes compiled binaries, model artifacts, generated captures, caches.
- `AGENTS.md` and `CLAUDE.md` - projectmem bridge; registered id kv260-camera.
- `camera/` - board-attached Python camera utilities.
  - `camera/camera_test.py` - capture 640x480 RGB, save d455_test.jpg.
  - `camera/depth_test.py` - raw simultaneous RGB/depth center-distance diagnostic.
  - `camera/aligned_depth_test.py` - full-frame depth-to-color alignment diagnostic.
  - `camera/live_stream.py` - threaded RGB HTTP MJPEG preview, FPS/JPEG options.
  - `camera/live_rgb_depth.py` - RGB browser stream with aligned central depth measurement.
- `inference/` - native board inference and reusable Pi networking components.
  - `inference/densebox_realsense.cpp` - single-shot DenseBox face/depth diagnostic with full-frame depth-to-color alignment.
  - `inference/live_densebox_realsense.cpp` - live DenseBox MJPEG, sparse depth, system/stage telemetry.
  - `inference/live_yolov26_realsense.cpp` - YOLO browser MJPEG entry point, renderer handoff, telemetry.
  - `inference/live_yolov26_rpi.cpp` - native YOLO Pi sender lifecycle, capture, render, status worker.
  - `inference/yolov26_common.hpp` - hardware-independent constants and inference/detection data types.
  - `inference/yolov26_realsense_core.hpp` - camera, runner, frame handoff, calibration and monitor interfaces.
  - `inference/yolov26_realsense_core.cpp` - profiles, letterbox/INT8, VART execution, decode/depth, annotations, hwmon.
  - `inference/yolov26_rpi_control.hpp` and `inference/yolov26_rpi_control.cpp` - CLI validation, face-state freshness, report/backoff scheduling, telemetry.
  - `inference/live_yolov26_hdmi.cpp` - YOLO local X11 display entry point, synthetic producer, real capture worker.
  - `inference/yolov26_hdmi_control.hpp` and `inference/yolov26_hdmi_control.cpp` - dependency-free HDMI CLI validation and aspect-fit geometry.
  - `inference/yolov26_hdmi_display.hpp` and `inference/yolov26_hdmi_display.cpp` - X11/OpenCV window adapter, black-canvas aspect-fit rendering, key/close exit.
  - `inference/yolov26_rpi_transport.hpp` and `inference/yolov26_rpi_transport.cpp` - host resolution, deadline-bounded Redis publishing and GStreamer encoder/video transport.
  - `inference/README.md` - native inference app overview, per-app build scripts, runtime/model requirements.
- `scripts/` - standalone builds and network tools.
  - `scripts/build_densebox_realsense.sh` - build DenseBox diagnostic.
  - `scripts/build_live_densebox.sh` - build live DenseBox, including dpu_task link.
  - `scripts/build_live_yolov26.sh` - build YOLO MJPEG with shared core.
  - `scripts/build_live_yolov26_rpi.sh` - build native sender with dependency/RealSense prefix checks.
  - `scripts/build_live_yolov26_hdmi.sh` - build HDMI display app with RealSense prefix and OpenCV/X11 checks.
  - `scripts/run_live_yolov26_hdmi.py` - HDMI launcher with offline --check-display/--wake-display X actions.
  - `scripts/run_live_yolov26_rpi.py` - prefix-aware native sender exec; imports mock dependency helpers.
  - `scripts/mock_yolov26_sender.py` - synthetic GStreamer video plus independent simulated face Redis publisher, no DPU/camera.
  - `scripts/README.md` - mock sender setup, wire contract and receiver coordination.
- `docs/` - deployment/reference docs.
  - `docs/architecture.md` - high-level camera and DPU flow.
  - `docs/realsense-rsusb.md` - shutdown kernel Oops, RSUSB installation paths, documented USB 2.0 constraint.
- `models/` - external-model placeholder directory.
  - `models/.gitkeep` - placeholder only; deployed xmodels are external and not included.
- `tests/` - host-side Python and native component fixtures.
  - `tests/test_mock_yolov26_sender.py` - stdlib unittest mock CLI, Redis freshness/reconnect, pipeline and lifecycle fixtures.
  - `tests/test_live_yolov26_rpi.py` - unittest native-fixture compiler/runner and launcher tests.
  - `tests/test_yolov26_rpi_control.cpp` - hardware-independent native CLI/state/scheduler assertions.
  - `tests/test_yolov26_rpi_transport.cpp` - native Redis/encoder/transport fixtures, requires development libraries.
  - `tests/test_live_yolov26_hdmi.py` - unittest HDMI launcher/build/control-fixture tests.
  - `tests/test_yolov26_hdmi_control.cpp` - hardware-independent HDMI CLI/geometry assertions.
  - `tests/test_yolov26_overlay_options.py` - extraction fixtures for browser CLI parsing and annotation drawing.
- `.projectmem/` - project memory: authored map/intent plus tool-managed audit and derived context.
  - `.projectmem/AI_INSTRUCTIONS.md` - setup/maintenance and event logging rules.
  - `.projectmem/PROJECT_MAP.md` - authored project purpose, navigable path index, relationships, and setup constraints.
  - `.projectmem/summary.md` - auto-generated bounded context from events and map purpose; never manually edit.
  - `.projectmem/events.jsonl` - append-only event log, use MCP.
  - `.projectmem/plan.md` - future intent, currently empty; no new roadmap inferred.
  - `.projectmem/issues/` - individual issue histories.

## Relationships

- `camera/camera_test.py`, `camera/depth_test.py`, `camera/aligned_depth_test.py`, `camera/live_stream.py`, and `camera/live_rgb_depth.py` use `requirements.txt` dependencies and the RSUSB setup documented in `docs/realsense-rsusb.md`.
- `scripts/build_densebox_realsense.sh` compiles `inference/densebox_realsense.cpp`; `scripts/build_live_densebox.sh` compiles `inference/live_densebox_realsense.cpp`.
- `scripts/build_live_yolov26.sh` compiles `inference/live_yolov26_realsense.cpp` together with `inference/yolov26_realsense_core.cpp`.
- `scripts/build_live_yolov26_rpi.sh` compiles `inference/live_yolov26_rpi.cpp` plus `inference/yolov26_realsense_core.cpp`, `inference/yolov26_rpi_control.cpp`, and `inference/yolov26_rpi_transport.cpp`.
- `inference/live_yolov26_realsense.cpp` and `inference/live_yolov26_rpi.cpp` call `inference/yolov26_realsense_core.hpp`/`inference/yolov26_realsense_core.cpp`; `inference/yolov26_common.hpp` supplies shared types/constants.
- `inference/live_yolov26_rpi.cpp` uses `inference/yolov26_rpi_control.hpp`/`inference/yolov26_rpi_control.cpp` for state/CLI and `inference/yolov26_rpi_transport.hpp`/`inference/yolov26_rpi_transport.cpp` for network delivery.
- `scripts/run_live_yolov26_rpi.py` imports dependency environment helpers from `scripts/mock_yolov26_sender.py` and execs the built `inference/live_yolov26_rpi` binary.
- `tests/test_live_yolov26_rpi.py` imports `scripts/run_live_yolov26_rpi.py` and compiles/runs `tests/test_yolov26_rpi_control.cpp` and `tests/test_yolov26_rpi_transport.cpp` against native sources.
- `tests/test_mock_yolov26_sender.py` imports `scripts/mock_yolov26_sender.py`.
- `scripts/build_live_yolov26_hdmi.sh` compiles `inference/live_yolov26_hdmi.cpp` plus `inference/yolov26_realsense_core.cpp`, `inference/yolov26_hdmi_control.cpp`, and `inference/yolov26_hdmi_display.cpp`.
- `inference/live_yolov26_hdmi.cpp` calls `inference/yolov26_hdmi_control.hpp`/`inference/yolov26_hdmi_control.cpp` for CLI/geometry and `inference/yolov26_hdmi_display.hpp`/`inference/yolov26_hdmi_display.cpp` for X11/HighGUI windowing; it shares `inference/yolov26_realsense_core` for inference and the now nonblocking-capable LatestFrameHandoff.
- `tests/test_live_yolov26_hdmi.py` imports `scripts/run_live_yolov26_hdmi.py` and compiles/runs `tests/test_yolov26_hdmi_control.cpp` against `inference/yolov26_hdmi_control.cpp`.
- projectmem MCP appends `.projectmem/events.jsonl` and regenerates `.projectmem/summary.md`; purpose is pulled from this authored `.projectmem/PROJECT_MAP.md`.

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
