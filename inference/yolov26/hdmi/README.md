# YOLOv26 → local monitor (HDMI / DisplayPort)

`live_yolov26_hdmi` shows the annotated stream directly on the monitor
attached to the KV260, with no browser and no network. One binary, two
display backends selected with `--renderer`:

| Backend | Flag | Path | Needs |
|---------|------|------|-------|
| **X11** | `--renderer x11` (default) | OpenCV HighGUI (GTK3) window on the running X desktop | X session (`xserver-nodm`) running |
| **KMS** | `--renderer kms` | Direct DRM/KMS scan-out with dumb buffers, no X, no GTK | X **stopped** (the app needs DRM master) |

The measured difference at 1920x1080 fullscreen is large: see the
[performance section of the main README](../../../README.md#performance-optimisation-at-1920x1080).
Use KMS for the lowest latency and CPU; use X11 when you need a normal desktop.

Overview of the pipeline and shared options: [`../README.md`](../README.md).

## Files

| File | Role |
|------|------|
| `live_yolov26_hdmi.cpp` | entry point: synthetic producer, real capture worker, GUI loop, telemetry |
| `yolov26_display.hpp` | `Display` interface (`show`, `poll`, `target_width/height`) both backends implement |
| `yolov26_hdmi_display.{hpp,cpp}` | X11 backend: `XOpenDisplay` + `cv::imshow`, GTK black background, aspect-fit upscale |
| `yolov26_kms_display.{hpp,cpp}` | KMS backend: plane discovery, mode-sized dumb buffers, software scaling, console restore |
| `yolov26_hdmi_control.{hpp,cpp}` | dependency-free CLI validation, fit geometry, `DisplayTelemetry` accumulator |

## Build

```sh
sh scripts/build_live_yolov26_hdmi.sh [--rs-prefix DIR]   # → build/live_yolov26_hdmi
```

Needs pkg-config modules `opencv4`, `x11`, `gtk+-3.0` and `libdrm`.

## Common options

Shared inference options: [`../README.md`](../README.md#shared-options).

| Option | Default | Meaning |
|--------|---------|---------|
| `--renderer x11\|kms` | x11 | display backend |
| `--fullscreen true\|false` | true | fullscreen, or native-size window / centred image |
| `--telemetry true\|false` | false | on-screen overlay and a `[telemetry]` line every second |
| `--duration-seconds S` | until stopped (10 s with `--test-pattern`) | stop after S seconds |
| `--test-pattern` | off | synthetic colour bars, no camera and no model |
| `--display NAME` | `:0` | X display (x11 only) |

`--test-pattern` renders labelled bars (`DISPLAY TEST - NO CAMERA / NO INFERENCE`)
through the same render path. It is the way to check the monitor and renderer without
touching the camera or DPU. `--help` is fully offline. Invalid arguments and an
unreachable `--display` are rejected before any camera or model initialisation.

**Exit:** Esc, `q`, window close or Ctrl+C (X11). On the KMS console Esc gives the
cleanest exit.

## X11 backend

```sh
sh scripts/build_live_yolov26_hdmi.sh
./build/live_yolov26_hdmi --model /home/root/models/yolov26n_facehuman_kria.xmodel
```

The window is created through OpenCV HighGUI. OpenCV 4.5.2's GTK backend draws the
image at its natural size and does not scale it to the window, so fullscreen mode
upscales the 848x480 frame aspect-fit to the screen with one `cv::resize` into a reused
buffer and sets a black GTK background (`GtkCssProvider`). Windowed mode shows the
frame at native size.

A Python launcher adds display diagnostics that never touch the camera or model:

```sh
python3 scripts/run_live_yolov26_hdmi.py --check-display   # query XRandR + DPMS, exit
python3 scripts/run_live_yolov26_hdmi.py --wake-display    # DPMS on + reset screensaver, once
python3 scripts/run_live_yolov26_hdmi.py -- --test-pattern # forwarded args go to the binary
```

The launcher defaults to `DISPLAY=:0` (`--display` overrides), keeps your
`XAUTHORITY`, never broadens X access, and does not change power-saving timeouts.

## KMS backend

```sh
systemctl stop xserver-nodm                       # release DRM master
./build/live_yolov26_hdmi --renderer kms \
    --model /home/root/models/yolov26n_facehuman_kria.xmodel
systemctl start xserver-nodm                      # get the desktop back afterwards
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--kms-plane primary\|overlay` | primary | which DPSUB layer carries the image |
| `--kms-scaling nearest\|linear` | nearest | fullscreen resize: nearest-neighbour or bilinear |

How it works on the ZynqMP DisplayPort subsystem (DPSUB):

- The DPSUB has two layers, **video** (overlay plane) and **graphics** (primary plane),
  and **no scaler**. Both enabled layers must have the same source size, or the driver
  logs `Layer width:height must be 1920:1080` and skips enabling the video layer.
  This is why "hardware scaling" does not work and why the first KMS attempt showed a
  black screen.
- So the app allocates **mode-sized** scan-out buffers, converts the 848x480 frame at
  native size, scales it in software to the panel (nearest-neighbour at 16/32-bit word
  level by default), and presents it with `src == dst == full mode`.
- Planes are chosen by type through `DRM_CLIENT_CAP_UNIVERSAL_PLANES`, calls are
  checked, and the graphics plane's alpha properties are saved and restored on exit.
- Windowed mode (`--fullscreen false`) centres the native-size frame.
- Keyboard input is read from the console in raw mode, since there is no X to deliver events.
- `modetest -M xlnx` modesets are unreliable on this driver; test with `--test-pattern`
  instead.

Telemetry fields: `fps`, `e2e_mean/max`, `wait` (blocked on the handoff), `draw`
(annotations + overlay), `show` (render + present), plus `acquire / preprocess / dpu /
depth` in real mode, and `cpu / ram / temp`. Synthetic mode omits the inference stages.

## Tests

```sh
python3 -m unittest discover -s tests/yolov26/hdmi
```

Covers CLI parsing, fit geometry, the KMS buffer composition and scaling
(`test_yolov26_kms_display.cpp`), launcher actions, build-script checks and the
synthetic path. Real camera, DPU and monitor behaviour is verified manually on the board.
