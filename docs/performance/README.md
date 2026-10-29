# Performance measurements (1920x1080 fullscreen)

Raw telemetry from the 1080p optimisation study summarised in the
[main README](../../README.md#performance-optimisation-at-1920x1080).
One log per step in [`1080p/`](1080p).

## Method

- **Board:** Kria KV260 (4× Cortex-A53), PetaLinux 2022.2, Vitis AI 3.0, DPU
  `DPUCZDX8G_ISA1_B4096` @ 300 MHz, Samsung S24D300 monitor at 1920x1080 over DisplayPort.
- **Camera:** RealSense D455 on USB 3 (5000 Mbit/s), colour 848x480@30, depth 424x240@30.
- **Model:** `yolov26n_facehuman_kria.xmodel`, 640x640 INT8, default confidence 0.30.
- **Run:** 35 s per step, `--telemetry true`, one run per step. Scene
  content was not controlled or recorded. The first
  5 one-second windows are dropped, and the remaining ~24-34 windows are averaged with
  `scripts/summarize_telemetry.py`.
- **Source versions:** each step is built from the git commit that introduced it,
  with the build script of that commit, on the board. The X11 steps ran under
  `xserver-nodm`. The MJPEG and KMS steps ran with X stopped.
- **`e2e`:** host pipeline latency from the camera frame request to hand-over
  to the display or socket. It is not glass-to-glass.
- **MJPEG client:** `curl` on the board reading `/stream.mjpg` into `/dev/null`,
  because the stream port was not reachable from the development PC. A real browser
  would add decode work on another machine, not on the board.
- The camera delivers 30 FPS, so 30 is the ceiling for every step.
  The measured 29.2 – 29.6 is the same limit seen through telemetry windows.

## Steps and logs

| Step | Log | Commit | Command |
|------|-----|--------|---------|
| 0 | `step0_mjpeg_baseline.txt` | `83cb6ca` | `live_yolov26_realsense --telemetry true` + curl client |
| 1 | `step1_x11_canvas.txt` | `d5aa59f` | `live_yolov26_hdmi --fullscreen true` |
| 2 | `step2_x11_canvas_reuse.txt` | `4413082` | `live_yolov26_hdmi --fullscreen true` |
| 3 | `step3_x11_direct_native.txt` | `3d19d4f` | `live_yolov26_hdmi --fullscreen false` (native-size window) |
| 4 | `step4_x11_fit_upscale.txt` | `3d19d4f` | `live_yolov26_hdmi --fullscreen true` |
| 5 | `step5_kms_linear.txt` | `83cb6ca` | `live_yolov26_hdmi --renderer kms --kms-scaling linear` |
| 6 | `step6_kms_nearest.txt` | `83cb6ca` | `live_yolov26_hdmi --renderer kms` |

Reproduce the table:

```sh
python3 scripts/summarize_telemetry.py docs/performance/1080p/step*.txt
```

## Caveats

- Single run per step on one board, one scene. Differences of a few percent are noise.
  The effects reported are a factor of 3 or more.
- The earlier optimisation notes in the project history were measured at the monitor's
  1024x768 X11 mode. They do not transfer: the same "aspect-fit upscale" code reached
  ~30 FPS at 1024x768 and 9.7 FPS at 1920x1080. All numbers here are 1080p.
- Quality of the nearest-neighbour upscale was judged by eye only.
