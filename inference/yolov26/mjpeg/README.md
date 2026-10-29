# YOLOv26 → browser (MJPEG)

Annotated video over HTTP. Open the page from any machine on the same network.
Boxes carry class, score and distance. This is the simplest mode and the
reference for the others: it reaches the camera's 30 FPS (29.6 measured) with ~36 % CPU and ~52 ms host latency.

Source: `live_yolov26_realsense.cpp` plus [`../core/`](../core). Overview of the
model, pipeline and shared options: [`../README.md`](../README.md).

## Build and run

```sh
sh scripts/build_live_yolov26.sh              # → build/live_yolov26_realsense
./build/live_yolov26_realsense \
    --model /home/root/models/yolov26n_facehuman_kria.xmodel
```

If the loader cannot find `librealsense2.so`, add the RSUSB library directory first:
`export LD_LIBRARY_PATH=/root/librealsense-rsusb/lib64:$LD_LIBRARY_PATH` (the HDMI and Pi
builds embed this path as an rpath; this one does not).

Then open `http://KV260_IP:8081/` (`hostname -I` on the board; the video itself is at `/stream.mjpg`). The page
removes margins and scales the stream to the viewport, keeping the aspect ratio.

## Options

Shared options (`--model`, `--confidence`, `--max-detections`,
`--depth-width/height`, `--show-depth-roi`, `--show-confidence`, `--telemetry`):
[`../README.md`](../README.md#shared-options). Mode-specific:

| Option | Default | Meaning |
|--------|---------|---------|
| `--port N` | 8081 | HTTP port |
| `--jpeg-quality N` | 90 | JPEG quality of the stream |
| `--list-profiles` | | print the native D455 colour/depth profiles and exit |

## Threading

The main thread owns the camera, DPU inference and depth mapping. A renderer
thread owns annotation drawing, JPEG encoding and sending. A capacity-1 latest
frame handoff sits between them, so a slow client replaces pending frames
instead of building latency.

## Telemetry (`--telemetry true`)

Terminal lines report pipeline FPS and stream FPS separately, per-stage means
(`acquire`, `preprocess`, `dpu`, `depth_postprocess`, `overlay`, `jpeg`, `send`),
cumulative dropped frames and the per-window drop rate, and end-to-end
latency from capture to completed send. Telemetry is off by default because
polling the system monitors costs CPU.

## Tests

`tests/yolov26/mjpeg/test_yolov26_overlay_options.py` extracts the real option
parser and the real overlay drawing code from the sources and runs them against
a stub, so compile errors fail the test instead of skipping it.
