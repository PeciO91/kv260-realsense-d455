# Camera bring-up tools

Small Python scripts (Python 3.9, `pyrealsense2` built with the RSUSB backend,
NumPy, OpenCV) written to answer one question first: *can the D455 be used on
this board at all?* They are not part of the detection pipeline. Keep them as
diagnostics for a new board, a new cable or a new librealsense build.

Run from the repository root. Setup of the RSUSB backend (needed, because the
default V4L2/UVC path crashed the kernel on stop): [`docs/realsense-rsusb.md`](../../docs/realsense-rsusb.md).

| Script | What it checks | Output |
|--------|----------------|--------|
| `camera_test.py` | Start the pipeline, grab 640x480 RGB, stop cleanly (the shutdown that used to crash) | `d455_test.jpg` |
| `depth_test.py` | RGB + depth at 640x480@30 together; prints the centre depth in metres for 30 frames. Raw, **unaligned** depth | console |
| `aligned_depth_test.py` | Same, but aligns the full depth frame to colour first; shows what full-frame alignment costs | console |
| `live_stream.py` | Threaded RGB MJPEG preview (`--port 8080 --stream-fps --jpeg-quality`) | `http://KV260_IP:8080/` |
| `live_rgb_depth.py` | RGB MJPEG with aligned central depth overlay | `http://KV260_IP:8080/` |

## What they were used for

- `camera_test.py` proved the RSUSB backend fixes the `uvc_stop_streaming()` kernel Oops.
- `depth_test.py` / `aligned_depth_test.py` exercised full-frame alignment. The C++ apps avoid that step and map only each detection's 5x5 sample grid through the calibration instead
  (see [`inference/yolov26/README.md`](../../inference/yolov26/README.md#depth)).
- The two live scripts were the first browser previews. The C++ MJPEG app
  ([`inference/yolov26/mjpeg`](../../inference/yolov26/mjpeg/README.md)) supersedes them.

Dependencies for the scripts: `pip install -r requirements.txt` (do **not** install the
PyPI `pyrealsense2`; it uses the crashing backend).
