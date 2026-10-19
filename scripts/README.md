# Mock YOLOv26 sender usage guide

Guide for `mock_yolov26_sender.py` in this directory. See also the
[Mock YOLOv26 network sender](../README.md#mock-yolov26-network-sender)
section in the project README.

## What it does

The mock sender transmits synthetic `videotestsrc` imagery as
H.264/RTP/UDP and, independently, publishes a simulated face-detection
boolean to a Redis Pub/Sub channel. There is no camera, model, DPU, or
inference of any kind, and the synthetic pixels have no relationship to
the published boolean. It lets the receiver integration be tested
independently of camera/DPU inference, using the same transport
contract as the real sender.

## Requirements

On the sender:

- Linux, Python >= 3.9. All imports are from the standard library; no
  requirements.txt or pip installs are needed for this mock.
- `gst-launch-1.0` and `gst-inspect-1.0` on `PATH`, plus the GStreamer
  factories `videotestsrc`, `capsfilter`, `queue`, `videoconvert`,
  `h264parse`, `rtph264pay`, `fakesink`, `udpsink`, and a working
  encoder such as `x264enc`.
- On the Kria board, a private GStreamer prefix under
  `~/kv260-network-deps` is auto-detected; no manual environment
  exports are needed.

On the receiver side: a video receiver with an H.264 decoder and a
reachable Redis server. The script does not create or start Redis and
does not supply a receiver. `redis-cli` is an optional diagnostic tool,
not a sender dependency.

## Run

From `scripts/` (or any folder containing the script):

```sh
python3 mock_yolov26_sender.py \
    --pi-host PI_IP \
    --encoder x264 \
    --pattern ball \
    --face-mode alternate \
    --video-port 5002 \
    --redis-channel face_detection_mock_test \
    --redis-interval-seconds 1 \
    --duration-seconds 30
```

Replace `PI_IP` with the actual Pi IP address. The sender stops 30
seconds after startup completes; Ctrl+C stops it earlier.
`--pattern ball` with `--face-mode alternate` produces changing video
and a changing status; after missed deadlines the boolean may repeat a
value rather than strictly alternating each delivered message.

## Video on the Pi

The example uses UDP port 5002; sender and receiver ports must match.
Nominal video is 848x480 at 30 FPS — a nominal rate, not a guaranteed
measured or displayed throughput. The RTP caps are exactly:

```
application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96
```

No receiver implementation or pipeline is supplied here.

## Redis

By default the sender publishes to the Pi host on TCP port 6379; use
`--redis-host REDIS_IP` and `--redis-port PORT` to point at a different
endpoint. The example channel is `face_detection_mock_test`; no channel
creation is needed — the subscriber only needs to subscribe to the same
name. If Redis runs on the Pi, an optional check on the Pi is:

```sh
redis-cli -h 127.0.0.1 -p 6379 SUBSCRIBE face_detection_mock_test
```

Payloads are exactly `{"face_detected":true}` or
`{"face_detected":false}`, published roughly every second
(`--redis-interval-seconds` adjusts it). Publishing is freshness-gated
and best-effort: missing messages mean unavailable, not false, nothing
is replayed, and the status stream is independent of the video. The
Redis endpoint must stay private/source-restricted; plain TCP only, no
AUTH or TLS support.

## Defaults vs. this example

The defaults are video port 5000 and channel `face_detection`; the
example deliberately uses port 5002 and a test channel so it does not
collide with live automation — coordinate test endpoints with the
receiver developer. Use `--pattern snow --face-mode random` for random
inputs, or `--face-mode true` / `--face-mode false` to force a state.

A clean exit 0 means local completion only; validate the displayed
video and the subscriber messages independently. See `--help` for all
options.
