# KV260 RealSense D455

Intel RealSense D455 camera pipeline for the AMD/Xilinx Kria KV260.

## Current status

Working:

- Intel RealSense D455 detection
- RGB capture at 640x480
- librealsense / pyrealsense2
- RSUSB backend on Xilinx 2022.2
- Stable pipeline start and stop
- JPEG frame capture
- Live MJPEG stream over HTTP

## Camera pipeline

RealSense D455
    |
    | USB
    v
librealsense RSUSB
    |
    v
pyrealsense2
    |
    v
RGB frame
    |
    +----> OpenCV / JPEG
    |
    +----> HTTP MJPEG stream
    |
    v
Future YOLO26 preprocessing
    |
    v
KV260 DPU
    |
    v
Person / Human Face detections

## Camera test

Run:

    python3 camera/camera_test.py

A test frame is saved as:

    d455_test.jpg

## Live preview

Run:

    python3 camera/live_stream.py

Find the KV260 IP address:

    hostname -I

Then open on another computer:

    http://KV260_IP:8080/

The default network preview runs at 15 FPS with JPEG quality 75.

Alternative:

    python3 camera/live_stream.py \
        --port 8080 \
        --stream-fps 20 \
        --jpeg-quality 80

## RealSense backend

The standard V4L2/UVC backend caused a kernel crash during
pipeline shutdown on the Xilinx 2022.2 kernel.

The project therefore uses the official librealsense RSUSB backend.

See:

    docs/realsense-rsusb.md

## Future work

- Depth stream
- RGB/depth alignment
- DPU-compatible YOLO26n
- INT8 Vitis AI inference
- Person and Human Face detection
- Detection overlays
- Distance estimation from depth data

## RGB + depth test

The D455 can capture RGB and depth simultaneously.

Run:

    python3 camera/depth_test.py

The test starts:

- RGB: 640x480 @ 30 FPS
- Depth: 640x480 @ 30 FPS

It captures 30 frames and prints the depth value at the center of the
depth image in metres.

Example:

    Frame 12: RGB 640x480, Depth 640x480, center distance = 1.842 m

This test does not yet align the depth image to the RGB camera.
RGB-to-depth alignment will be added separately before YOLO integration.
