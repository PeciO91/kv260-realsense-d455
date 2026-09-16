# KV260 RealSense D455

Intel RealSense D455 camera pipeline for the AMD/Xilinx Kria KV260.

## Phase 1 - Camera bring-up

- Detect Intel RealSense D455
- Capture RGB frames
- Capture depth frames
- Stream camera output to a PC

## Phase 2 - YOLO26 FPGA inference

- DPU-compatible YOLO26n
- Person and Human Face detection
- INT8 quantization with Vitis AI
- DPU inference on KV260
- Real-time RealSense D455 input
- Detection visualization

## Pipeline

RealSense D455
    |
    | USB
    v
KV260 ARM/Linux
    |
    | pyrealsense2
    v
RGB / Depth frame
    |
    | preprocessing
    v
KV260 DPU
    |
    | YOLO26n INT8
    v
Person / Human Face detections

## Current status

Camera bring-up.
