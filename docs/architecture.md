# Architecture

## Camera pipeline

Intel RealSense D455
-> USB
-> KV260 Linux
-> pyrealsense2
-> RGB / depth frames

## Detection pipeline

Intel RealSense D455
-> RGB frame
-> preprocessing
-> YOLO26n INT8
-> KV260 DPU
-> postprocessing
-> Person / Human Face detections
