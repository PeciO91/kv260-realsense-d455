#!/bin/sh

set -e

RS_PREFIX="/root/librealsense-rsusb"

mkdir -p build

g++ \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    inference/yolov26/mjpeg/live_yolov26_realsense.cpp \
    inference/yolov26/core/yolov26_realsense_core.cpp \
    -I inference/yolov26/core \
    -I"${RS_PREFIX}/include" \
    -L"${RS_PREFIX}/lib64" \
    $(pkg-config --cflags --libs opencv4) \
    -lrealsense2 \
    -lvart-runner \
    -lxir \
    -lunilog \
    -lglog \
    -pthread \
    -o build/live_yolov26_realsense

echo "Built build/live_yolov26_realsense"
