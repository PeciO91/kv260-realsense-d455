#!/bin/sh

set -e

RS_PREFIX="/root/librealsense-rsusb"

mkdir -p build

g++ \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    inference/densebox/live_densebox_realsense.cpp \
    -I"${RS_PREFIX}/include" \
    -L"${RS_PREFIX}/lib64" \
    $(pkg-config --cflags --libs opencv4) \
    -lrealsense2 \
    -lvitis_ai_library-facedetect \
    -lvitis_ai_library-dpu_task \
    -lglog \
    -pthread \
    -o build/live_densebox_realsense

echo "Built build/live_densebox_realsense"
