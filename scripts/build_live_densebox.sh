#!/bin/sh

set -e

RS_PREFIX="/root/librealsense-rsusb"

g++ \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    inference/live_densebox_realsense.cpp \
    -I"${RS_PREFIX}/include" \
    -L"${RS_PREFIX}/lib64" \
    $(pkg-config --cflags --libs opencv4) \
    -lrealsense2 \
    -lvitis_ai_library-facedetect \
    -lglog \
    -pthread \
    -o inference/live_densebox_realsense

echo "Built inference/live_densebox_realsense"
