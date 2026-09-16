#!/bin/sh

set -e

CXX="${CXX:-g++}"

RS_PREFIX="/root/librealsense-rsusb"

$CXX \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    inference/densebox_realsense.cpp \
    -I"${RS_PREFIX}/include" \
    -L"${RS_PREFIX}/lib64" \
    $(pkg-config --cflags --libs opencv4) \
    -lrealsense2 \
    -lvitis_ai_library-facedetect \
    -lglog \
    -pthread \
    -o inference/densebox_realsense

echo "Built inference/densebox_realsense"
