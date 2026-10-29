#!/bin/sh

set -e

CXX="${CXX:-g++}"

RS_PREFIX="/root/librealsense-rsusb"

mkdir -p build

$CXX \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    inference/densebox/densebox_realsense.cpp \
    -I"${RS_PREFIX}/include" \
    -L"${RS_PREFIX}/lib64" \
    $(pkg-config --cflags --libs opencv4) \
    -lrealsense2 \
    -lvitis_ai_library-facedetect \
    -lglog \
    -pthread \
    -o build/densebox_realsense

echo "Built build/densebox_realsense"
