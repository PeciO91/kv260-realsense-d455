#!/bin/sh

set -e

usage() {
    echo "Usage: $0 [--rs-prefix DIR]" >&2
    exit 2
}

RS_PREFIX="/root/librealsense-rsusb"

while [ $# -gt 0 ]; do
    case "$1" in
        --rs-prefix)
            [ $# -ge 2 ] || usage
            RS_PREFIX="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [ ! -d "$RS_PREFIX/include" ]; then
    echo "error: RealSense prefix $RS_PREFIX lacks include/" >&2
    exit 1
fi

RS_LIBDIR=""
for candidate in "$RS_PREFIX/lib64" "$RS_PREFIX/lib"; do
    if [ -f "$candidate/librealsense2.so" ] \
        || ls "$candidate"/librealsense2.so.* >/dev/null 2>&1; then
        RS_LIBDIR="$candidate"
        break
    fi
done

if [ -z "$RS_LIBDIR" ]; then
    echo "error: no librealsense2 under $RS_PREFIX/lib64 or lib" >&2
    exit 1
fi

MISSING=""
for module in opencv4 x11 gtk+-3.0 libdrm; do
    if ! pkg-config --exists "$module"; then
        MISSING="$MISSING $module"
    fi
done

if [ -n "$MISSING" ]; then
    echo "error: missing pkg-config modules:$MISSING" >&2
    exit 1
fi

echo "RealSense prefix:  $RS_PREFIX ($RS_LIBDIR)"

mkdir -p build

g++ \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    inference/yolov26/hdmi/live_yolov26_hdmi.cpp \
    inference/yolov26/core/yolov26_realsense_core.cpp \
    inference/yolov26/hdmi/yolov26_hdmi_control.cpp \
    inference/yolov26/hdmi/yolov26_hdmi_display.cpp \
    inference/yolov26/hdmi/yolov26_kms_display.cpp \
    -I inference/yolov26/core \
    -I inference/yolov26/hdmi \
    -I"${RS_PREFIX}/include" \
    -L"${RS_LIBDIR}" \
    -Wl,-rpath,"${RS_LIBDIR}" \
    $(pkg-config --cflags --libs opencv4 x11 gtk+-3.0 libdrm) \
    -lrealsense2 \
    -lvart-runner \
    -lxir \
    -lunilog \
    -lglog \
    -pthread \
    -o build/live_yolov26_hdmi

echo "Built build/live_yolov26_hdmi"
