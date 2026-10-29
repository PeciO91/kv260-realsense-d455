#!/bin/sh

set -e

usage() {
    echo "Usage: $0 [--deps-prefix DIR] [--rs-prefix DIR]" >&2
    exit 2
}

DEPS_PREFIX=""
RS_PREFIX="/root/librealsense-rsusb"

while [ $# -gt 0 ]; do
    case "$1" in
        --deps-prefix)
            [ $# -ge 2 ] || usage
            DEPS_PREFIX="$2"
            shift 2
            ;;
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

if [ -z "$DEPS_PREFIX" ]; then
    if [ -d "$HOME/kv260-network-deps/lib/gstreamer-1.0" ]; then
        DEPS_PREFIX="$HOME/kv260-network-deps"
    else
        echo "error: no dependency prefix found" >&2
        echo "  expected $HOME/kv260-network-deps/lib/gstreamer-1.0" >&2
        echo "  or pass --deps-prefix DIR" >&2
        exit 1
    fi
elif [ ! -d "$DEPS_PREFIX/lib/gstreamer-1.0" ]; then
    echo "error: --deps-prefix $DEPS_PREFIX lacks lib/gstreamer-1.0" >&2
    exit 1
fi

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

PKG_CONFIG_PATH="${DEPS_PREFIX}/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PKG_CONFIG_PATH

MISSING=""
for module in opencv4 gstreamer-1.0 gstreamer-app-1.0 \
              gstreamer-video-1.0 hiredis; do
    if ! pkg-config --exists "$module"; then
        MISSING="$MISSING $module"
    fi
done

if [ -n "$MISSING" ]; then
    echo "error: missing pkg-config modules:$MISSING" >&2
    echo "  PKG_CONFIG_PATH=$PKG_CONFIG_PATH" >&2
    exit 1
fi

echo "Dependency prefix: $DEPS_PREFIX"
echo "RealSense prefix:  $RS_PREFIX ($RS_LIBDIR)"

mkdir -p build

g++ \
    -std=c++17 \
    -O2 \
    -Wall \
    -Wextra \
    inference/yolov26/rpi/live_yolov26_rpi.cpp \
    inference/yolov26/core/yolov26_realsense_core.cpp \
    inference/yolov26/rpi/yolov26_rpi_control.cpp \
    inference/yolov26/rpi/yolov26_rpi_transport.cpp \
    -I inference/yolov26/core \
    -I inference/yolov26/rpi \
    -I"${RS_PREFIX}/include" \
    -L"${RS_LIBDIR}" \
    -Wl,-rpath,"${RS_LIBDIR}" \
    -Wl,-rpath,"${DEPS_PREFIX}/lib" \
    $(pkg-config --cflags --libs opencv4 gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 hiredis) \
    -lrealsense2 \
    -lvart-runner \
    -lxir \
    -lunilog \
    -lglog \
    -pthread \
    -o build/live_yolov26_rpi

echo "Built build/live_yolov26_rpi"
