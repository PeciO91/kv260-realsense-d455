#pragma once

#include <string>

#include "yolov26_common.hpp"

namespace yolov26 {
namespace hdmi {

constexpr const char* DEFAULT_DISPLAY = ":0";
constexpr double DEFAULT_TEST_PATTERN_SECONDS = 10.0;
constexpr double MAX_DURATION_SECONDS = 630720000.0;

struct Options {
    InferenceOptions inference;
    std::string display = DEFAULT_DISPLAY;
    bool fullscreen = true;
    bool telemetry = false;
    bool test_pattern = false;
    bool duration_set = false;
    double duration_seconds = 0.0;
};

struct ParseResult {
    Options options;
    bool show_help = false;
};

void print_usage(const char* argv0);

ParseResult parse_options(int argc, char* argv[]);

struct FittedImage {
    int width = 0;
    int height = 0;
    int x = 0;
    int y = 0;
};

FittedImage fit_image(
    int src_width,
    int src_height,
    int dst_width,
    int dst_height);

}
}
