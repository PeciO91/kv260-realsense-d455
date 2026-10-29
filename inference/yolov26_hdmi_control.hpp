#pragma once

#include <array>
#include <optional>
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
    std::string renderer = "x11";
    std::string kms_plane = "primary";
    std::string kms_scaling = "nearest";
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

struct TelemetrySnapshot {
    uint64_t submissions = 0;
    double window_seconds = 0.0;
    double fps = 0.0;
    double acquire_ms = 0.0;
    double preprocess_ms = 0.0;
    double dpu_ms = 0.0;
    double depth_ms = 0.0;
    double e2e_mean_ms = 0.0;
    double e2e_max_ms = 0.0;
    double wait_ms = 0.0;
    double draw_ms = 0.0;
    double show_ms = 0.0;
};

class DisplayTelemetry {
public:
    void record(
        double acquire_ms,
        double preprocess_ms,
        double dpu_ms,
        double depth_ms,
        double e2e_ms,
        double wait_ms,
        double draw_ms,
        double show_ms);

    std::optional<TelemetrySnapshot> snapshot_if_due(
        Clock::time_point now);

private:
    Clock::time_point start_ = Clock::now();
    uint64_t submissions_ = 0;
    std::array<double, 8> sums_{};
    double e2e_max_ = 0.0;
};

}
}
