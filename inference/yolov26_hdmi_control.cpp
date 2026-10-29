#include "yolov26_hdmi_control.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace yolov26 {
namespace hdmi {

namespace {

int parse_int(
    const char* name,
    const std::string& text,
    long long minimum,
    long long maximum)
{
    if (text.empty()) {
        throw std::invalid_argument(
            std::string("invalid ") + name + " '" + text + "'"
        );
    }

    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);

    if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        throw std::invalid_argument(
            std::string("invalid integer ") + name
            + " '" + text + "'"
        );
    }

    if (value < minimum || value > maximum) {
        throw std::invalid_argument(
            std::string(name) + " out of range '" + text + "'"
        );
    }

    return static_cast<int>(value);
}

double parse_seconds(
    const char* name,
    const std::string& text)
{
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);

    if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        throw std::invalid_argument(
            std::string("invalid number ") + name
            + " '" + text + "'"
        );
    }

    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(
            std::string(name) + " must be positive '"
            + text + "'"
        );
    }

    if (value > MAX_DURATION_SECONDS) {
        throw std::invalid_argument(
            std::string(name) + " too large '" + text
            + "' (max " + std::to_string(MAX_DURATION_SECONDS)
            + " s)"
        );
    }

    return value;
}

bool parse_bool(const char* name, const std::string& text)
{
    if (text == "true") {
        return true;
    }

    if (text == "false") {
        return false;
    }

    throw std::invalid_argument(
        std::string(name) + " must be true or false"
    );
}

}

void print_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " [options]" << std::endl
        << std::endl
        << "Show the D455 + DPU YOLOv26 stream on the local"
        << " X desktop." << std::endl
        << std::endl
        << "Options:" << std::endl
        << "  --renderer x11|kms        Display backend"
        << " (default x11; kms needs Xorg stopped)" << std::endl
        << "  --kms-plane primary|overlay  KMS output plane"
        << " (default primary; kms only)" << std::endl
        << "  --display NAME            X display"
        << " (default " << DEFAULT_DISPLAY << ")" << std::endl
        << "  --fullscreen true|false   Fullscreen window"
        << " (default true)" << std::endl
        << "  --telemetry true|false    Overlay/log metrics"
        << " (default false)" << std::endl
        << "  --test-pattern            Synthetic frames, no"
        << " camera/model (default "
        << DEFAULT_TEST_PATTERN_SECONDS
        << " s)" << std::endl
        << "  --duration-seconds SECONDS  Stop after SECONDS"
        << " (default: " << DEFAULT_TEST_PATTERN_SECONDS
        << " s with --test-pattern," << std::endl
        << "                            run until stopped"
        << " otherwise)" << std::endl
        << "  --model PATH              Model xmodel path"
        << " (default " << DEFAULT_MODEL_PATH << ")" << std::endl
        << "  --confidence VALUE        Detection confidence 0..1"
        << " (default " << DEFAULT_CONFIDENCE_THRESHOLD << ")"
        << std::endl
        << "  --max-detections COUNT    Maximum detections per"
        << " frame (default " << DEFAULT_MAX_DETECTIONS << ")"
        << std::endl
        << "  --depth-width WIDTH       Depth stream width"
        << std::endl
        << "  --depth-height HEIGHT     Depth stream height"
        << std::endl
        << "  --show-depth-roi true|false  Draw the depth"
        << " sampling ROI" << std::endl
        << "                            (default false)" << std::endl
        << "  --show-confidence true|false  Include the score in"
        << " box labels" << std::endl
        << "                            (default true)" << std::endl
        << "  --help                    Show this help and exit"
        << std::endl
        << std::endl
        << "Exit on Escape, q, window close or Ctrl+C."
        << " --test-pattern and --help never touch the"
        << std::endl
        << "camera or load a model."
        << std::endl;
}

ParseResult parse_options(int argc, char* argv[])
{
    ParseResult result;
    Options& options = result.options;
    bool kms_plane_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);

        auto require_value = [&](const char* name) {
            if (++i >= argc) {
                throw std::invalid_argument(
                    std::string("Missing value for ") + name
                );
            }
            return std::string(argv[i]);
        };

        if (argument == "--display") {
            options.display = require_value("--display");
            if (options.display.empty()) {
                throw std::invalid_argument(
                    "--display must not be empty"
                );
            }
        } else if (argument == "--renderer") {
            options.renderer = require_value("--renderer");
            if (options.renderer != "x11"
                && options.renderer != "kms") {
                throw std::invalid_argument(
                    "--renderer must be x11 or kms"
                );
            }
        } else if (argument == "--kms-plane") {
            options.kms_plane = require_value("--kms-plane");
            if (options.kms_plane != "primary"
                && options.kms_plane != "overlay") {
                throw std::invalid_argument(
                    "--kms-plane must be primary or overlay"
                );
            }
            kms_plane_set = true;
        } else if (argument == "--fullscreen") {
            options.fullscreen = parse_bool(
                "--fullscreen",
                require_value("--fullscreen")
            );
        } else if (argument == "--telemetry") {
            options.telemetry = parse_bool(
                "--telemetry",
                require_value("--telemetry")
            );
        } else if (argument == "--test-pattern") {
            options.test_pattern = true;
        } else if (argument == "--duration-seconds") {
            options.duration_seconds = parse_seconds(
                "--duration-seconds",
                require_value("--duration-seconds")
            );
            options.duration_set = true;
        } else if (argument == "--model") {
            options.inference.model_path =
                require_value("--model");
            if (options.inference.model_path.empty()) {
                throw std::invalid_argument(
                    "--model must not be empty"
                );
            }
        } else if (argument == "--confidence") {
            char* end = nullptr;
            const std::string text =
                require_value("--confidence");
            const float value = std::strtof(text.c_str(), &end);
            if (end == text.c_str() || *end != '\0'
                || !std::isfinite(value)
                || value < 0.0f || value > 1.0f) {
                throw std::invalid_argument(
                    "--confidence must be a finite number 0..1"
                );
            }
            options.inference.confidence_threshold = value;
        } else if (argument == "--max-detections") {
            options.inference.max_detections = parse_int(
                "--max-detections",
                require_value("--max-detections"),
                1,
                std::numeric_limits<int>::max()
            );
        } else if (argument == "--depth-width") {
            options.inference.depth_width = parse_int(
                "--depth-width",
                require_value("--depth-width"),
                1,
                4096
            );
        } else if (argument == "--depth-height") {
            options.inference.depth_height = parse_int(
                "--depth-height",
                require_value("--depth-height"),
                1,
                4096
            );
        } else if (argument == "--show-depth-roi") {
            options.inference.show_depth_roi = parse_bool(
                "--show-depth-roi",
                require_value("--show-depth-roi")
            );
        } else if (argument == "--show-confidence") {
            options.inference.show_confidence = parse_bool(
                "--show-confidence",
                require_value("--show-confidence")
            );
        } else if (argument == "--help" || argument == "-h") {
            result.show_help = true;
        } else {
            throw std::invalid_argument(
                "Unknown option: " + argument
            );
        }
    }

    if (kms_plane_set && options.renderer != "kms") {
        throw std::invalid_argument(
            "--kms-plane requires --renderer kms"
        );
    }

    if ((options.inference.depth_width == 0)
        != (options.inference.depth_height == 0)) {
        throw std::invalid_argument(
            "--depth-width and --depth-height must be set together"
        );
    }

    return result;
}

FittedImage fit_image(
    int src_width,
    int src_height,
    int dst_width,
    int dst_height)
{
    if (src_width <= 0 || src_height <= 0
        || dst_width <= 0 || dst_height <= 0) {
        throw std::invalid_argument(
            "fit_image requires positive dimensions"
        );
    }

    const double scale = std::min(
        static_cast<double>(dst_width)
            / static_cast<double>(src_width),
        static_cast<double>(dst_height)
            / static_cast<double>(src_height)
    );

    FittedImage fit;
    fit.width = std::max(
        1,
        static_cast<int>(src_width * scale)
    );
    fit.height = std::max(
        1,
        static_cast<int>(src_height * scale)
    );

    if (fit.width > dst_width) {
        fit.width = dst_width;
    }
    if (fit.height > dst_height) {
        fit.height = dst_height;
    }

    fit.x = (dst_width - fit.width) / 2;
    fit.y = (dst_height - fit.height) / 2;
    return fit;
}

void DisplayTelemetry::record(
    double acquire_ms,
    double preprocess_ms,
    double dpu_ms,
    double depth_ms,
    double e2e_ms)
{
    ++submissions_;
    sums_[0] += acquire_ms;
    sums_[1] += preprocess_ms;
    sums_[2] += dpu_ms;
    sums_[3] += depth_ms;
    sums_[4] += e2e_ms;
    if (e2e_ms > e2e_max_) {
        e2e_max_ = e2e_ms;
    }
}

std::optional<TelemetrySnapshot> DisplayTelemetry::snapshot_if_due(
    Clock::time_point now)
{
    const double seconds =
        std::chrono::duration<double>(now - start_).count();
    if (seconds < TELEMETRY_INTERVAL_SECONDS) {
        return std::nullopt;
    }

    TelemetrySnapshot result;
    result.submissions = submissions_;
    result.window_seconds = seconds;
    result.fps = static_cast<double>(submissions_) / seconds;
    if (submissions_ > 0) {
        const double n = static_cast<double>(submissions_);
        result.acquire_ms = sums_[0] / n;
        result.preprocess_ms = sums_[1] / n;
        result.dpu_ms = sums_[2] / n;
        result.depth_ms = sums_[3] / n;
        result.e2e_mean_ms = sums_[4] / n;
    }
    result.e2e_max_ms = e2e_max_;

    submissions_ = 0;
    sums_.fill(0.0);
    e2e_max_ = 0.0;
    start_ = now;
    return result;
}

}
}
