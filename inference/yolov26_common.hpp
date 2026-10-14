#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace yolov26 {

constexpr int DEFAULT_COLOR_WIDTH = 848;
constexpr int DEFAULT_COLOR_HEIGHT = 480;
constexpr int DEFAULT_COLOR_FPS = 30;
constexpr float DEPTH_ROI_SCALE = 0.35f;
constexpr int DEPTH_SAMPLE_GRID_SIZE = 5;
constexpr float DEPTH_SEARCH_MIN_METERS = 0.1f;
constexpr float DEPTH_SEARCH_MAX_METERS = 10.0f;
static_assert(
    DEPTH_SAMPLE_GRID_SIZE > 0 && DEPTH_SAMPLE_GRID_SIZE % 2 == 1,
    "Depth sampling grid size must be a positive odd number"
);
constexpr double TELEMETRY_INTERVAL_SECONDS = 1.0;
constexpr const char* DEFAULT_MODEL_PATH =
    "yolov26n_facehuman_kria.xmodel";
constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.1f;
constexpr int DEFAULT_MAX_DETECTIONS = 300;
constexpr int NUM_CLASSES = 2;
constexpr int REG_MAX = 1;
constexpr std::array<int, 3> STRIDES = {8, 16, 32};
constexpr std::array<const char*, NUM_CLASSES> CLASS_NAMES = {
    "Person",
    "Human face"
};

using Clock = std::chrono::steady_clock;

inline double elapsed_ms(
    Clock::time_point start,
    Clock::time_point end)
{
    return std::chrono::duration<double, std::milli>(
        end - start
    ).count();
}

struct InferenceOptions {
    int depth_width = 0;
    int depth_height = 0;
    std::string model_path = DEFAULT_MODEL_PATH;
    float confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD;
    int max_detections = DEFAULT_MAX_DETECTIONS;
};

struct Detection {
    int class_id = 0;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct DetectionAnnotation {
    int class_id = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    int depth_x1 = 0;
    int depth_y1 = 0;
    int depth_x2 = 0;
    int depth_y2 = 0;
    float score = 0.0f;
    float distance = 0.0f;
};

struct LetterboxTransform {
    float gain = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct SystemStats {
    bool cpu_available = false;
    double cpu_percent = 0.0;
    bool ram_available = false;
    uint64_t ram_used_mb = 0;
    uint64_t ram_total_mb = 0;
    double ram_percent = 0.0;
    bool temperature_available = false;
    double temperature_c = 0.0;
    bool pl_temperature_available = false;
    double pl_temperature_c = 0.0;
    bool power_available = false;
    double power_w = 0.0;
    bool voltage_available = false;
    double voltage_v = 0.0;
    bool current_available = false;
    double current_a = 0.0;
};

}
