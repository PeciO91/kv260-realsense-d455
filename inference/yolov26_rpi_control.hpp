#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolov26_common.hpp"

namespace yolov26 {
namespace rpi {

constexpr int RTP_PAYLOAD_TYPE = 96;
constexpr int RTP_MTU = 1200;
constexpr int DEFAULT_VIDEO_PORT = 5000;
constexpr int DEFAULT_REDIS_PORT = 6379;
constexpr int DEFAULT_BITRATE_KBPS = 4000;
constexpr double DEFAULT_REDIS_INTERVAL_SECONDS = 1.0;
constexpr double MIN_REDIS_INTERVAL_SECONDS = 0.1;
constexpr double DEFAULT_FACE_STALE_SECONDS = 2.0;
constexpr int DEFAULT_REDIS_TIMEOUT_MS = 200;
constexpr int64_t MAX_SECONDS_NS = 630720000000000000LL;
constexpr double MAX_SECONDS = 630720000.0;
constexpr double RECONNECT_BACKOFF_SECONDS[] = {1.0, 2.0, 4.0, 5.0};
constexpr int RECONNECT_BACKOFF_STEPS = 4;
constexpr int POLL_SLICE_MS = 50;

struct SenderOptions {
    InferenceOptions inference;
    std::string pi_host;
    int video_port = DEFAULT_VIDEO_PORT;
    std::string encoder = "auto";
    int bitrate_kbps = DEFAULT_BITRATE_KBPS;
    std::string redis_host;
    bool redis_host_set = false;
    int redis_port = DEFAULT_REDIS_PORT;
    std::string redis_channel = "face_detection";
    double redis_interval_seconds = DEFAULT_REDIS_INTERVAL_SECONDS;
    double face_stale_seconds = DEFAULT_FACE_STALE_SECONDS;
    int redis_timeout_ms = DEFAULT_REDIS_TIMEOUT_MS;
    bool duration_set = false;
    double duration_seconds = 0.0;
    bool telemetry = false;
    bool list_profiles = false;
};

struct ParseResult {
    SenderOptions options;
    bool show_help = false;
};

void print_sender_usage(const char* argv0);

ParseResult parse_sender_options(int argc, char* argv[]);

bool face_detected(
    const std::vector<DetectionAnnotation>& annotations,
    int image_width,
    int image_height);

std::string face_payload(bool detected);

class LatestFaceState {
public:
    void update(bool detected, Clock::time_point completed)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_ = true;
        detected_ = detected;
        completed_ = completed;
    }

    std::optional<std::pair<bool, Clock::time_point>> snapshot() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!available_) {
            return std::nullopt;
        }
        return std::make_pair(detected_, completed_);
    }

    void invalidate()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        available_ = false;
        detected_ = false;
    }

private:
    mutable std::mutex mutex_;
    bool available_ = false;
    bool detected_ = false;
    Clock::time_point completed_{};
};

bool state_fresh(
    Clock::time_point completed,
    Clock::time_point now,
    double stale_seconds);

Clock::time_point next_report_deadline(
    Clock::time_point origin,
    double interval_seconds,
    Clock::time_point now);

double reconnect_backoff_seconds(int attempt_index);

struct SenderTelemetrySnapshot {
    bool available = false;
    double window_seconds = 0.0;
    double inference_fps = 0.0;
    double appsrc_submissions_per_second = 0.0;
    double encoded_au_per_second = 0.0;
    uint64_t replaced_frames = 0;
    std::array<double, 7> mean_ms{};
};

class SenderTelemetry {
public:
    explicit SenderTelemetry(Clock::time_point start = Clock::now()) : start_(start) {}
    void record_inference(double acquire, double preprocess, double dpu, double decode_depth) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++inferences_;
        sums_[0] += acquire; sums_[1] += preprocess;
        sums_[2] += dpu; sums_[3] += decode_depth;
    }
    void record_submission(double overlay, double push, double start_to_enqueue) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++submissions_;
        sums_[4] += overlay; sums_[5] += push; sums_[6] += start_to_enqueue;
    }
    void record_replacement() {
        std::lock_guard<std::mutex> lock(mutex_);
        ++replacements_;
    }
    std::optional<SenderTelemetrySnapshot> snapshot_if_due(uint64_t encoded_total, Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        const double seconds = std::chrono::duration<double>(now - start_).count();
        if (seconds < 1.0) return std::nullopt;
        if (encoded_total < encoded_origin_) throw std::logic_error("Encoded AU counter regressed");
        SenderTelemetrySnapshot result;
        result.available = true; result.window_seconds = seconds;
        result.inference_fps = static_cast<double>(inferences_) / seconds;
        result.appsrc_submissions_per_second = static_cast<double>(submissions_) / seconds;
        result.encoded_au_per_second = static_cast<double>(encoded_total - encoded_origin_) / seconds;
        result.replaced_frames = replacements_;
        for (std::size_t i = 0; i < sums_.size(); ++i) {
            const uint64_t samples = i < 4 ? inferences_ : submissions_;
            result.mean_ms[i] = samples ? sums_[i] / static_cast<double>(samples) : 0.0;
        }
        inferences_ = 0; submissions_ = 0; sums_.fill(0.0);
        encoded_origin_ = encoded_total; start_ = now;
        return result;
    }
private:
    std::mutex mutex_;
    Clock::time_point start_;
    uint64_t inferences_ = 0;
    uint64_t submissions_ = 0;
    uint64_t replacements_ = 0;
    uint64_t encoded_origin_ = 0;
    std::array<double, 7> sums_{};
};

}
}
