#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "yolov26_common.hpp"

namespace yolov26 {

struct CameraStreamConfiguration {
    int width = 0;
    int height = 0;
    int fps = 0;
    rs2_format native_format = RS2_FORMAT_ANY;
};

struct CameraConfiguration {
    CameraStreamConfiguration color;
    CameraStreamConfiguration depth;
};

bool is_native_color_format(rs2_format format);

void print_supported_profiles();

CameraConfiguration select_camera_configuration(
    const InferenceOptions& options);

void print_camera_configuration(
    const CameraConfiguration& configuration);

struct DepthMappingCalibration {
    rs2_intrinsics color_intrinsics{};
    rs2_intrinsics depth_intrinsics{};
    rs2_extrinsics color_to_depth{};
    rs2_extrinsics depth_to_color{};
    float depth_scale = 0.0f;
    float depth_min_m = 0.0f;
    float depth_max_m = 0.0f;
};

DepthMappingCalibration create_depth_mapping_calibration(
    const rs2::pipeline_profile& profile,
    float depth_scale);

struct FramePacket {
    uint64_t sequence = 0;
    cv::Mat image;
    std::vector<DetectionAnnotation> annotations;
    Clock::time_point frame_start{};
    Clock::time_point acquired_at{};
    Clock::time_point completed_at{};
    double acquire_ms = 0.0;
    double preprocess_ms = 0.0;
    double dpu_ms = 0.0;
    double depth_map_ms = 0.0;
};

enum class PublishResult {
    published,
    replaced,
    stopped
};

class LatestFrameHandoff {
public:
    PublishResult publish(FramePacket&& packet)
    {
        PublishResult result;

        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (stopped_) {
                return PublishResult::stopped;
            }

            result = pending_.has_value()
                ? PublishResult::replaced
                : PublishResult::published;
            pending_ = std::move(packet);
        }

        condition_.notify_one();
        return result;
    }

    bool wait_pop(FramePacket& packet)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(
            lock,
            [this] {
                return stopped_ || pending_.has_value();
            }
        );

        if (stopped_) {
            return false;
        }

        packet = std::move(*pending_);
        pending_.reset();
        return true;
    }

    bool try_pop(FramePacket& packet)
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (!pending_.has_value()) {
            return false;
        }

        packet = std::move(*pending_);
        pending_.reset();
        return true;
    }

    void stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            pending_.reset();
        }

        condition_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<FramePacket> pending_;
    bool stopped_ = false;
};

class YoloRunner {
public:
    explicit YoloRunner(const std::string& model_path);
    ~YoloRunner();

    YoloRunner(const YoloRunner&) = delete;
    YoloRunner& operator=(const YoloRunner&) = delete;

    int input_width() const;
    int input_height() const;

    void preprocess(
        const cv::Mat& image,
        LetterboxTransform& transform);
    void execute();
    std::vector<Detection> decode(
        float confidence_threshold,
        int max_detections) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

float median_depth_in_color_roi(
    const rs2::depth_frame& depth,
    const DepthMappingCalibration& calibration,
    int x1,
    int y1,
    int x2,
    int y2);

void draw_detection_annotations(
    cv::Mat& image,
    const std::vector<DetectionAnnotation>& annotations,
    const InferenceOptions& options);

class SystemMonitor {
public:
    bool update_if_due()
    {
        const auto now = Clock::now();

        if (has_update_) {
            const double elapsed =
                std::chrono::duration<double>(
                    now - last_update_
                ).count();

            if (elapsed < TELEMETRY_INTERVAL_SECONDS) {
                return false;
            }
        }

        last_update_ = now;
        has_update_ = true;

        if (ams_path_.empty() || ina260_path_.empty()) {
            discover_hwmon();
        }

        update_cpu();
        update_memory();
        update_sensors();
        return true;
    }

    const SystemStats& stats() const
    {
        return stats_;
    }

private:
    struct CpuTimes {
        uint64_t user = 0;
        uint64_t nice = 0;
        uint64_t system = 0;
        uint64_t idle = 0;
        uint64_t iowait = 0;
        uint64_t irq = 0;
        uint64_t softirq = 0;
        uint64_t steal = 0;

        uint64_t idle_total() const
        {
            return idle + iowait;
        }

        uint64_t total() const
        {
            return user + nice + system + idle + iowait
                + irq + softirq + steal;
        }
    };

    void discover_hwmon();
    void update_cpu();
    void update_memory();
    void update_sensors();

    Clock::time_point last_update_{};
    bool has_update_ = false;
    CpuTimes previous_cpu_;
    bool has_cpu_sample_ = false;
    std::string ams_path_;
    std::string ina260_path_;
    SystemStats stats_;
};

std::optional<FramePacket> infer_frame(
    const rs2::frameset& frames,
    const CameraConfiguration& camera_configuration,
    YoloRunner& detector,
    const DepthMappingCalibration& depth_mapping,
    const InferenceOptions& options,
    uint64_t sequence,
    Clock::time_point frame_start,
    Clock::time_point acquire_end);

}
