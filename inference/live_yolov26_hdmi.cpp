#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "yolov26_common.hpp"
#include "yolov26_realsense_core.hpp"
#include "yolov26_hdmi_control.hpp"
#include "yolov26_hdmi_display.hpp"

using namespace yolov26;
using namespace yolov26::hdmi;

namespace {

volatile std::sig_atomic_t g_signal_fired = 0;

void on_exit_signal(int)
{
    g_signal_fired = 1;
}

void install_signal_handlers()
{
    struct sigaction action;
    std::memset(&action, 0, sizeof(action));
    action.sa_handler = on_exit_signal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    sigaction(SIGINT, &action, nullptr);
    sigaction(SIGTERM, &action, nullptr);
}

cv::Mat make_test_frame(
    int width,
    int height,
    uint64_t sequence,
    double remaining_seconds)
{
    cv::Mat frame(height, width, CV_8UC3);
    const int bar_width = width / 8;
    const cv::Scalar bars[8] = {
        cv::Scalar(255, 255, 255),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 255),
        cv::Scalar(0, 0, 255),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 0, 0),
    };

    for (int i = 0; i < 8; ++i) {
        const int x1 = i * bar_width;
        const int x2 =
            (i == 7) ? width : (i + 1) * bar_width;
        cv::rectangle(
            frame,
            cv::Rect(x1, 0, x2 - x1, height),
            bars[i],
            cv::FILLED
        );
    }

    const int box = height / 6;
    const int travel = width - box;
    const int x =
        static_cast<int>(travel > 0
            ? (sequence * 7) % (2 * travel)
            : 0);
    const int x_pos =
        x <= travel ? x : 2 * travel - x;
    cv::rectangle(
        frame,
        cv::Rect(x_pos, height / 2 - box / 2, box, box),
        cv::Scalar(0, 0, 0),
        3
    );

    char text[128];
    std::snprintf(
        text,
        sizeof(text),
        "DISPLAY TEST - NO CAMERA / NO INFERENCE"
    );
    cv::putText(
        frame,
        text,
        cv::Point(width / 20, height / 8),
        cv::FONT_HERSHEY_SIMPLEX,
        std::max(0.6, width / 1100.0),
        cv::Scalar(0, 0, 0),
        4,
        cv::LINE_AA
    );
    cv::putText(
        frame,
        text,
        cv::Point(width / 20, height / 8),
        cv::FONT_HERSHEY_SIMPLEX,
        std::max(0.6, width / 1100.0),
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    std::snprintf(
        text,
        sizeof(text),
        "frame %llu  %.1f s left",
        static_cast<unsigned long long>(sequence),
        remaining_seconds
    );
    cv::putText(
        frame,
        text,
        cv::Point(width / 20, height - height / 10),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );

    return frame;
}

void draw_hdmi_telemetry(
    cv::Mat& image,
    const SystemStats& system,
    const TelemetrySnapshot* last_snapshot)
{
    char text[128];
    int y = 20;
    const auto line = [&](const char* value) {
        cv::putText(
            image,
            value,
            cv::Point(8, y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA
        );
        y += 15;
    };

    if (last_snapshot && last_snapshot->submissions > 0) {
        std::snprintf(
            text,
            sizeof(text),
            "output %.1f FPS",
            last_snapshot->fps
        );
        line(text);
        std::snprintf(
            text,
            sizeof(text),
            "acq %.1f ms  prep %.1f ms",
            last_snapshot->acquire_ms,
            last_snapshot->preprocess_ms
        );
        line(text);
        std::snprintf(
            text,
            sizeof(text),
            "dpu %.1f ms  depth %.1f ms",
            last_snapshot->dpu_ms,
            last_snapshot->depth_ms
        );
        line(text);
        std::snprintf(
            text,
            sizeof(text),
            "e2e %.1f ms mean  %.1f ms max",
            last_snapshot->e2e_mean_ms,
            last_snapshot->e2e_max_ms
        );
        line(text);
    } else {
        line("metrics: warming up");
    }

    if (system.cpu_available) {
        std::snprintf(
            text, sizeof(text), "CPU %.0f%%", system.cpu_percent
        );
        line(text);
    }
    if (system.ram_available) {
        std::snprintf(
            text,
            sizeof(text),
            "RAM %llu/%llu MB",
            static_cast<unsigned long long>(system.ram_used_mb),
            static_cast<unsigned long long>(system.ram_total_mb)
        );
        line(text);
    }
    if (system.temperature_available) {
        std::snprintf(
            text,
            sizeof(text),
            "PS temp %.1f C",
            system.temperature_c
        );
        line(text);
    }
    if (system.pl_temperature_available) {
        std::snprintf(
            text,
            sizeof(text),
            "PL temp %.1f C",
            system.pl_temperature_c
        );
        line(text);
    }
    if (system.power_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Power %.2f W",
            system.power_w
        );
        line(text);
    }
}

struct WorkerGuard {
    std::atomic<bool>& running;
    LatestFrameHandoff& handoff;
    std::thread& worker;

    ~WorkerGuard()
    {
        stop_and_join();
    }

    void stop_and_join()
    {
        running.store(false);
        handoff.stop();
        if (worker.joinable()) {
            worker.join();
        }
    }
};

using WorkerBody = std::function<void(std::atomic<bool>&)>;

std::thread start_capture_worker(
    const Options& options,
    std::atomic<bool>& running,
    std::exception_ptr& worker_error,
    const WorkerBody& synthetic_body,
    const WorkerBody& real_body)
{
    WorkerBody body =
        options.test_pattern ? synthetic_body : real_body;
    return std::thread([body, &running, &worker_error]() {
        try {
            body(running);
        } catch (...) {
            worker_error = std::current_exception();
        }
        running.store(false);
    });
}

}

int main(int argc, char* argv[])
{
    try {
        const ParseResult parsed = parse_options(argc, argv);

        if (parsed.show_help) {
            print_usage(argv[0]);
            return 0;
        }

        const Options& options = parsed.options;
        const double duration_seconds =
            options.duration_set
                ? options.duration_seconds
                : (options.test_pattern
                    ? DEFAULT_TEST_PATTERN_SECONDS
                    : -1.0);

        install_signal_handlers();

        HdmiDisplay display(options.display, options.fullscreen);

        std::cout
            << "Display: " << options.display << " "
            << display.target_width() << "x"
            << display.target_height()
            << (options.fullscreen ? " fullscreen" : " windowed")
            << std::endl;

        std::atomic<bool> running{true};
        std::exception_ptr worker_error;
        LatestFrameHandoff handoff;
        std::thread worker;
        WorkerGuard worker_guard{running, handoff, worker};

        const WorkerBody synthetic_body =
            [&](std::atomic<bool>& active) {
                uint64_t sequence = 0;
                const auto start = Clock::now();

                while (active.load() && !g_signal_fired) {
                    double remaining = 0.0;
                    if (duration_seconds > 0.0) {
                        remaining = duration_seconds
                            - std::chrono::duration<double>(
                                Clock::now() - start
                            ).count();
                    }

                    FramePacket packet;
                    packet.sequence = sequence;
                    packet.image = make_test_frame(
                        DEFAULT_COLOR_WIDTH,
                        DEFAULT_COLOR_HEIGHT,
                        sequence,
                        remaining
                    );
                    packet.frame_start = Clock::now();
                    packet.acquired_at = packet.frame_start;
                    packet.completed_at = packet.frame_start;
                    ++sequence;

                    const PublishResult result =
                        handoff.publish(std::move(packet));

                    if (result == PublishResult::stopped) {
                        break;
                    }

                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(33)
                    );
                }
            };

        const WorkerBody real_body =
            [&](std::atomic<bool>& active) {
                {
                    struct stat model_stat;
                    std::ifstream model_stream;

                    if (::stat(
                            options.inference.model_path.c_str(),
                            &model_stat) != 0
                        || !S_ISREG(model_stat.st_mode)
                        || model_stat.st_size == 0) {
                        throw std::invalid_argument(
                            "model file missing, unreadable,"
                            " or empty: "
                            + options.inference.model_path
                        );
                    }

                    model_stream.open(
                        options.inference.model_path,
                        std::ios::binary
                    );

                    if (!model_stream.good()
                        || model_stream.peek()
                            == std::char_traits<char>::eof()) {
                        throw std::invalid_argument(
                            "model file missing, unreadable,"
                            " or empty: "
                            + options.inference.model_path
                        );
                    }
                }

                const CameraConfiguration camera_configuration =
                    select_camera_configuration(
                        options.inference
                    );
                print_camera_configuration(
                    camera_configuration
                );

                std::cout
                    << "Loading YOLOv26 model: "
                    << options.inference.model_path
                    << std::endl;

                YoloRunner detector(
                    options.inference.model_path
                );

                std::cout
                    << "Model input: "
                    << detector.input_width()
                    << "x"
                    << detector.input_height()
                    << std::endl;

                std::cout
                    << "Confidence threshold: "
                    << options.inference.confidence_threshold
                    << std::endl;

                std::cout
                    << "Max detections: "
                    << options.inference.max_detections
                    << std::endl;

                rs2::pipeline pipeline;
                rs2::config config;

                config.enable_stream(
                    RS2_STREAM_COLOR,
                    camera_configuration.color.width,
                    camera_configuration.color.height,
                    RS2_FORMAT_BGR8,
                    camera_configuration.color.fps
                );
                config.enable_stream(
                    RS2_STREAM_DEPTH,
                    camera_configuration.depth.width,
                    camera_configuration.depth.height,
                    RS2_FORMAT_Z16,
                    camera_configuration.depth.fps
                );

                std::cout
                    << "Starting RealSense D455..."
                    << std::endl;

                pipeline.start(config);

                struct CameraStopGuard {
                    rs2::pipeline& pipeline;
                    bool active = true;
                    ~CameraStopGuard()
                    {
                        if (active) {
                            try {
                                pipeline.stop();
                            } catch (...) {
                            }
                        }
                    }
                } camera_guard{pipeline};

                auto profile = pipeline.get_active_profile();
                auto device = profile.get_device();
                auto depth_sensor =
                    device.first<rs2::depth_sensor>();
                const float depth_scale =
                    depth_sensor.get_depth_scale();
                const DepthMappingCalibration depth_mapping =
                    create_depth_mapping_calibration(
                        profile,
                        depth_scale
                    );

                std::cout
                    << "Camera: "
                    << device.get_info(RS2_CAMERA_INFO_NAME)
                    << std::endl;
                std::cout
                    << "Depth scale: " << depth_scale
                    << " m/unit" << std::endl;
                std::cout
                    << "Depth sampling: "
                    << DEPTH_SAMPLE_GRID_SIZE << "x"
                    << DEPTH_SAMPLE_GRID_SIZE
                    << " calibrated grid" << std::endl;
                std::cout
                    << "Warming up..." << std::endl;

                int warmed = 0;
                const auto warmup_deadline =
                    Clock::now() + std::chrono::seconds(5);

                while (warmed < 10
                    && active.load()
                    && !g_signal_fired) {
                    if (Clock::now() >= warmup_deadline) {
                        throw std::runtime_error(
                            "no usable camera frames"
                            " during warmup"
                        );
                    }

                    rs2::frameset frames;
                    if (pipeline.try_wait_for_frames(
                            &frames, 100)
                        && frames.get_color_frame()
                        && frames.get_depth_frame()) {
                        ++warmed;
                    }
                }

                if (!active.load() || g_signal_fired) {
                    return;
                }

                std::cout
                    << "Streaming. Press Escape, q or"
                    << " Ctrl+C to stop."
                    << std::endl;

                uint64_t sequence = 0;
                Clock::time_point last_valid = Clock::now();

                while (active.load() && !g_signal_fired) {
                    const auto frame_start = Clock::now();
                    rs2::frameset frames;

                    if (!pipeline.try_wait_for_frames(
                            &frames, 100)) {
                        if (std::chrono::duration<double>(
                                Clock::now() - last_valid
                            ).count() > 5.0) {
                            throw std::runtime_error(
                                "no frames from camera"
                                " for 5 seconds"
                            );
                        }
                        continue;
                    }

                    const auto acquire_end = Clock::now();

                    std::optional<FramePacket> packet =
                        infer_frame(
                            frames,
                            camera_configuration,
                            detector,
                            depth_mapping,
                            options.inference,
                            sequence,
                            frame_start,
                            acquire_end
                        );

                    if (!packet) {
                        if (std::chrono::duration<double>(
                                Clock::now() - last_valid
                            ).count() > 5.0) {
                            throw std::runtime_error(
                                "no usable camera frames"
                                " for 5 seconds"
                            );
                        }
                        continue;
                    }

                    ++sequence;
                    last_valid = Clock::now();

                    const PublishResult result =
                        handoff.publish(std::move(*packet));

                    if (result == PublishResult::stopped) {
                        break;
                    }
                }
            };

        worker = start_capture_worker(
            options,
            running,
            worker_error,
            synthetic_body,
            real_body
        );

        const auto display_start = Clock::now();
        SystemMonitor system_monitor;
        DisplayTelemetry telemetry;
        std::optional<TelemetrySnapshot> last_snapshot;
        FramePacket packet;

        while (running.load() && !g_signal_fired) {
            if (duration_seconds > 0.0
                && std::chrono::duration<double>(
                    Clock::now() - display_start
                ).count() >= duration_seconds) {
                break;
            }

            if (handoff.try_pop(packet)) {
                draw_detection_annotations(
                    packet.image,
                    packet.annotations,
                    options.inference
                );

                if (options.telemetry) {
                    system_monitor.update_if_due();
                    draw_hdmi_telemetry(
                        packet.image,
                        system_monitor.stats(),
                        last_snapshot
                            ? &*last_snapshot
                            : nullptr
                    );
                }

                const auto submit_start = Clock::now();
                if (!display.show(packet.image)) {
                    break;
                }

                const auto now = Clock::now();
                const double e2e_ms = elapsed_ms(
                    packet.frame_start,
                    now
                );

                if (options.telemetry) {
                    telemetry.record(
                        packet.acquire_ms,
                        packet.preprocess_ms,
                        packet.dpu_ms,
                        packet.depth_map_ms,
                        e2e_ms
                    );

                    std::optional<TelemetrySnapshot> due =
                        telemetry.snapshot_if_due(now);

                    if (due) {
                        last_snapshot = *due;

                        std::ostringstream line;
                        line
                            << "[telemetry] fps="
                            << due->fps
                            << " e2e_mean="
                            << due->e2e_mean_ms
                            << " ms e2e_max="
                            << due->e2e_max_ms
                            << " ms";

                        if (!options.test_pattern) {
                            line
                                << " acquire="
                                << due->acquire_ms
                                << " ms preprocess="
                                << due->preprocess_ms
                                << " ms dpu="
                                << due->dpu_ms
                                << " ms depth="
                                << due->depth_ms
                                << " ms";
                        }

                        const SystemStats& system =
                            system_monitor.stats();

                        if (system.cpu_available) {
                            line
                                << " cpu="
                                << system.cpu_percent
                                << "%";
                        }
                        if (system.ram_available) {
                            line
                                << " ram="
                                << system.ram_used_mb
                                << "/"
                                << system.ram_total_mb
                                << " MB";
                        }
                        if (system.temperature_available) {
                            line
                                << " temp="
                                << system.temperature_c
                                << " C";
                        }

                        std::cout << line.str() << std::endl;
                    }
                }
                continue;
            }

            if (!display.poll()) {
                break;
            }
        }

        worker_guard.stop_and_join();

        if (worker_error && !g_signal_fired) {
            std::rethrow_exception(worker_error);
        }

        std::cout << "Display closed." << std::endl;
        return 0;
    }

    catch (const std::exception& e) {
        std::cerr
            << "Error: " << e.what() << std::endl;
        return 1;
    }
}
