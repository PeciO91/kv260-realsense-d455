#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <fstream>
#include <pthread.h>
#include <sys/stat.h>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

#include "yolov26_common.hpp"
#include "yolov26_realsense_core.hpp"
#include "yolov26_rpi_control.hpp"
#include "yolov26_rpi_transport.hpp"

using namespace yolov26;
using namespace yolov26::rpi;

namespace {

std::atomic<bool> g_workers_stop{false};

class SignalWatch {
public:
    explicit SignalWatch(std::atomic<bool>& fired) : fired_(fired)
    {
        sigemptyset(&watched_);
        sigaddset(&watched_, SIGINT);
        sigaddset(&watched_, SIGTERM);

        if (pthread_sigmask(SIG_BLOCK, &watched_, &saved_) != 0) {
            throw std::runtime_error("cannot block process signals");
        }

        mask_owned_ = true;

        try {
            watcher_ = std::thread([this]() { run(); });
        } catch (...) {
            pthread_sigmask(SIG_SETMASK, &saved_, nullptr);
            mask_owned_ = false;
            throw;
        }
    }

    SignalWatch(const SignalWatch&) = delete;
    SignalWatch& operator=(const SignalWatch&) = delete;

    ~SignalWatch()
    {
        stop_.store(true, std::memory_order_relaxed);

        if (watcher_.joinable()) {
            watcher_.join();
        }

        if (mask_owned_) {
            pthread_sigmask(SIG_SETMASK, &saved_, nullptr);
        }
    }

private:
    void run()
    {
        while (!stop_.load(std::memory_order_relaxed)) {
            timespec window{};
            window.tv_nsec = 50 * 1000 * 1000;
            const int received =
                sigtimedwait(&watched_, nullptr, &window);

            if (received == SIGINT || received == SIGTERM) {
                fired_.store(true, std::memory_order_relaxed);
                return;
            }

            if (received < 0
                && errno != EAGAIN && errno != EINTR) {
                return;
            }
        }
    }

    std::atomic<bool>& fired_;
    std::atomic<bool> stop_{false};
    std::thread watcher_;
    sigset_t watched_{};
    sigset_t saved_{};
    bool mask_owned_ = false;
};

class CompositeStop : public StopToken {
public:
    explicit CompositeStop(const std::atomic<bool>& signal)
        : signal_(signal)
    {
    }

    bool requested() const override
    {
        return signal_.load() || g_workers_stop.load();
    }

private:
    const std::atomic<bool>& signal_;
};

constexpr int SENDER_PANEL_WIDTH = 400;
constexpr int SENDER_PANEL_HEIGHT = 300;
constexpr int SENDER_PANEL_MARGIN = 10;
constexpr int SENDER_PANEL_X =
    DEFAULT_COLOR_WIDTH
    - SENDER_PANEL_WIDTH
    - SENDER_PANEL_MARGIN;
constexpr int SENDER_PANEL_Y = SENDER_PANEL_MARGIN;

void draw_sender_telemetry(
    cv::Mat& image,
    const SenderTelemetrySnapshot& stats,
    const SystemStats& system)
{
    if (image.cols
        < SENDER_PANEL_X + SENDER_PANEL_WIDTH) {
        return;
    }

    const cv::Rect panel(
        SENDER_PANEL_X,
        SENDER_PANEL_Y,
        SENDER_PANEL_WIDTH,
        SENDER_PANEL_HEIGHT
    );

    cv::Mat roi = image(panel);
    const cv::Mat dark(roi.size(), roi.type(), cv::Scalar(0, 0, 0));
    cv::addWeighted(roi, 0.4, dark, 0.6, 0.0, roi);
    cv::rectangle(
        image,
        panel,
        cv::Scalar(80, 80, 80),
        1
    );

    auto text = [&](int row, const std::string& value) {
        cv::putText(
            image,
            value,
            cv::Point(
                SENDER_PANEL_X + 10,
                SENDER_PANEL_Y + 22 + row * 15
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA
        );
    };

    const auto format_ms = [](double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.1f ms", value);
        return std::string(buffer);
    };

    const auto format_rate = [](double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.1f/s", value);
        return std::string(buffer);
    };

    const auto format_percent = [](double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.0f%%", value);
        return std::string(buffer);
    };

    const auto format_temp = [](double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.1f C", value);
        return std::string(buffer);
    };

    if (!stats.available) {
        text(0, "Sender telemetry: N/A");
    } else {
        text(0,
            "inference " + format_rate(stats.inference_fps)
            + " submissions "
            + format_rate(stats.appsrc_submissions_per_second)
        );
        text(1,
            "encoded AU "
            + format_rate(stats.encoded_au_per_second)
            + " replaced "
            + std::to_string(stats.replaced_frames)
        );
        text(2,
            "acquire " + format_ms(stats.mean_ms[0])
            + "  preprocess " + format_ms(stats.mean_ms[1])
        );
        text(3,
            "DPU " + format_ms(stats.mean_ms[2])
            + "  decode-depth " + format_ms(stats.mean_ms[3])
        );
        text(4,
            "overlay " + format_ms(stats.mean_ms[4])
        );
        text(5,
            "push preparation + admission "
            + format_ms(stats.mean_ms[5])
        );
        text(6,
            "host start-to-enqueue "
            + format_ms(stats.mean_ms[6])
        );
    }

    text(8, "System telemetry");
    text(9,
        system.cpu_available
            ? "CPU " + format_percent(system.cpu_percent)
            : "CPU N/A"
    );
    text(10,
        system.ram_available
            ? "RAM "
                + std::to_string(system.ram_used_mb)
                + " / "
                + std::to_string(system.ram_total_mb)
                + " MB ("
                + format_percent(system.ram_percent)
                + ")"
            : "RAM N/A"
    );
    text(11,
        system.temperature_available
            ? "PS temp " + format_temp(system.temperature_c)
            : "PS temp N/A"
    );
    text(12,
        system.pl_temperature_available
            ? "PL temp " + format_temp(system.pl_temperature_c)
            : "PL temp N/A"
    );
    text(13,
        system.power_available
            ? "Power "
                + std::to_string(system.power_w).substr(0, 5)
                + " W"
            : "Power N/A"
    );
    text(14,
        system.voltage_available
            ? "Voltage "
                + std::to_string(system.voltage_v).substr(0, 5)
                + " V"
            : "Voltage N/A"
    );
    text(15,
        system.current_available
            ? "Current "
                + std::to_string(system.current_a).substr(0, 4)
                + " A"
            : "Current N/A"
    );
}

void log_sender_telemetry(
    const SenderTelemetrySnapshot& stats)
{
    std::cout
        << "[telemetry] inference="
        << stats.inference_fps
        << " FPS submissions="
        << stats.appsrc_submissions_per_second
        << "/s encoded_au="
        << stats.encoded_au_per_second
        << "/s replaced="
        << stats.replaced_frames
        << " acquire_ms="
        << stats.mean_ms[0]
        << " preprocess_ms="
        << stats.mean_ms[1]
        << " dpu_ms="
        << stats.mean_ms[2]
        << " decode_depth_ms="
        << stats.mean_ms[3]
        << " overlay_ms="
        << stats.mean_ms[4]
        << " push_prep_admission_ms="
        << stats.mean_ms[5]
        << " host_start_to_enqueue_ms="
        << stats.mean_ms[6]
        << std::endl;
}

std::string select_encoder(
    const SenderOptions& options,
    const StopToken& stop)
{
    const auto try_kind = [&](const char* kind)
        -> std::optional<std::string> {
        if (stop.requested()) {
            return std::string("cancelled");
        }

        const EncoderPreflightResult result = smoke_encoder(
            kind,
            DEFAULT_COLOR_WIDTH,
            DEFAULT_COLOR_HEIGHT,
            DEFAULT_COLOR_FPS,
            options.bitrate_kbps,
            stop
        );

        if (result.ok) {
            std::cout
                << "Encoder preflight passed: "
                << kind
                << std::endl;
            return std::nullopt;
        }

        if (result.detail == "cancelled") {
            return std::string("cancelled");
        }

        return result.detail;
    };

    if (options.encoder == "auto") {
        for (const char* kind : {"omx", "x264"}) {
            const std::optional<std::string> failure =
                try_kind(kind);

            if (!failure) {
                return kind;
            }

            if (*failure == "cancelled") {
                throw std::runtime_error("cancelled");
            }

            std::cerr
                << "Encoder " << kind << " unavailable: "
                << *failure
                << std::endl;
        }

        throw std::runtime_error(
            "no usable H.264 encoder (omx/x264)"
        );
    }

    const char* kind =
        options.encoder == "x264" ? "x264" : "omx";
    const std::optional<std::string> failure =
        try_kind(kind);

    if (failure) {
        if (*failure == "cancelled") {
            throw std::runtime_error("cancelled");
        }
        throw std::runtime_error(
            std::string("encoder ") + kind
            + " failed preflight: " + *failure
        );
    }

    return kind;
}

}

int main(int argc, char* argv[])
{
    std::atomic<bool> signal_fired{false};

    try {
        const ParseResult parsed = parse_sender_options(argc, argv);

        if (parsed.show_help) {
            print_sender_usage(argv[0]);
            return 0;
        }

        SignalWatch signal_watch(signal_fired);
        const SenderOptions& options = parsed.options;
        const InferenceOptions& inference = options.inference;
        const CompositeStop stop(signal_fired);

        if (options.list_profiles) {
            print_supported_profiles();
            return 0;
        }

        {
            struct stat model_stat;
            std::ifstream model_stream;

            if (::stat(inference.model_path.c_str(), &model_stat)
                    != 0
                || !S_ISREG(model_stat.st_mode)
                || model_stat.st_size == 0) {
                throw std::invalid_argument(
                    "model file missing, unreadable, or empty: "
                    + inference.model_path
                );
            }

            model_stream.open(
                inference.model_path, std::ios::binary
            );

            if (!model_stream.good()
                || model_stream.peek()
                    == std::char_traits<char>::eof()) {
                throw std::invalid_argument(
                    "model file missing, unreadable, or empty: "
                    + inference.model_path
                );
            }
        }

        std::string video_host;
        {
            std::string error;

            if (!resolve_numeric_host(
                    options.pi_host, video_host, error)) {
                throw std::runtime_error(
                    "Video destination unresolvable: " + error
                );
            }
        }

        std::vector<ResolvedAddress> redis_candidates;
        {
            std::string error;
            redis_candidates = resolve_tcp_addresses(
                options.redis_host,
                options.redis_port,
                error
            );

            if (redis_candidates.empty()) {
                std::cerr
                    << "Redis unavailable: " << error
                    << std::endl;
            }
        }

        const std::string encoder = select_encoder(options, stop);

        const CameraConfiguration camera_configuration =
            select_camera_configuration(inference);
        print_camera_configuration(camera_configuration);

        std::cout
            << "Loading YOLOv26 model: "
            << inference.model_path
            << std::endl;

        YoloRunner detector(inference.model_path);

        std::cout
            << "Model input: "
            << detector.input_width()
            << "x"
            << detector.input_height()
            << std::endl;

        std::cout
            << "Confidence threshold: "
            << inference.confidence_threshold
            << std::endl;

        std::cout
            << "Max detections: "
            << inference.max_detections
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

        auto profile = pipeline.start(config);

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
            void release() { active = false; }
        } camera_guard{pipeline};

        auto device = profile.get_device();

        auto depth_sensor =
            device.first<rs2::depth_sensor>();

        float depth_scale =
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
            << "Depth scale: "
            << depth_scale
            << " m/unit"
            << std::endl;

        std::cout
            << "Depth sampling: "
            << DEPTH_SAMPLE_GRID_SIZE
            << "x"
            << DEPTH_SAMPLE_GRID_SIZE
            << " calibrated grid"
            << std::endl;

        std::cout
            << "RTP destination: udp://"
            << options.pi_host
            << ":"
            << options.video_port
            << " (H.264 PT "
            << RTP_PAYLOAD_TYPE
            << ", MTU "
            << RTP_MTU
            << ")"
            << std::endl;

        std::cout
            << "Encoder: "
            << encoder
            << ", bitrate "
            << options.bitrate_kbps
            << " kbps"
            << std::endl;

        std::cout
            << "Redis: "
            << options.redis_host
            << ":"
            << options.redis_port
            << " channel "
            << options.redis_channel
            << " interval "
            << options.redis_interval_seconds
            << " s stale "
            << options.face_stale_seconds
            << " s"
            << std::endl;

        std::cout
            << "Warming up..."
            << std::endl;

        int warmed = 0;
        const auto warmup_deadline =
            Clock::now() + std::chrono::seconds(5);

        while (warmed < 10 && !stop.requested()) {
            if (Clock::now() >= warmup_deadline) {
                throw std::runtime_error(
                    "no usable camera frames during warmup"
                );
            }

            rs2::frameset frames;

            if (pipeline.try_wait_for_frames(&frames, 100)
                && frames.get_color_frame()
                && frames.get_depth_frame()) {
                ++warmed;
            }
        }

        if (stop.requested()) {
            return 0;
        }

        VideoSenderConfig video_config;
        video_config.host = video_host;
        video_config.port = options.video_port;
        video_config.width = camera_configuration.color.width;
        video_config.height = camera_configuration.color.height;
        video_config.fps = camera_configuration.color.fps;
        video_config.bitrate_kbps = options.bitrate_kbps;
        video_config.encoder = encoder;
        video_config.count_access_units = options.telemetry;

        GstVideoSender video(video_config);
        video.start();

        std::cout
            << "Streaming. Press Ctrl+C to stop."
            << std::endl;

        const Clock::time_point video_start = Clock::now();
        const uint64_t frame_duration_ns =
            1000000000ull
            / static_cast<uint64_t>(
                camera_configuration.color.fps
            );

        LatestFrameHandoff handoff;
        LatestFaceState face_state;
        std::atomic<bool> running{true};
        std::exception_ptr render_error;
        std::exception_ptr capture_error;
        SenderTelemetry telemetry(video_start);
        std::mutex telemetry_mutex;
        SenderTelemetrySnapshot last_snapshot;

        std::thread renderer;
        std::thread redis_worker;

        try {
            renderer = std::thread([&]() {
                try {
                    SystemMonitor system_monitor;

                    if (options.telemetry) {
                        system_monitor.update_if_due();
                    }

                    FramePacket packet;

                    while (running.load()) {
                        if (!handoff.wait_pop(packet)) {
                            break;
                        }

                        const auto overlay_start = Clock::now();
                        draw_detection_annotations(
                            packet.image,
                            packet.annotations,
                            inference
                        );

                        if (options.telemetry) {
                            system_monitor.update_if_due();

                            std::optional<SenderTelemetrySnapshot>
                                due;

                            {
                                std::lock_guard<std::mutex>
                                    lock(telemetry_mutex);
                                due = telemetry.snapshot_if_due(
                                    video.access_unit_count()
                                );

                                if (due) {
                                    last_snapshot = *due;
                                }
                            }

                            if (due) {
                                log_sender_telemetry(*due);
                            }

                            draw_sender_telemetry(
                                packet.image,
                                last_snapshot,
                                system_monitor.stats()
                            );
                        }

                        const auto push_start = Clock::now();
                        const double overlay_ms = elapsed_ms(
                            overlay_start,
                            push_start
                        );
                        const bool pushed = video.push(
                            packet.image.data,
                            packet.image.step1(),
                            packet.acquired_at,
                            frame_duration_ns
                        );
                        const auto push_end = Clock::now();

                        if (!pushed) {
                            if (running.load()
                                && !stop.requested()) {
                                throw std::runtime_error(
                                    "appsrc push failed"
                                );
                            }
                            break;
                        }

                        if (options.telemetry) {
                            const double push_ms = elapsed_ms(
                                push_start,
                                push_end
                            );
                            const double start_to_enqueue_ms =
                                elapsed_ms(
                                    packet.frame_start,
                                    push_end
                                );

                            std::lock_guard<std::mutex> lock(
                                telemetry_mutex
                            );
                            telemetry.record_submission(
                                overlay_ms,
                                push_ms,
                                start_to_enqueue_ms
                            );
                        }
                    }
                } catch (...) {
                    render_error = std::current_exception();
                }

                running.store(false);
            });

            redis_worker = std::thread([&]() {
                if (redis_candidates.empty()) {
                    return;
                }

                try {
                    RedisPublisher publisher;
                    PublisherParams params;
                    params.candidates = redis_candidates;
                    params.channel = options.redis_channel;
                    params.interval_seconds =
                        options.redis_interval_seconds;
                    params.stale_seconds =
                        options.face_stale_seconds;
                    params.timeout_seconds =
                        options.redis_timeout_ms / 1000.0;

                    PublisherHooks hooks;
                    hooks.connect =
                        [&](const std::vector<ResolvedAddress>& c,
                            double t,
                            const StopToken& s,
                            std::string& e) {
                            return publisher.connect_deadline(
                                c, t, s, e
                            );
                        };
                    hooks.publish =
                        [&](const std::string& ch,
                            const std::string& p,
                            double t,
                            const StopToken& s,
                            std::string& e) {
                            return publisher.publish(
                                ch, p, t, s, e
                            );
                        };
                    hooks.close =
                        [&]() { publisher.close(); };
                    hooks.connected =
                        [&]() { return publisher.connected(); };
                    hooks.log_error =
                        [](const std::string& m) {
                            std::cerr
                                << "Redis: " << m << std::endl;
                        };
                    hooks.log_info =
                        [](const std::string& m) {
                            std::cout << m << std::endl;
                        };

                    run_face_publisher(
                        params,
                        face_state,
                        stop,
                        hooks
                    );
                } catch (const std::exception& worker_failure) {
                    std::cerr
                        << "Redis worker failed: "
                        << worker_failure.what()
                        << std::endl;
                } catch (...) {
                    std::cerr
                        << "Redis worker failed: unknown error"
                        << std::endl;
                }
            });
        } catch (...) {
            capture_error = std::current_exception();
            running.store(false);
            g_workers_stop.store(true);
            handoff.stop();
        }

        uint64_t sequence = 0;
        Clock::time_point last_valid = Clock::now();

        try {
            while (running.load() && !stop.requested()) {
                if (options.duration_set
                    && std::chrono::duration<double>(
                        Clock::now() - video_start
                    ).count() >= options.duration_seconds) {
                    break;
                }

                std::string bus_error;

                if (video.poll_bus_error(bus_error)) {
                    throw std::runtime_error(bus_error);
                }

                const auto frame_start = Clock::now();
                rs2::frameset frames;

                if (!pipeline.try_wait_for_frames(&frames, 100)) {
                    if (std::chrono::duration<double>(
                            Clock::now() - last_valid
                        ).count() > 5.0) {
                        throw std::runtime_error(
                            "no frames from camera for 5 seconds"
                        );
                    }
                    continue;
                }

                const auto acquire_end = Clock::now();

                std::optional<FramePacket> packet = infer_frame(
                    frames,
                    camera_configuration,
                    detector,
                    depth_mapping,
                    inference,
                    sequence,
                    frame_start,
                    acquire_end
                );

                if (!packet) {
                    if (std::chrono::duration<double>(
                            Clock::now() - last_valid
                        ).count() > 5.0) {
                        throw std::runtime_error(
                            "no usable camera frames for 5 seconds"
                        );
                    }
                    continue;
                }

                ++sequence;
                last_valid = Clock::now();

                if (stop.requested()) {
                    break;
                }

                face_state.update(
                    face_detected(
                        packet->annotations,
                        packet->image.cols,
                        packet->image.rows
                    ),
                    packet->completed_at
                );

                if (options.telemetry) {
                    std::lock_guard<std::mutex> lock(
                        telemetry_mutex
                    );
                    telemetry.record_inference(
                        packet->acquire_ms,
                        packet->preprocess_ms,
                        packet->dpu_ms,
                        packet->depth_map_ms
                    );
                }

                const PublishResult result =
                    handoff.publish(std::move(*packet));

                if (result == PublishResult::replaced) {
                    if (options.telemetry) {
                        std::lock_guard<std::mutex> lock(
                            telemetry_mutex
                        );
                        telemetry.record_replacement();
                    }
                } else if (result == PublishResult::stopped) {
                    break;
                }
            }
        } catch (...) {
            capture_error = std::current_exception();
        }

        running.store(false);
        face_state.invalidate();
        handoff.stop();
        g_workers_stop.store(true);
        video.stop();

        if (renderer.joinable()) {
            renderer.join();
        }

        if (redis_worker.joinable()) {
            redis_worker.join();
        }

        std::cout
            << "Stopping camera..."
            << std::endl;

        pipeline.stop();
        camera_guard.release();

        std::cout
            << "Camera stopped"
            << std::endl;

        if (capture_error) {
            std::rethrow_exception(capture_error);
        }

        if (render_error) {
            std::rethrow_exception(render_error);
        }

        return 0;
    }

    catch (const rs2::error& e) {
        if (signal_fired.load()) {
            return 0;
        }

        std::cerr
            << "RealSense error: "
            << e.what()
            << std::endl;

        return 1;
    }

    catch (const std::exception& e) {
        if (signal_fired.load()
            && std::string(e.what()) == "cancelled") {
            return 0;
        }

        std::cerr
            << "Error: "
            << e.what()
            << std::endl;

        return 1;
    }
}
