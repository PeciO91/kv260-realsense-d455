#include <algorithm>
#include <array>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <exception>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <opencv2/opencv.hpp>

#include "yolov26_common.hpp"
#include "yolov26_realsense_core.hpp"

using namespace yolov26;


constexpr int DEFAULT_PORT = 8081;
constexpr int DEFAULT_JPEG_QUALITY = 90;
constexpr int TELEMETRY_MARGIN = 4;
constexpr int PERFORMANCE_PANEL_WIDTH = 185;
constexpr int PERFORMANCE_PANEL_HEIGHT = 86;
constexpr int SYSTEM_PANEL_WIDTH = 215;
constexpr int SYSTEM_PANEL_HEIGHT = 128;


struct Options : InferenceOptions {
    int port = DEFAULT_PORT;
    int jpeg_quality = DEFAULT_JPEG_QUALITY;
    bool telemetry = false;
    bool list_profiles = false;
};

struct PerformanceStats {
    bool available = false;
    double stream_fps = 0.0;
    double pipeline_fps = 0.0;
    double acquire_ms = 0.0;
    double preprocess_ms = 0.0;
    double dpu_latency_ms = 0.0;
    double dpu_fps = 0.0;
    double depth_map_ms = 0.0;
    double overlay_ms = 0.0;
    double jpeg_ms = 0.0;
    double send_ms = 0.0;
    double e2e_latency_ms = 0.0;
    uint64_t dropped_frames = 0;
    double drop_rate_percent = 0.0;
};


void print_usage(const char* program)
{
    std::cout
        << "Usage: "
        << program
        << " [--jpeg-quality N] [--port N]"
        << " [--depth-width N --depth-height N] [--list-profiles]"
        << " [--model PATH] [--confidence FLOAT] [--max-detections N]"
        << " [--show-depth-roi true|false]"
        << " [--show-confidence true|false]"
        << " [--telemetry true|false]"
        << std::endl
        << "Defaults: confidence "
        << DEFAULT_CONFIDENCE_THRESHOLD
        << ", max-detections "
        << DEFAULT_MAX_DETECTIONS
        << ", show-depth-roi false, show-confidence true,"
        << " telemetry false"
        << std::endl;
}

int parse_integer(
    const char* option_name,
    const char* value,
    int minimum,
    int maximum)
{
    try {
        std::size_t consumed = 0;
        const long parsed = std::stol(value, &consumed);

        if (consumed != std::strlen(value)
            || parsed < minimum
            || parsed > maximum) {
            throw std::invalid_argument("range");
        }

        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            std::string("Invalid value for ")
            + option_name
            + ": "
            + value
        );
    }
}

float parse_float(
    const char* option_name,
    const char* value,
    float minimum,
    float maximum)
{
    try {
        std::size_t consumed = 0;
        const float parsed = std::stof(value, &consumed);

        if (consumed != std::strlen(value)
            || !std::isfinite(parsed)
            || parsed < minimum
            || parsed > maximum) {
            throw std::invalid_argument("range");
        }

        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(
            std::string("Invalid value for ")
            + option_name
            + ": "
            + value
        );
    }
}

bool parse_bool(
    const char* option_name,
    const char* value)
{
    if (std::strcmp(value, "true") == 0) {
        return true;
    }

    if (std::strcmp(value, "false") == 0) {
        return false;
    }

    throw std::invalid_argument(
        std::string("Invalid value for ")
        + option_name
        + ": "
        + value
    );
}

Options parse_options(int argc, char* argv[])
{
    Options options;
    constexpr int HELP_OPTION = 1000;
    constexpr int DEPTH_WIDTH_OPTION = 1001;
    constexpr int DEPTH_HEIGHT_OPTION = 1002;
    constexpr int MODEL_OPTION = 1003;
    constexpr int CONFIDENCE_OPTION = 1004;
    constexpr int MAX_DETECTIONS_OPTION = 1005;
    constexpr int SHOW_DEPTH_ROI_OPTION = 1006;
    constexpr int SHOW_CONFIDENCE_OPTION = 1007;
    constexpr int TELEMETRY_OPTION = 1008;
    const option long_options[] = {
        {"jpeg-quality", required_argument, nullptr, 'q'},
        {"port", required_argument, nullptr, 'p'},
        {"depth-width", required_argument, nullptr, DEPTH_WIDTH_OPTION},
        {"depth-height", required_argument, nullptr, DEPTH_HEIGHT_OPTION},
        {"list-profiles", no_argument, nullptr, 'l'},
        {"model", required_argument, nullptr, MODEL_OPTION},
        {"confidence", required_argument, nullptr, CONFIDENCE_OPTION},
        {"max-detections", required_argument, nullptr,
            MAX_DETECTIONS_OPTION},
        {"show-depth-roi", required_argument, nullptr,
            SHOW_DEPTH_ROI_OPTION},
        {"show-confidence", required_argument, nullptr,
            SHOW_CONFIDENCE_OPTION},
        {"telemetry", required_argument, nullptr,
            TELEMETRY_OPTION},
        {"help", no_argument, nullptr, HELP_OPTION},
        {nullptr, 0, nullptr, 0}
    };

    while (true) {
        const int selected = getopt_long(
            argc,
            argv,
            "q:p:l",
            long_options,
            nullptr
        );

        if (selected == -1) {
            break;
        }

        switch (selected) {
        case 'q':
            options.jpeg_quality = parse_integer(
                "--jpeg-quality",
                optarg,
                1,
                100
            );
            break;
        case 'p':
            options.port = parse_integer(
                "--port",
                optarg,
                1,
                65535
            );
            break;
        case DEPTH_WIDTH_OPTION:
            options.depth_width = parse_integer(
                "--depth-width",
                optarg,
                1,
                4096
            );
            break;
        case DEPTH_HEIGHT_OPTION:
            options.depth_height = parse_integer(
                "--depth-height",
                optarg,
                1,
                4096
            );
            break;
        case 'l':
            options.list_profiles = true;
            break;
        case MODEL_OPTION:
            options.model_path = optarg;
            break;
        case CONFIDENCE_OPTION:
            options.confidence_threshold = parse_float(
                "--confidence",
                optarg,
                0.0f,
                1.0f
            );
            break;
        case MAX_DETECTIONS_OPTION:
            options.max_detections = parse_integer(
                "--max-detections",
                optarg,
                1,
                std::numeric_limits<int>::max()
            );
            break;
        case SHOW_DEPTH_ROI_OPTION:
            options.show_depth_roi = parse_bool(
                "--show-depth-roi",
                optarg
            );
            break;
        case SHOW_CONFIDENCE_OPTION:
            options.show_confidence = parse_bool(
                "--show-confidence",
                optarg
            );
            break;
        case TELEMETRY_OPTION:
            options.telemetry = parse_bool(
                "--telemetry",
                optarg
            );
            break;
        case HELP_OPTION:
            print_usage(argv[0]);
            std::exit(0);
        default:
            print_usage(argv[0]);
            throw std::invalid_argument("Invalid command-line arguments");
        }
    }

    if (optind != argc) {
        print_usage(argv[0]);
        throw std::invalid_argument(
            std::string("Unexpected argument: ") + argv[optind]
        );
    }

    if ((options.depth_width == 0) != (options.depth_height == 0)) {
        throw std::invalid_argument(
            "--depth-width and --depth-height must be used together"
        );
    }

    return options;
}

class PerformanceMonitor {
public:
    PerformanceMonitor()
        : window_start_(Clock::now())
    {
    }

    void record_pipeline(
        double acquire_ms,
        double preprocess_ms,
        double dpu_ms,
        double depth_map_ms)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pipeline_frames_++;
        acquire_seconds_ += acquire_ms / 1000.0;
        acquire_samples_++;
        preprocess_seconds_ += preprocess_ms / 1000.0;
        preprocess_samples_++;
        dpu_seconds_ += dpu_ms / 1000.0;
        dpu_samples_++;
        depth_map_seconds_ += depth_map_ms / 1000.0;
        depth_map_samples_++;
    }

    void record_drop()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dropped_frames_++;
        window_drops_++;
    }

    bool record_stream(
        double overlay_ms,
        double jpeg_ms,
        double send_ms,
        double e2e_latency_ms,
        PerformanceStats& snapshot)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = Clock::now();
        stream_frames_++;
        overlay_seconds_ += overlay_ms / 1000.0;
        overlay_samples_++;
        jpeg_seconds_ += jpeg_ms / 1000.0;
        jpeg_samples_++;
        send_seconds_ += send_ms / 1000.0;
        send_samples_++;
        e2e_seconds_ += e2e_latency_ms / 1000.0;
        e2e_samples_++;

        const double elapsed =
            std::chrono::duration<double>(
                now - window_start_
            ).count();

        if (elapsed < TELEMETRY_INTERVAL_SECONDS) {
            return false;
        }

        stats_.available = true;
        stats_.stream_fps = stream_frames_ / elapsed;
        stats_.pipeline_fps = pipeline_frames_ / elapsed;
        stats_.acquire_ms = average_ms(
            acquire_seconds_,
            acquire_samples_
        );
        stats_.preprocess_ms = average_ms(
            preprocess_seconds_,
            preprocess_samples_
        );
        stats_.dpu_latency_ms = average_ms(
            dpu_seconds_,
            dpu_samples_
        );
        stats_.dpu_fps = stats_.dpu_latency_ms > 0.0
            ? 1000.0 / stats_.dpu_latency_ms
            : 0.0;
        stats_.depth_map_ms = average_ms(
            depth_map_seconds_,
            depth_map_samples_
        );
        stats_.overlay_ms = average_ms(
            overlay_seconds_,
            overlay_samples_
        );
        stats_.jpeg_ms = average_ms(
            jpeg_seconds_,
            jpeg_samples_
        );
        stats_.send_ms = average_ms(
            send_seconds_,
            send_samples_
        );
        stats_.e2e_latency_ms = average_ms(
            e2e_seconds_,
            e2e_samples_
        );
        stats_.dropped_frames = dropped_frames_;
        stats_.drop_rate_percent = pipeline_frames_ > 0
            ? 100.0 * window_drops_ / pipeline_frames_
            : 0.0;

        snapshot = stats_;
        reset_window(now);
        return true;
    }

    PerformanceStats stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    static double average_ms(
        double total_seconds,
        uint64_t samples)
    {
        return samples > 0
            ? total_seconds * 1000.0 / samples
            : 0.0;
    }

    void reset_window(Clock::time_point now)
    {
        window_start_ = now;
        stream_frames_ = 0;
        pipeline_frames_ = 0;
        window_drops_ = 0;
        acquire_samples_ = 0;
        preprocess_samples_ = 0;
        dpu_samples_ = 0;
        depth_map_samples_ = 0;
        overlay_samples_ = 0;
        jpeg_samples_ = 0;
        send_samples_ = 0;
        e2e_samples_ = 0;
        acquire_seconds_ = 0.0;
        preprocess_seconds_ = 0.0;
        dpu_seconds_ = 0.0;
        depth_map_seconds_ = 0.0;
        overlay_seconds_ = 0.0;
        jpeg_seconds_ = 0.0;
        send_seconds_ = 0.0;
        e2e_seconds_ = 0.0;
    }

    mutable std::mutex mutex_;
    Clock::time_point window_start_;
    PerformanceStats stats_;
    uint64_t stream_frames_ = 0;
    uint64_t pipeline_frames_ = 0;
    uint64_t window_drops_ = 0;
    uint64_t dropped_frames_ = 0;
    uint64_t acquire_samples_ = 0;
    uint64_t preprocess_samples_ = 0;
    uint64_t dpu_samples_ = 0;
    uint64_t depth_map_samples_ = 0;
    uint64_t overlay_samples_ = 0;
    uint64_t jpeg_samples_ = 0;
    uint64_t send_samples_ = 0;
    uint64_t e2e_samples_ = 0;
    double acquire_seconds_ = 0.0;
    double preprocess_seconds_ = 0.0;
    double dpu_seconds_ = 0.0;
    double depth_map_seconds_ = 0.0;
    double overlay_seconds_ = 0.0;
    double jpeg_seconds_ = 0.0;
    double send_seconds_ = 0.0;
    double e2e_seconds_ = 0.0;
};


void draw_telemetry(
    cv::Mat& image,
    const PerformanceStats& performance,
    const SystemStats& system)
{
    const cv::Rect performance_panel(
        TELEMETRY_MARGIN,
        TELEMETRY_MARGIN,
        PERFORMANCE_PANEL_WIDTH,
        PERFORMANCE_PANEL_HEIGHT
    );
    const cv::Rect system_panel(
        image.cols - TELEMETRY_MARGIN - SYSTEM_PANEL_WIDTH,
        TELEMETRY_MARGIN,
        SYSTEM_PANEL_WIDTH,
        SYSTEM_PANEL_HEIGHT
    );

    if (performance_panel.x < 0
        || performance_panel.y < 0
        || performance_panel.width <= 0
        || performance_panel.height <= 0
        || system_panel.x < 0
        || system_panel.y < 0
        || system_panel.width <= 0
        || system_panel.height <= 0
        || performance_panel.x + performance_panel.width > image.cols
        || performance_panel.y + performance_panel.height > image.rows
        || system_panel.x + system_panel.width > image.cols
        || system_panel.y + system_panel.height > image.rows) {
        return;
    }

    static const cv::Mat performance_background(
        PERFORMANCE_PANEL_HEIGHT,
        PERFORMANCE_PANEL_WIDTH,
        CV_8UC3,
        cv::Scalar(28, 28, 28)
    );
    static const cv::Mat system_background(
        SYSTEM_PANEL_HEIGHT,
        SYSTEM_PANEL_WIDTH,
        CV_8UC3,
        cv::Scalar(28, 28, 28)
    );

    cv::Mat performance_roi = image(performance_panel);
    cv::addWeighted(
        performance_roi,
        0.32,
        performance_background,
        0.68,
        0.0,
        performance_roi
    );

    cv::Mat system_roi = image(system_panel);
    cv::addWeighted(
        system_roi,
        0.32,
        system_background,
        0.68,
        0.0,
        system_roi
    );

    const int font = cv::FONT_HERSHEY_SIMPLEX;
    char text[128];

    int performance_y = performance_panel.y + 13;

    const auto draw_performance_heading = [&](const char* value) {
        cv::putText(
            image,
            value,
            cv::Point(
                performance_panel.x + 7,
                performance_y
            ),
            font,
            0.36,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA
        );
        performance_y += 15;
    };

    const auto draw_performance_line = [&](const char* value) {
        cv::putText(
            image,
            value,
            cv::Point(
                performance_panel.x + 7,
                performance_y
            ),
            font,
            0.30,
            cv::Scalar(245, 245, 245),
            1,
            cv::LINE_AA
        );
        performance_y += 13;
    };

    int system_y = system_panel.y + 13;

    const auto draw_system_heading = [&](const char* value) {
        cv::putText(
            image,
            value,
            cv::Point(
                system_panel.x + 7,
                system_y
            ),
            font,
            0.36,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA
        );
        system_y += 15;
    };

    const auto draw_system_line = [&](const char* value) {
        cv::putText(
            image,
            value,
            cv::Point(
                system_panel.x + 7,
                system_y
            ),
            font,
            0.30,
            cv::Scalar(245, 245, 245),
            1,
            cv::LINE_AA
        );
        system_y += 13;
    };

    draw_performance_heading("Performance");

    if (performance.available) {
        std::snprintf(
            text,
            sizeof(text),
            "Stream FPS: %.1f",
            performance.stream_fps
        );
        draw_performance_line(text);
        std::snprintf(
            text,
            sizeof(text),
            "Pipeline FPS: %.1f",
            performance.pipeline_fps
        );
        draw_performance_line(text);
        std::snprintf(
            text,
            sizeof(text),
            "DPU latency: %.1f ms",
            performance.dpu_latency_ms
        );
        draw_performance_line(text);
        std::snprintf(
            text,
            sizeof(text),
            "DPU rate: %.1f FPS",
            performance.dpu_fps
        );
        draw_performance_line(text);
    } else {
        draw_performance_line("Stream FPS: N/A");
        draw_performance_line("Pipeline FPS: N/A");
        draw_performance_line("DPU latency: N/A");
        draw_performance_line("DPU rate: N/A");
    }

    draw_system_heading("System Info");

    if (system.cpu_available) {
        std::snprintf(
            text,
            sizeof(text),
            "CPU: %.1f%%",
            system.cpu_percent
        );
        draw_system_line(text);
    } else {
        draw_system_line("CPU: N/A");
    }

    if (system.ram_available) {
        std::snprintf(
            text,
            sizeof(text),
            "RAM: %llu/%llu MB (%.1f%%)",
            static_cast<unsigned long long>(system.ram_used_mb),
            static_cast<unsigned long long>(system.ram_total_mb),
            system.ram_percent
        );
        draw_system_line(text);
    } else {
        draw_system_line("RAM: N/A");
    }

    if (system.temperature_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Temp: %.1f C",
            system.temperature_c
        );
        draw_system_line(text);
    } else {
        draw_system_line("Temp: N/A");
    }

    if (system.pl_temperature_available) {
        std::snprintf(
            text,
            sizeof(text),
            "PL Temp: %.1f C",
            system.pl_temperature_c
        );
        draw_system_line(text);
    } else {
        draw_system_line("PL Temp: N/A");
    }

    if (system.power_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Power: %.2f W",
            system.power_w
        );
        draw_system_line(text);
    } else {
        draw_system_line("Power: N/A");
    }

    if (system.voltage_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Voltage: %.2f V",
            system.voltage_v
        );
        draw_system_line(text);
    } else {
        draw_system_line("Voltage: N/A");
    }

    if (system.current_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Current: %.2f A",
            system.current_a
        );
        draw_system_line(text);
    } else {
        draw_system_line("Current: N/A");
    }
}

void log_telemetry(
    const PerformanceStats& performance,
    const SystemStats& system)
{
    std::ostringstream output;
    output
        << std::fixed
        << std::setprecision(1)
        << "Performance: stream="
        << performance.stream_fps
        << " FPS pipeline="
        << performance.pipeline_fps
        << " FPS acquire="
        << performance.acquire_ms
        << " ms preprocess="
        << performance.preprocess_ms
        << " ms dpu="
        << performance.dpu_latency_ms
        << " ms dpu_rate="
        << performance.dpu_fps
        << " FPS depth_postprocess="
        << performance.depth_map_ms
        << " ms overlay="
        << performance.overlay_ms
        << " ms jpeg="
        << performance.jpeg_ms
        << " ms send="
        << performance.send_ms
        << " ms e2e_latency="
        << performance.e2e_latency_ms
        << " ms dropped="
        << performance.dropped_frames
        << " drop_rate="
        << performance.drop_rate_percent
        << "% | CPU=";

    if (system.cpu_available) {
        output << system.cpu_percent << "%";
    } else {
        output << "N/A";
    }

    output << " RAM=";

    if (system.ram_available) {
        output
            << system.ram_used_mb
            << "/"
            << system.ram_total_mb
            << " MB ("
            << system.ram_percent
            << "%)";
    } else {
        output << "N/A";
    }

    output << " Temp=";

    if (system.temperature_available) {
        output << system.temperature_c << " C";
    } else {
        output << "N/A";
    }

    output << " PL=";

    if (system.pl_temperature_available) {
        output << system.pl_temperature_c << " C";
    } else {
        output << "N/A";
    }

    output << " Power=";

    if (system.power_available) {
        output << system.power_w << " W";
    } else {
        output << "N/A";
    }

    output << " Voltage=";

    if (system.voltage_available) {
        output << system.voltage_v << " V";
    } else {
        output << "N/A";
    }

    output << " Current=";

    if (system.current_available) {
        output << system.current_a << " A";
    } else {
        output << "N/A";
    }

    std::cout << output.str() << std::endl;
}




int create_server(int port)
{
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);

    if (server_fd < 0) {
        throw std::runtime_error("socket() failed");
    }

    int opt = 1;

    setsockopt(
        server_fd,
        SOL_SOCKET,
        SO_REUSEADDR,
        &opt,
        sizeof(opt)
    );

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(
            server_fd,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0) {
        close(server_fd);
        throw std::runtime_error("bind() failed");
    }

    if (listen(server_fd, 4) < 0) {
        close(server_fd);
        throw std::runtime_error("listen() failed");
    }

    return server_fd;
}


bool send_all(
    int fd,
    const void* data,
    size_t size)
{
    const char* ptr =
        reinterpret_cast<const char*>(data);

    while (size > 0) {
        ssize_t sent = send(
            fd,
            ptr,
            size,
            MSG_NOSIGNAL
        );

        if (sent <= 0) {
            return false;
        }

        ptr += sent;
        size -= sent;
    }

    return true;
}

bool send_html_page(int client_fd)
{
    static const std::string page =
        "<!doctype html>"
        "<html><head>"
        "<meta charset=\"utf-8\">"
        "<meta name=\"viewport\" "
        "content=\"width=device-width,initial-scale=1\">"
        "<title>KV260 YOLOv26 Stream</title>"
        "<style>"
        "html,body{width:100%;height:100%;margin:0;background:#000;"
        "overflow:hidden;}"
        "body{display:flex;align-items:center;justify-content:center;}"
        "img{width:100vw;height:100vh;object-fit:contain;display:block;}"
        "</style></head>"
        "<body><img src=\"/stream.mjpg\" "
        "alt=\"KV260 YOLOv26 camera stream\"></body></html>";

    const std::string header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "Content-Length: "
        + std::to_string(page.size())
        + "\r\n"
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n\r\n";

    return send_all(
        client_fd,
        header.data(),
        header.size()
    ) && send_all(
        client_fd,
        page.data(),
        page.size()
    );
}

int wait_for_stream_client(int server_fd)
{
    while (true) {
        sockaddr_in client_address{};
        socklen_t client_len = sizeof(client_address);
        const int client_fd = accept(
            server_fd,
            reinterpret_cast<sockaddr*>(&client_address),
            &client_len
        );

        if (client_fd < 0) {
            throw std::runtime_error("accept() failed");
        }

        char request_buffer[2048];
        const ssize_t request_size = recv(
            client_fd,
            request_buffer,
            sizeof(request_buffer),
            0
        );

        if (request_size <= 0) {
            close(client_fd);
            continue;
        }

        const std::string request(
            request_buffer,
            static_cast<std::size_t>(request_size)
        );

        if (request.compare(0, 17, "GET /stream.mjpg ") == 0) {
            return client_fd;
        }

        if (request.compare(0, 6, "GET / ") == 0
            || request.compare(0, 16, "GET /index.html ") == 0) {
            send_html_page(client_fd);
            close(client_fd);
            continue;
        }

        static const std::string not_found =
            "HTTP/1.1 404 Not Found\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n\r\n";
        send_all(
            client_fd,
            not_found.data(),
            not_found.size()
        );
        close(client_fd);
    }
}


int main(int argc, char* argv[])
{
    try {
        const Options options = parse_options(argc, argv);

        if (options.list_profiles) {
            print_supported_profiles();
            return 0;
        }

        const CameraConfiguration camera_configuration =
            select_camera_configuration(options);
        print_camera_configuration(camera_configuration);

        std::cout
            << "Loading YOLOv26 model: "
            << options.model_path
            << std::endl;

        YoloRunner detector(options.model_path);

        std::cout
            << "Model input: "
            << detector.input_width()
            << "x"
            << detector.input_height()
            << std::endl;

        std::cout
            << "Confidence threshold: "
            << options.confidence_threshold
            << std::endl;

        std::cout
            << "Max detections: "
            << options.max_detections
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
            << device.get_info(
                RS2_CAMERA_INFO_NAME
            )
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
            << "JPEG quality: "
            << options.jpeg_quality
            << std::endl;

        std::cout
            << "Warming up..."
            << std::endl;

        for (int i = 0; i < 10; ++i) {
            pipeline.wait_for_frames();
        }

        int server_fd = create_server(options.port);

        std::cout << std::endl;
        std::cout
            << "Open on PC:"
            << std::endl;

        std::cout
            << "http://147.32.163.22:"
            << options.port
            << "/"
            << std::endl;

        std::cout
            << "Waiting for browser..."
            << std::endl;

        int client_fd = wait_for_stream_client(server_fd);

        const std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Cache-Control: no-cache\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n"
            "Content-Type: "
            "multipart/x-mixed-replace; "
            "boundary=frame\r\n\r\n";

        if (!send_all(
                client_fd,
                header.data(),
                header.size())) {
            throw std::runtime_error(
                "Failed to send HTTP header"
            );
        }

        std::cout
            << "Client connected. Streaming..."
            << std::endl;

        PerformanceMonitor performance_monitor;
        LatestFrameHandoff handoff;
        std::atomic<bool> running{true};
        std::exception_ptr render_error;
        std::exception_ptr capture_error;
        std::thread renderer;

        try {
            renderer = std::thread([&]() {
                try {
                    SystemMonitor system_monitor;

                    if (options.telemetry) {
                        system_monitor.update_if_due();
                    }
                    std::vector<uchar> jpeg;
                    jpeg.reserve(
                        camera_configuration.color.width
                        * camera_configuration.color.height
                        / 4
                    );
                    const std::vector<int> jpeg_params = {
                        cv::IMWRITE_JPEG_QUALITY,
                        options.jpeg_quality
                    };
                    FramePacket packet;

                    while (running.load()) {
                        if (!handoff.wait_pop(packet)) {
                            break;
                        }

                        const auto overlay_start = Clock::now();
                        draw_detection_annotations(
                            packet.image,
                            packet.annotations,
                            options
                        );
                        if (options.telemetry) {
                            system_monitor.update_if_due();
                            const PerformanceStats displayed =
                                performance_monitor.stats();
                            draw_telemetry(
                                packet.image,
                                displayed,
                                system_monitor.stats()
                            );
                        }
                        const double overlay_ms = elapsed_ms(
                            overlay_start,
                            Clock::now()
                        );

                        jpeg.clear();
                        const auto jpeg_start = Clock::now();

                        const bool encoded = cv::imencode(
                            ".jpg",
                            packet.image,
                            jpeg,
                            jpeg_params
                        );
                        const double jpeg_ms = elapsed_ms(
                            jpeg_start,
                            Clock::now()
                        );

                        if (!encoded) {
                            continue;
                        }

                        std::string frame_header =
                            "--frame\r\n"
                            "Content-Type: image/jpeg\r\n"
                            "Content-Length: "
                            + std::to_string(
                                jpeg.size()
                            )
                            + "\r\n\r\n";

                        const auto send_start = Clock::now();
                        const char* ending = "\r\n";
                        const bool sent =
                            send_all(
                                client_fd,
                                frame_header.data(),
                                frame_header.size())
                            && send_all(
                                client_fd,
                                jpeg.data(),
                                jpeg.size())
                            && send_all(
                                client_fd,
                                ending,
                                2);
                        const double send_ms = elapsed_ms(
                            send_start,
                            Clock::now()
                        );

                        if (!sent) {
                            break;
                        }

                        const double e2e_ms = elapsed_ms(
                            packet.frame_start,
                            Clock::now()
                        );
                        PerformanceStats snapshot;

                        if (options.telemetry
                            && performance_monitor.record_stream(
                                overlay_ms,
                                jpeg_ms,
                                send_ms,
                                e2e_ms,
                                snapshot)) {
                            log_telemetry(
                                snapshot,
                                system_monitor.stats()
                            );
                        }
                    }
                } catch (...) {
                    render_error = std::current_exception();
                }

                running.store(false);
                handoff.stop();
            });
        } catch (...) {
            capture_error = std::current_exception();
        }

        if (renderer.joinable()) {
            uint64_t sequence = 0;

            try {
                while (running.load()) {
                    const auto frame_start = Clock::now();

                    auto frames =
                        pipeline.wait_for_frames();

                    const auto acquire_end = Clock::now();

                    std::optional<FramePacket> inferred =
                        infer_frame(
                            frames,
                            camera_configuration,
                            detector,
                            depth_mapping,
                            options,
                            sequence,
                            frame_start,
                            acquire_end
                        );

                    if (!inferred) {
                        continue;
                    }

                    ++sequence;
                    FramePacket packet = std::move(*inferred);

                    if (options.telemetry) {
                        performance_monitor.record_pipeline(
                            packet.acquire_ms,
                            packet.preprocess_ms,
                            packet.dpu_ms,
                            packet.depth_map_ms
                        );
                    }

                    const PublishResult result =
                        handoff.publish(std::move(packet));

                    if (result == PublishResult::replaced) {
                        if (options.telemetry) {
                            performance_monitor.record_drop();
                        }
                    } else if (result == PublishResult::stopped) {
                        break;
                    }
                }
            } catch (...) {
                capture_error = std::current_exception();
            }
        }

        running.store(false);
        handoff.stop();

        if (capture_error) {
            shutdown(client_fd, SHUT_RDWR);
        }

        if (renderer.joinable()) {
            renderer.join();
        }

        if (!capture_error && !render_error) {
            std::cout
                << "Client disconnected."
                << std::endl;
        }

        close(client_fd);
        close(server_fd);

        std::cout
            << "Stopping camera..."
            << std::endl;

        pipeline.stop();

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
        std::cerr
            << "RealSense error: "
            << e.what()
            << std::endl;

        return 1;
    }

    catch (const std::exception& e) {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;

        return 1;
    }
}
