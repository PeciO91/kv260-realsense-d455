#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vitis/ai/facedetect.hpp>


constexpr int DEFAULT_PORT = 8081;
constexpr int DEFAULT_JPEG_QUALITY = 90;
constexpr int DEFAULT_COLOR_WIDTH = 848;
constexpr int DEFAULT_COLOR_HEIGHT = 480;
constexpr int DEFAULT_COLOR_FPS = 30;
constexpr float DEPTH_ROI_SCALE = 0.35f;
constexpr double TELEMETRY_INTERVAL_SECONDS = 1.0;
constexpr int TELEMETRY_MARGIN = 4;
constexpr int PERFORMANCE_PANEL_WIDTH = 185;
constexpr int PERFORMANCE_PANEL_HEIGHT = 86;
constexpr int SYSTEM_PANEL_WIDTH = 215;
constexpr int SYSTEM_PANEL_HEIGHT = 128;

using Clock = std::chrono::steady_clock;

struct Options {
    int port = DEFAULT_PORT;
    int jpeg_quality = DEFAULT_JPEG_QUALITY;
    int depth_width = 0;
    int depth_height = 0;
    bool list_profiles = false;
};

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

struct PerformanceStats {
    bool available = false;
    double stream_fps = 0.0;
    double pipeline_fps = 0.0;
    double acquire_ms = 0.0;
    double align_ms = 0.0;
    double preprocess_ms = 0.0;
    double dpu_latency_ms = 0.0;
    double dpu_fps = 0.0;
    double depth_ms = 0.0;
    double overlay_ms = 0.0;
    double jpeg_ms = 0.0;
    double send_ms = 0.0;
    double total_ms = 0.0;
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

void print_usage(const char* program)
{
    std::cout
        << "Usage: "
        << program
        << " [--jpeg-quality N] [--port N]"
        << " [--depth-width N --depth-height N] [--list-profiles]"
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

Options parse_options(int argc, char* argv[])
{
    Options options;
    constexpr int HELP_OPTION = 1000;
    constexpr int DEPTH_WIDTH_OPTION = 1001;
    constexpr int DEPTH_HEIGHT_OPTION = 1002;
    const option long_options[] = {
        {"jpeg-quality", required_argument, nullptr, 'q'},
        {"port", required_argument, nullptr, 'p'},
        {"depth-width", required_argument, nullptr, DEPTH_WIDTH_OPTION},
        {"depth-height", required_argument, nullptr, DEPTH_HEIGHT_OPTION},
        {"list-profiles", no_argument, nullptr, 'l'},
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

bool is_native_color_format(rs2_format format);

void print_supported_profiles()
{
    rs2::context context;
    const rs2::device_list devices = context.query_devices();

    if (devices.size() == 0) {
        throw std::runtime_error("No RealSense device found");
    }

    for (const rs2::device& device : devices) {
        std::cout
            << "Device: "
            << device.get_info(RS2_CAMERA_INFO_NAME)
            << std::endl;

        for (const rs2::sensor& sensor : device.query_sensors()) {
            for (const rs2::stream_profile& profile
                 : sensor.get_stream_profiles()) {
                const rs2_stream stream = profile.stream_type();

                if (stream != RS2_STREAM_COLOR
                    && stream != RS2_STREAM_DEPTH) {
                    continue;
                }

                if ((stream == RS2_STREAM_COLOR
                     && !is_native_color_format(profile.format()))
                    || (stream == RS2_STREAM_DEPTH
                        && profile.format() != RS2_FORMAT_Z16)) {
                    continue;
                }

                const auto video =
                    profile.as<rs2::video_stream_profile>();

                if (!video) {
                    continue;
                }

                std::cout
                    << (stream == RS2_STREAM_COLOR
                        ? "Color "
                        : "Depth ")
                    << video.width()
                    << "x"
                    << video.height()
                    << " @ "
                    << profile.fps()
                    << " FPS "
                    << rs2_format_to_string(profile.format())
                    << std::endl;
            }
        }
    }
}

bool is_native_color_format(rs2_format format)
{
    return format == RS2_FORMAT_YUYV
        || format == RS2_FORMAT_UYVY;
}

bool is_lower_resolution_profile(
    const CameraStreamConfiguration& candidate,
    const CameraStreamConfiguration& current)
{
    const int64_t candidate_pixels =
        static_cast<int64_t>(candidate.width) * candidate.height;
    const int64_t current_pixels =
        static_cast<int64_t>(current.width) * current.height;

    if (candidate_pixels != current_pixels) {
        return candidate_pixels < current_pixels;
    }

    if (candidate.width != current.width) {
        return candidate.width < current.width;
    }

    return candidate.height < current.height;
}

CameraConfiguration select_camera_configuration(const Options& options)
{
    rs2::context context;
    const rs2::device_list devices = context.query_devices();

    if (devices.size() == 0) {
        throw std::runtime_error("No RealSense device found");
    }

    const rs2::device device = devices[0];
    CameraConfiguration selected;
    bool color_found = false;

    for (const rs2::sensor& sensor : device.query_sensors()) {
        for (const rs2::stream_profile& profile
             : sensor.get_stream_profiles()) {
            if (profile.stream_type() != RS2_STREAM_COLOR
                || !is_native_color_format(profile.format())) {
                continue;
            }

            const auto video =
                profile.as<rs2::video_stream_profile>();

            if (!video) {
                continue;
            }

            const CameraStreamConfiguration candidate = {
                video.width(),
                video.height(),
                profile.fps(),
                profile.format()
            };

            if (candidate.width != DEFAULT_COLOR_WIDTH
                || candidate.height != DEFAULT_COLOR_HEIGHT
                || candidate.fps != DEFAULT_COLOR_FPS) {
                continue;
            }

            selected.color = candidate;
            color_found = true;
        }
    }

    if (!color_found) {
        throw std::runtime_error(
            "Requested native RealSense color profile is unavailable"
        );
    }

    bool depth_found = false;

    for (const rs2::sensor& sensor : device.query_sensors()) {
        for (const rs2::stream_profile& profile
             : sensor.get_stream_profiles()) {
            if (profile.stream_type() != RS2_STREAM_DEPTH
                || profile.format() != RS2_FORMAT_Z16
                || profile.fps() != selected.color.fps) {
                continue;
            }

            const auto video =
                profile.as<rs2::video_stream_profile>();

            if (!video) {
                continue;
            }

            if (options.depth_width != 0
                && (video.width() != options.depth_width
                    || video.height() != options.depth_height)) {
                continue;
            }

            const CameraStreamConfiguration candidate = {
                video.width(),
                video.height(),
                profile.fps(),
                profile.format()
            };

            if (!depth_found
                || (options.depth_width == 0
                    && is_lower_resolution_profile(
                        candidate,
                        selected.depth
                    ))) {
                selected.depth = candidate;
                depth_found = true;
            }
        }
    }

    if (!depth_found) {
        throw std::runtime_error(
            "No Z16 depth profile matches the selected color FPS"
        );
    }

    return selected;
}

void print_camera_configuration(
    const CameraConfiguration& configuration)
{
    std::cout
        << "Selected D455 configuration:"
        << std::endl
        << "RGB:   "
        << configuration.color.width
        << "x"
        << configuration.color.height
        << " @ "
        << configuration.color.fps
        << " FPS (native "
        << rs2_format_to_string(configuration.color.native_format)
        << ", OpenCV BGR8)"
        << std::endl
        << "Depth: "
        << configuration.depth.width
        << "x"
        << configuration.depth.height
        << " @ "
        << configuration.depth.fps
        << " FPS "
        << rs2_format_to_string(configuration.depth.native_format)
        << std::endl;
}

bool read_text_file(
    const std::string& path,
    std::string& value)
{
    std::ifstream input(path);

    if (!input) {
        return false;
    }

    return static_cast<bool>(std::getline(input, value));
}

bool read_number_file(
    const std::string& path,
    double& value)
{
    std::ifstream input(path);
    return static_cast<bool>(input >> value);
}

class PerformanceMonitor {
public:
    PerformanceMonitor()
        : window_start_(Clock::now()),
          frame_start_(window_start_),
          stage_start_(window_start_)
    {
    }

    void begin_frame()
    {
        frame_start_ = Clock::now();
        stage_start_ = frame_start_;
    }

    void mark_acquired()
    {
        add_stage(acquire_seconds_, acquire_samples_);
    }

    void mark_aligned()
    {
        add_stage(align_seconds_, align_samples_);
    }

    void mark_preprocessed()
    {
        add_stage(preprocess_seconds_, preprocess_samples_);
    }

    void mark_dpu_complete(
        Clock::time_point start,
        Clock::time_point end)
    {
        dpu_seconds_ +=
            std::chrono::duration<double>(end - start).count();
        dpu_samples_++;
        stage_start_ = end;
    }

    void mark_depth_complete()
    {
        add_stage(depth_seconds_, depth_samples_);
    }

    void mark_overlay_complete()
    {
        add_stage(overlay_seconds_, overlay_samples_);
        pipeline_frames_++;
    }

    void begin_jpeg()
    {
        stage_start_ = Clock::now();
    }

    void mark_jpeg_complete()
    {
        add_stage(jpeg_seconds_, jpeg_samples_);
    }

    void begin_send()
    {
        stage_start_ = Clock::now();
    }

    bool mark_frame_sent()
    {
        const auto now = Clock::now();
        send_seconds_ +=
            std::chrono::duration<double>(
                now - stage_start_
            ).count();
        send_samples_++;
        total_seconds_ +=
            std::chrono::duration<double>(
                now - frame_start_
            ).count();
        total_samples_++;
        stream_frames_++;

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
        stats_.align_ms = average_ms(
            align_seconds_,
            align_samples_
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
        stats_.depth_ms = average_ms(
            depth_seconds_,
            depth_samples_
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
        stats_.total_ms = average_ms(
            total_seconds_,
            total_samples_
        );

        reset_window(now);
        return true;
    }

    const PerformanceStats& stats() const
    {
        return stats_;
    }

private:
    void add_stage(
        double& total_seconds,
        uint64_t& samples)
    {
        const auto now = Clock::now();
        total_seconds +=
            std::chrono::duration<double>(
                now - stage_start_
            ).count();
        samples++;
        stage_start_ = now;
    }

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
        acquire_samples_ = 0;
        align_samples_ = 0;
        preprocess_samples_ = 0;
        dpu_samples_ = 0;
        depth_samples_ = 0;
        overlay_samples_ = 0;
        jpeg_samples_ = 0;
        send_samples_ = 0;
        total_samples_ = 0;
        acquire_seconds_ = 0.0;
        align_seconds_ = 0.0;
        preprocess_seconds_ = 0.0;
        dpu_seconds_ = 0.0;
        depth_seconds_ = 0.0;
        overlay_seconds_ = 0.0;
        jpeg_seconds_ = 0.0;
        send_seconds_ = 0.0;
        total_seconds_ = 0.0;
    }

    Clock::time_point window_start_;
    Clock::time_point frame_start_;
    Clock::time_point stage_start_;
    PerformanceStats stats_;
    uint64_t stream_frames_ = 0;
    uint64_t pipeline_frames_ = 0;
    uint64_t acquire_samples_ = 0;
    uint64_t align_samples_ = 0;
    uint64_t preprocess_samples_ = 0;
    uint64_t dpu_samples_ = 0;
    uint64_t depth_samples_ = 0;
    uint64_t overlay_samples_ = 0;
    uint64_t jpeg_samples_ = 0;
    uint64_t send_samples_ = 0;
    uint64_t total_samples_ = 0;
    double acquire_seconds_ = 0.0;
    double align_seconds_ = 0.0;
    double preprocess_seconds_ = 0.0;
    double dpu_seconds_ = 0.0;
    double depth_seconds_ = 0.0;
    double overlay_seconds_ = 0.0;
    double jpeg_seconds_ = 0.0;
    double send_seconds_ = 0.0;
    double total_seconds_ = 0.0;
};

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

    void discover_hwmon()
    {
        DIR* directory = opendir("/sys/class/hwmon");

        if (!directory) {
            return;
        }

        while (dirent* entry = readdir(directory)) {
            const std::string entry_name(entry->d_name);

            if (entry_name.compare(0, 5, "hwmon") != 0) {
                continue;
            }

            const std::string path =
                "/sys/class/hwmon/" + entry_name;
            std::string name;

            if (!read_text_file(path + "/name", name)) {
                continue;
            }

            if (name == "ams") {
                ams_path_ = path;
            } else if (name.compare(0, 6, "ina260") == 0) {
                ina260_path_ = path;
            }
        }

        closedir(directory);
    }

    void update_cpu()
    {
        std::ifstream input("/proc/stat");
        std::string label;
        CpuTimes current;

        if (!(input
              >> label
              >> current.user
              >> current.nice
              >> current.system
              >> current.idle
              >> current.iowait
              >> current.irq
              >> current.softirq
              >> current.steal)
            || label != "cpu") {
            stats_.cpu_available = false;
            return;
        }

        stats_.cpu_available = false;

        if (has_cpu_sample_) {
            const uint64_t current_total = current.total();
            const uint64_t previous_total = previous_cpu_.total();
            const uint64_t current_idle = current.idle_total();
            const uint64_t previous_idle = previous_cpu_.idle_total();

            if (current_total > previous_total
                && current_idle >= previous_idle) {
                const uint64_t total_delta =
                    current_total - previous_total;
                const uint64_t idle_delta =
                    current_idle - previous_idle;

                if (idle_delta <= total_delta) {
                    stats_.cpu_percent =
                        100.0
                        * static_cast<double>(
                            total_delta - idle_delta
                        )
                        / static_cast<double>(total_delta);
                    stats_.cpu_available = true;
                }
            }
        }

        previous_cpu_ = current;
        has_cpu_sample_ = true;
    }

    void update_memory()
    {
        std::ifstream input("/proc/meminfo");
        std::string line;
        uint64_t total_kb = 0;
        uint64_t available_kb = 0;

        while (std::getline(input, line)) {
            std::istringstream values(line);
            std::string key;
            uint64_t value = 0;

            if (!(values >> key >> value)) {
                continue;
            }

            if (key == "MemTotal:") {
                total_kb = value;
            } else if (key == "MemAvailable:") {
                available_kb = value;
            }
        }

        stats_.ram_available =
            total_kb > 0 && available_kb <= total_kb;

        if (!stats_.ram_available) {
            return;
        }

        const uint64_t used_kb = total_kb - available_kb;
        stats_.ram_used_mb = used_kb / 1024;
        stats_.ram_total_mb = total_kb / 1024;
        stats_.ram_percent =
            100.0
            * static_cast<double>(used_kb)
            / static_cast<double>(total_kb);
    }

    void update_sensors()
    {
        stats_.temperature_available = false;
        stats_.pl_temperature_available = false;
        stats_.power_available = false;
        stats_.voltage_available = false;
        stats_.current_available = false;

        if (!ams_path_.empty()) {
            double ps_lpd = 0.0;
            double ps_fpd = 0.0;
            double pl = 0.0;
            const bool has_ps_lpd = read_number_file(
                ams_path_ + "/temp1_input",
                ps_lpd
            );
            const bool has_ps_fpd = read_number_file(
                ams_path_ + "/temp2_input",
                ps_fpd
            );

            if (has_ps_lpd || has_ps_fpd) {
                const double ps_temperature =
                    has_ps_lpd && has_ps_fpd
                    ? std::max(ps_lpd, ps_fpd)
                    : (has_ps_lpd ? ps_lpd : ps_fpd);
                stats_.temperature_c = ps_temperature / 1000.0;
                stats_.temperature_available = true;
            }

            if (read_number_file(
                    ams_path_ + "/temp3_input",
                    pl)) {
                stats_.pl_temperature_c = pl / 1000.0;
                stats_.pl_temperature_available = true;
            }
        }

        if (!ina260_path_.empty()) {
            double power_uw = 0.0;
            double voltage_mv = 0.0;
            double current_ma = 0.0;

            if (read_number_file(
                    ina260_path_ + "/power1_input",
                    power_uw)) {
                stats_.power_w = power_uw / 1000000.0;
                stats_.power_available = true;
            }

            if (read_number_file(
                    ina260_path_ + "/in1_input",
                    voltage_mv)) {
                stats_.voltage_v = voltage_mv / 1000.0;
                stats_.voltage_available = true;
            }

            if (read_number_file(
                    ina260_path_ + "/curr1_input",
                    current_ma)) {
                stats_.current_a = current_ma / 1000.0;
                stats_.current_available = true;
            }
        }
    }

    Clock::time_point last_update_{};
    bool has_update_ = false;
    CpuTimes previous_cpu_;
    bool has_cpu_sample_ = false;
    std::string ams_path_;
    std::string ina260_path_;
    SystemStats stats_;
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
        << " ms align="
        << performance.align_ms
        << " ms resize="
        << performance.preprocess_ms
        << " ms dpu="
        << performance.dpu_latency_ms
        << " ms dpu_rate="
        << performance.dpu_fps
        << " FPS depth="
        << performance.depth_ms
        << " ms overlay="
        << performance.overlay_ms
        << " ms jpeg="
        << performance.jpeg_ms
        << " ms send="
        << performance.send_ms
        << " ms total="
        << performance.total_ms
        << " ms | CPU=";

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


struct FaceAnnotation {
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

float median_depth_in_roi(
    const rs2::depth_frame& depth,
    float depth_scale,
    int x1,
    int y1,
    int x2,
    int y2)
{
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(depth.get_width() - 1, x2);
    y2 = std::min(depth.get_height() - 1, y2);

    if (x2 < x1 || y2 < y1) {
        return 0.0f;
    }

    const auto* data =
        reinterpret_cast<const uint16_t*>(depth.get_data());

    const int width = depth.get_width();

    std::vector<uint16_t> values;
    values.reserve((x2 - x1 + 1) * (y2 - y1 + 1));

    for (int y = y1; y <= y2; ++y) {
        for (int x = x1; x <= x2; ++x) {
            uint16_t raw = data[y * width + x];

            if (raw != 0) {
                values.push_back(raw);
            }
        }
    }

    if (values.empty()) {
        return 0.0f;
    }

    auto middle = values.begin() + values.size() / 2;

    std::nth_element(
        values.begin(),
        middle,
        values.end()
    );

    return static_cast<float>(*middle) * depth_scale;
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
        "<title>KV260 DenseBox Stream</title>"
        "<style>"
        "html,body{width:100%;height:100%;margin:0;background:#000;"
        "overflow:hidden;}"
        "body{display:flex;align-items:center;justify-content:center;}"
        "img{width:100vw;height:100vh;object-fit:contain;display:block;}"
        "</style></head>"
        "<body><img src=\"/stream.mjpg\" "
        "alt=\"KV260 DenseBox camera stream\"></body></html>";

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
            << "Creating DenseBox detector..."
            << std::endl;

        auto detector =
            vitis::ai::FaceDetect::create(
                "densebox_320_320"
            );

        if (!detector) {
            throw std::runtime_error(
                "Failed to create FaceDetect"
            );
        }

        const int model_width =
            detector->getInputWidth();

        const int model_height =
            detector->getInputHeight();

        std::cout
            << "Model input: "
            << model_width
            << "x"
            << model_height
            << std::endl;

        std::cout
            << "Detection threshold: "
            << detector->getThreshold()
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

        rs2::align align_to_color(
            RS2_STREAM_COLOR
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
            << "JPEG quality: "
            << options.jpeg_quality
            << std::endl;

        std::cout
            << "Warming up..."
            << std::endl;

        for (int i = 0; i < 10; ++i) {
            auto frames =
                pipeline.wait_for_frames();

            align_to_color.process(frames);
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
        SystemMonitor system_monitor;
        system_monitor.update_if_due();
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
        std::vector<FaceAnnotation> annotations;
        annotations.reserve(16);

        while (true) {
            performance_monitor.begin_frame();

            auto frames =
                pipeline.wait_for_frames();

            performance_monitor.mark_acquired();

            auto aligned =
                align_to_color.process(frames);

            auto color =
                aligned.get_color_frame();

            auto depth =
                aligned.get_depth_frame();

            if (!color || !depth) {
                continue;
            }

            performance_monitor.mark_aligned();

            cv::Mat image(
                cv::Size(
                    camera_configuration.color.width,
                    camera_configuration.color.height
                ),
                CV_8UC3,
                const_cast<void*>(
                    color.get_data()
                ),
                cv::Mat::AUTO_STEP
            );

            image = image.clone();

            cv::Mat model_input;

            cv::resize(
                image,
                model_input,
                cv::Size(
                    model_width,
                    model_height
                )
            );

            performance_monitor.mark_preprocessed();

            const auto dpu_start = Clock::now();
            auto result = detector->run(model_input);
            const auto dpu_end = Clock::now();

            performance_monitor.mark_dpu_complete(
                dpu_start,
                dpu_end
            );

            annotations.clear();

            if (annotations.capacity() < result.rects.size()) {
                annotations.reserve(result.rects.size());
            }

            for (const auto& face : result.rects) {

                int x = static_cast<int>(
                    face.x * image.cols
                );

                int y = static_cast<int>(
                    face.y * image.rows
                );

                int w = static_cast<int>(
                    face.width * image.cols
                );

                int h = static_cast<int>(
                    face.height * image.rows
                );

                x = std::clamp(
                    x,
                    0,
                    image.cols - 1
                );

                y = std::clamp(
                    y,
                    0,
                    image.rows - 1
                );

                w = std::min(
                    w,
                    image.cols - x
                );

                h = std::min(
                    h,
                    image.rows - y
                );

                if (w <= 0 || h <= 0) {
                    continue;
                }

                int cx = x + w / 2;
                int cy = y + h / 2;

                int roi_w = std::max(
                    2,
                    static_cast<int>(
                        w * DEPTH_ROI_SCALE
                    )
                );

                int roi_h = std::max(
                    2,
                    static_cast<int>(
                        h * DEPTH_ROI_SCALE
                    )
                );

                int dx1 =
                    cx - roi_w / 2;

                int dy1 =
                    cy - roi_h / 2;

                int dx2 =
                    cx + roi_w / 2;

                int dy2 =
                    cy + roi_h / 2;

                const float distance =
                    median_depth_in_roi(
                        depth,
                        depth_scale,
                        dx1,
                        dy1,
                        dx2,
                        dy2
                    );

                annotations.push_back({
                    x,
                    y,
                    w,
                    h,
                    dx1,
                    dy1,
                    dx2,
                    dy2,
                    face.score,
                    distance
                });
            }

            performance_monitor.mark_depth_complete();

            for (const FaceAnnotation& face : annotations) {
                cv::rectangle(
                    image,
                    cv::Rect(
                        face.x,
                        face.y,
                        face.width,
                        face.height
                    ),
                    cv::Scalar(0, 255, 0),
                    2
                );

                cv::rectangle(
                    image,
                    cv::Point(
                        std::max(0, face.depth_x1),
                        std::max(0, face.depth_y1)
                    ),
                    cv::Point(
                        std::min(
                            image.cols - 1,
                            face.depth_x2
                        ),
                        std::min(
                            image.rows - 1,
                            face.depth_y2
                        )
                    ),
                    cv::Scalar(255, 0, 0),
                    1
                );

                char label[128];

                if (face.distance > 0.0f) {
                    std::snprintf(
                        label,
                        sizeof(label),
                        "Face %.2f | %.2f m",
                        face.score,
                        face.distance
                    );
                } else {
                    std::snprintf(
                        label,
                        sizeof(label),
                        "Face %.2f | N/A",
                        face.score
                    );
                }

                cv::putText(
                    image,
                    label,
                    cv::Point(
                        face.x,
                        std::max(20, face.y - 8)
                    ),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA
                );
            }

            system_monitor.update_if_due();

            draw_telemetry(
                image,
                performance_monitor.stats(),
                system_monitor.stats()
            );

            performance_monitor.mark_overlay_complete();

            jpeg.clear();
            performance_monitor.begin_jpeg();

            const bool encoded = cv::imencode(
                ".jpg",
                image,
                jpeg,
                jpeg_params
            );

            performance_monitor.mark_jpeg_complete();

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

            performance_monitor.begin_send();

            if (!send_all(
                    client_fd,
                    frame_header.data(),
                    frame_header.size())) {
                break;
            }

            if (!send_all(
                    client_fd,
                    jpeg.data(),
                    jpeg.size())) {
                break;
            }

            const char* ending = "\r\n";

            if (!send_all(
                    client_fd,
                    ending,
                    2)) {
                break;
            }

            if (performance_monitor.mark_frame_sent()) {
                log_telemetry(
                    performance_monitor.stats(),
                    system_monitor.stats()
                );
            }
        }

        std::cout
            << "Client disconnected."
            << std::endl;

        close(client_fd);
        close(server_fd);

        std::cout
            << "Stopping camera..."
            << std::endl;

        pipeline.stop();

        std::cout
            << "Camera stopped"
            << std::endl;

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
