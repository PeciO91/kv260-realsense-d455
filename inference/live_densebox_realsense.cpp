#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vitis/ai/facedetect.hpp>


constexpr int WIDTH = 640;
constexpr int HEIGHT = 480;
constexpr int CAMERA_FPS = 30;
constexpr int PORT = 8081;

constexpr float DEPTH_ROI_SCALE = 0.35f;
constexpr int JPEG_QUALITY = 75;
constexpr double TELEMETRY_INTERVAL_SECONDS = 1.0;
constexpr int TELEMETRY_PANEL_WIDTH = 300;
constexpr int TELEMETRY_PANEL_HEIGHT = 260;

using Clock = std::chrono::steady_clock;

struct PerformanceStats {
    bool available = false;
    double stream_fps = 0.0;
    double pipeline_fps = 0.0;
    double dpu_latency_ms = 0.0;
    double dpu_fps = 0.0;
    double align_ms = 0.0;
    double preprocess_ms = 0.0;
    double postprocess_ms = 0.0;
    double jpeg_ms = 0.0;
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
          stage_start_(window_start_)
    {
    }

    void begin_pipeline()
    {
        stage_start_ = Clock::now();
    }

    void mark_aligned()
    {
        add_stage(align_seconds_, align_samples_);
    }

    void mark_preprocessed()
    {
        add_stage(preprocess_seconds_, preprocess_samples_);
    }

    void mark_dpu_complete()
    {
        add_stage(dpu_seconds_, dpu_samples_);
    }

    void mark_postprocessed()
    {
        add_stage(postprocess_seconds_, postprocess_samples_);
    }

    void finish_pipeline()
    {
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

    bool mark_stream_frame()
    {
        stream_frames_++;

        const auto now = Clock::now();
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
        stats_.postprocess_ms = average_ms(
            postprocess_seconds_,
            postprocess_samples_
        );
        stats_.jpeg_ms = average_ms(
            jpeg_seconds_,
            jpeg_samples_
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
        align_samples_ = 0;
        preprocess_samples_ = 0;
        dpu_samples_ = 0;
        postprocess_samples_ = 0;
        jpeg_samples_ = 0;
        align_seconds_ = 0.0;
        preprocess_seconds_ = 0.0;
        dpu_seconds_ = 0.0;
        postprocess_seconds_ = 0.0;
        jpeg_seconds_ = 0.0;
    }

    Clock::time_point window_start_;
    Clock::time_point stage_start_;
    PerformanceStats stats_;
    uint64_t stream_frames_ = 0;
    uint64_t pipeline_frames_ = 0;
    uint64_t align_samples_ = 0;
    uint64_t preprocess_samples_ = 0;
    uint64_t dpu_samples_ = 0;
    uint64_t postprocess_samples_ = 0;
    uint64_t jpeg_samples_ = 0;
    double align_seconds_ = 0.0;
    double preprocess_seconds_ = 0.0;
    double dpu_seconds_ = 0.0;
    double postprocess_seconds_ = 0.0;
    double jpeg_seconds_ = 0.0;
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
    const int margin = 8;
    const int panel_width = std::min(
        TELEMETRY_PANEL_WIDTH,
        image.cols - 2 * margin
    );
    const int panel_height = std::min(
        TELEMETRY_PANEL_HEIGHT,
        image.rows - 2 * margin
    );

    if (panel_width <= 0 || panel_height <= 0) {
        return;
    }

    const cv::Rect panel_rect(
        margin,
        margin,
        panel_width,
        panel_height
    );
    static const cv::Mat dark_panel(
        TELEMETRY_PANEL_HEIGHT,
        TELEMETRY_PANEL_WIDTH,
        CV_8UC3,
        cv::Scalar(0, 0, 0)
    );
    cv::Mat frame_roi = image(panel_rect);
    const cv::Mat dark_roi = dark_panel(
        cv::Rect(0, 0, panel_width, panel_height)
    );

    cv::addWeighted(
        frame_roi,
        0.35,
        dark_roi,
        0.65,
        0.0,
        frame_roi
    );

    const int x = margin + 10;
    int y = margin + 19;
    const int font = cv::FONT_HERSHEY_SIMPLEX;
    char text[128];

    const auto draw_heading = [&](const char* value) {
        cv::putText(
            image,
            value,
            cv::Point(x, y),
            font,
            0.47,
            cv::Scalar(0, 255, 255),
            1,
            cv::LINE_AA
        );
        y += 19;
    };

    const auto draw_line = [&](const char* value) {
        cv::putText(
            image,
            value,
            cv::Point(x, y),
            font,
            0.42,
            cv::Scalar(245, 245, 245),
            1,
            cv::LINE_AA
        );
        y += 17;
    };

    draw_heading("Performance");

    if (performance.available) {
        std::snprintf(
            text,
            sizeof(text),
            "Stream FPS: %.1f",
            performance.stream_fps
        );
        draw_line(text);
        std::snprintf(
            text,
            sizeof(text),
            "Pipeline FPS: %.1f",
            performance.pipeline_fps
        );
        draw_line(text);
        std::snprintf(
            text,
            sizeof(text),
            "DPU latency: %.1f ms",
            performance.dpu_latency_ms
        );
        draw_line(text);
        std::snprintf(
            text,
            sizeof(text),
            "DPU rate: %.1f FPS",
            performance.dpu_fps
        );
        draw_line(text);
    } else {
        draw_line("Stream FPS: N/A");
        draw_line("Pipeline FPS: N/A");
        draw_line("DPU latency: N/A");
        draw_line("DPU rate: N/A");
    }

    y += 7;
    draw_heading("System Info");

    if (system.cpu_available) {
        std::snprintf(
            text,
            sizeof(text),
            "CPU: %.1f%%",
            system.cpu_percent
        );
        draw_line(text);
    } else {
        draw_line("CPU: N/A");
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
        draw_line(text);
    } else {
        draw_line("RAM: N/A");
    }

    if (system.temperature_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Temp: %.1f C",
            system.temperature_c
        );
        draw_line(text);
    } else {
        draw_line("Temp: N/A");
    }

    if (system.pl_temperature_available) {
        std::snprintf(
            text,
            sizeof(text),
            "PL Temp: %.1f C",
            system.pl_temperature_c
        );
        draw_line(text);
    } else {
        draw_line("PL Temp: N/A");
    }

    if (system.power_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Power: %.2f W",
            system.power_w
        );
        draw_line(text);
    } else {
        draw_line("Power: N/A");
    }

    if (system.voltage_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Voltage: %.2f V",
            system.voltage_v
        );
        draw_line(text);
    } else {
        draw_line("Voltage: N/A");
    }

    if (system.current_available) {
        std::snprintf(
            text,
            sizeof(text),
            "Current: %.2f A",
            system.current_a
        );
        draw_line(text);
    } else {
        draw_line("Current: N/A");
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
        << " FPS dpu="
        << performance.dpu_latency_ms
        << " ms dpu_rate="
        << performance.dpu_fps
        << " FPS align="
        << performance.align_ms
        << " ms preprocess="
        << performance.preprocess_ms
        << " ms postprocess="
        << performance.postprocess_ms
        << " ms jpeg="
        << performance.jpeg_ms
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


int create_server()
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
    address.sin_port = htons(PORT);

    if (bind(
            server_fd,
            reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) < 0) {
        close(server_fd);
        throw std::runtime_error("bind() failed");
    }

    if (listen(server_fd, 1) < 0) {
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


int main()
{
    try {
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
            WIDTH,
            HEIGHT,
            RS2_FORMAT_BGR8,
            CAMERA_FPS
        );

        config.enable_stream(
            RS2_STREAM_DEPTH,
            WIDTH,
            HEIGHT,
            RS2_FORMAT_Z16,
            CAMERA_FPS
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
            << "Warming up..."
            << std::endl;

        for (int i = 0; i < 10; ++i) {
            auto frames =
                pipeline.wait_for_frames();

            align_to_color.process(frames);
        }

        int server_fd = create_server();

        std::cout << std::endl;
        std::cout
            << "Open on PC:"
            << std::endl;

        std::cout
            << "http://147.32.163.22:"
            << PORT
            << "/"
            << std::endl;

        std::cout
            << "Waiting for browser..."
            << std::endl;

        sockaddr_in client_address{};
        socklen_t client_len =
            sizeof(client_address);

        int client_fd = accept(
            server_fd,
            reinterpret_cast<sockaddr*>(
                &client_address
            ),
            &client_len
        );

        if (client_fd < 0) {
            throw std::runtime_error(
                "accept() failed"
            );
        }

        char request[2048];
        recv(
            client_fd,
            request,
            sizeof(request),
            0
        );

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

        while (true) {
            performance_monitor.begin_pipeline();

            auto frames =
                pipeline.wait_for_frames();

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
                cv::Size(WIDTH, HEIGHT),
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

            auto result =
                detector->run(model_input);

            performance_monitor.mark_dpu_complete();

            for (const auto& face :
                 result.rects) {

                int x = static_cast<int>(
                    face.x * WIDTH
                );

                int y = static_cast<int>(
                    face.y * HEIGHT
                );

                int w = static_cast<int>(
                    face.width * WIDTH
                );

                int h = static_cast<int>(
                    face.height * HEIGHT
                );

                x = std::clamp(
                    x,
                    0,
                    WIDTH - 1
                );

                y = std::clamp(
                    y,
                    0,
                    HEIGHT - 1
                );

                w = std::min(
                    w,
                    WIDTH - x
                );

                h = std::min(
                    h,
                    HEIGHT - y
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

                float distance =
                    median_depth_in_roi(
                        depth,
                        depth_scale,
                        dx1,
                        dy1,
                        dx2,
                        dy2
                    );

                cv::rectangle(
                    image,
                    cv::Rect(x, y, w, h),
                    cv::Scalar(
                        0,
                        255,
                        0
                    ),
                    2
                );

                cv::rectangle(
                    image,
                    cv::Point(
                        std::max(0, dx1),
                        std::max(0, dy1)
                    ),
                    cv::Point(
                        std::min(
                            WIDTH - 1,
                            dx2
                        ),
                        std::min(
                            HEIGHT - 1,
                            dy2
                        )
                    ),
                    cv::Scalar(
                        255,
                        0,
                        0
                    ),
                    1
                );

                char label[128];

                if (distance > 0.0f) {
                    std::snprintf(
                        label,
                        sizeof(label),
                        "Face %.2f | %.2f m",
                        face.score,
                        distance
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
                        x,
                        std::max(
                            20,
                            y - 8
                        )
                    ),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.55,
                    cv::Scalar(
                        0,
                        255,
                        0
                    ),
                    2,
                    cv::LINE_AA
                );
            }

            performance_monitor.mark_postprocessed();
            system_monitor.update_if_due();

            draw_telemetry(
                image,
                performance_monitor.stats(),
                system_monitor.stats()
            );

            performance_monitor.finish_pipeline();

            std::vector<uchar> jpeg;

            std::vector<int> params = {
                cv::IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            };

            performance_monitor.begin_jpeg();

            const bool encoded = cv::imencode(
                ".jpg",
                image,
                jpeg,
                params
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

            if (performance_monitor.mark_stream_frame()) {
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
