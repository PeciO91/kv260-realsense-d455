#include <algorithm>
#include <arpa/inet.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <netinet/in.h>
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

        auto fps_start =
            std::chrono::steady_clock::now();

        int frame_counter = 0;

        while (true) {
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

            auto result =
                detector->run(model_input);

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

            frame_counter++;

            auto now =
                std::chrono::steady_clock::now();

            double seconds =
                std::chrono::duration<double>(
                    now - fps_start
                ).count();

            if (seconds >= 1.0) {
                double fps =
                    frame_counter / seconds;

                char fps_text[64];

                std::snprintf(
                    fps_text,
                    sizeof(fps_text),
                    "Pipeline: %.1f FPS",
                    fps
                );

                std::cout
                    << fps_text
                    << " | faces: "
                    << result.rects.size()
                    << std::endl;

                frame_counter = 0;
                fps_start = now;
            }

            std::vector<uchar> jpeg;

            std::vector<int> params = {
                cv::IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            };

            if (!cv::imencode(
                    ".jpg",
                    image,
                    jpeg,
                    params)) {
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
