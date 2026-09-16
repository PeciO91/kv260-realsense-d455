#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <vitis/ai/facedetect.hpp>


constexpr int WIDTH = 640;
constexpr int HEIGHT = 480;
constexpr int FPS = 30;
constexpr int WARMUP_FRAMES = 10;

constexpr float DEPTH_ROI_SCALE = 0.35f;


float median_depth_in_roi(
    const rs2::depth_frame& depth,
    int x1,
    int y1,
    int x2,
    int y2)
{
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(WIDTH - 1, x2);
    y2 = std::min(HEIGHT - 1, y2);

    std::vector<float> values;

    for (int y = y1; y <= y2; ++y) {
        for (int x = x1; x <= x2; ++x) {
            float d = depth.get_distance(x, y);

            if (d > 0.0f && std::isfinite(d)) {
                values.push_back(d);
            }
        }
    }

    if (values.empty()) {
        return 0.0f;
    }

    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());

    return *middle;
}


int main()
{
    try {
        std::cout << "Creating DenseBox face detector..." << std::endl;

        auto detector =
            vitis::ai::FaceDetect::create("densebox_320_320");

        if (!detector) {
            std::cerr << "Failed to create FaceDetect model." << std::endl;
            return 1;
        }

        std::cout << "Face detector ready." << std::endl;

        rs2::pipeline pipeline;
        rs2::config config;

        config.enable_stream(
            RS2_STREAM_COLOR,
            WIDTH,
            HEIGHT,
            RS2_FORMAT_BGR8,
            FPS);

        config.enable_stream(
            RS2_STREAM_DEPTH,
            WIDTH,
            HEIGHT,
            RS2_FORMAT_Z16,
            FPS);

        rs2::align align_to_color(RS2_STREAM_COLOR);

        std::cout << "Starting RealSense D455..." << std::endl;

        auto profile = pipeline.start(config);

        auto device = profile.get_device();

        std::cout
            << "Camera: "
            << device.get_info(RS2_CAMERA_INFO_NAME)
            << std::endl;

        std::cout
            << "Serial: "
            << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
            << std::endl;

        std::cout
            << "Warming up..."
            << std::endl;

        for (int i = 0; i < WARMUP_FRAMES; ++i) {
            auto frames = pipeline.wait_for_frames();
            align_to_color.process(frames);
        }

        std::cout
            << "Capturing aligned RGB + depth frame..."
            << std::endl;

        auto frames = pipeline.wait_for_frames();
        auto aligned = align_to_color.process(frames);

        auto color = aligned.get_color_frame();
        auto depth = aligned.get_depth_frame();

        if (!color || !depth) {
            std::cerr << "Incomplete frameset." << std::endl;
            pipeline.stop();
            return 1;
        }

        cv::Mat image(
            cv::Size(WIDTH, HEIGHT),
            CV_8UC3,
            const_cast<void*>(color.get_data()),
            cv::Mat::AUTO_STEP);

        // Make our own buffer before modifying the image.
        image = image.clone();

        std::cout << "Running DenseBox on DPU..." << std::endl;

        auto result = detector->run(image);

        std::cout
            << "Detected faces: "
            << result.rects.size()
            << std::endl;

        int face_index = 0;

        for (const auto& face : result.rects) {
            ++face_index;

            int x = static_cast<int>(face.x * image.cols);
            int y = static_cast<int>(face.y * image.rows);
            int w = static_cast<int>(face.width * image.cols);
            int h = static_cast<int>(face.height * image.rows);

            x = std::clamp(x, 0, image.cols - 1);
            y = std::clamp(y, 0, image.rows - 1);
            w = std::min(w, image.cols - x);
            h = std::min(h, image.rows - y);

            if (w <= 0 || h <= 0) {
                continue;
            }

            // Use a smaller region around the centre of the detected face.
            // This avoids including too much background in depth estimation.
            const int cx = x + w / 2;
            const int cy = y + h / 2;

            const int roi_w =
                std::max(2, static_cast<int>(w * DEPTH_ROI_SCALE));

            const int roi_h =
                std::max(2, static_cast<int>(h * DEPTH_ROI_SCALE));

            const int dx1 = cx - roi_w / 2;
            const int dy1 = cy - roi_h / 2;
            const int dx2 = cx + roi_w / 2;
            const int dy2 = cy + roi_h / 2;

            float distance = median_depth_in_roi(
                depth,
                dx1,
                dy1,
                dx2,
                dy2);

            std::cout
                << "Face " << face_index
                << " score=" << face.score
                << " bbox=("
                << x << ", "
                << y << ", "
                << w << ", "
                << h << ")"
                << " distance=";

            if (distance > 0.0f) {
                std::cout << distance << " m";
            } else {
                std::cout << "N/A";
            }

            std::cout << std::endl;

            cv::rectangle(
                image,
                cv::Rect(x, y, w, h),
                cv::Scalar(0, 255, 0),
                2);

            cv::rectangle(
                image,
                cv::Point(
                    std::max(0, dx1),
                    std::max(0, dy1)),
                cv::Point(
                    std::min(image.cols - 1, dx2),
                    std::min(image.rows - 1, dy2)),
                cv::Scalar(255, 0, 0),
                1);

            std::string label;

            if (distance > 0.0f) {
                char buffer[128];

                std::snprintf(
                    buffer,
                    sizeof(buffer),
                    "Face %.2f  %.2f m",
                    face.score,
                    distance);

                label = buffer;
            } else {
                label = "Face - distance N/A";
            }

            int text_y = std::max(20, y - 10);

            cv::putText(
                image,
                label,
                cv::Point(x, text_y),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 255, 0),
                2,
                cv::LINE_AA);
        }

        const std::string output =
            "densebox_realsense_result.jpg";

        if (!cv::imwrite(output, image)) {
            std::cerr
                << "Failed to save "
                << output
                << std::endl;

            pipeline.stop();
            return 1;
        }

        std::cout
            << "Saved: "
            << output
            << std::endl;

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
