#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "yolov26_hdmi_control.hpp"

namespace yolov26 {
namespace hdmi {

class HdmiDisplay {
public:
    HdmiDisplay(
        const std::string& display_name,
        bool fullscreen);
    ~HdmiDisplay();

    HdmiDisplay(const HdmiDisplay&) = delete;
    HdmiDisplay& operator=(const HdmiDisplay&) = delete;

    int target_width() const;
    int target_height() const;

    bool show(const cv::Mat& image);
    bool poll();

private:
    bool poll_key(int key);

    void* display_ = nullptr;
    bool fullscreen_ = false;
    int screen_width_ = 0;
    int screen_height_ = 0;
    bool window_open_ = true;
    cv::Mat scaled_;
};

}
}
