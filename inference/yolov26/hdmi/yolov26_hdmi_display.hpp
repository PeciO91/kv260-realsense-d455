#pragma once

#include <string>

#include <opencv2/opencv.hpp>

#include "yolov26_display.hpp"
#include "yolov26_hdmi_control.hpp"

namespace yolov26 {
namespace hdmi {

class HdmiDisplay : public Display {
public:
    HdmiDisplay(
        const std::string& display_name,
        bool fullscreen);
    ~HdmiDisplay() override;

    HdmiDisplay(const HdmiDisplay&) = delete;
    HdmiDisplay& operator=(const HdmiDisplay&) = delete;

    int target_width() const override;
    int target_height() const override;

    bool show(const cv::Mat& image) override;
    bool poll() override;

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
