#pragma once

#include <opencv2/opencv.hpp>

namespace yolov26 {

class Display {
public:
    virtual ~Display() = default;

    virtual int target_width() const = 0;
    virtual int target_height() const = 0;
    virtual bool show(const cv::Mat& image) = 0;
    virtual bool poll() = 0;
};

}
