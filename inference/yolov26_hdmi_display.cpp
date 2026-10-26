#include "yolov26_hdmi_display.hpp"

#include <cstdlib>
#include <stdexcept>

#include <X11/Xlib.h>
#include <opencv2/highgui/highgui_c.h>

namespace yolov26 {
namespace hdmi {

namespace {

constexpr const char* WINDOW_NAME = "KV260 YOLOv26";

bool key_requests_exit(int key)
{
    return key == 27 || key == 'q' || key == 'Q';
}

bool window_visible(const char* name)
{
    try {
        return cvGetWindowHandle(name) != nullptr;
    } catch (...) {
        return false;
    }
}

}

HdmiDisplay::HdmiDisplay(
    const std::string& display_name,
    bool fullscreen)
    : fullscreen_(fullscreen)
{
    Display* display = XOpenDisplay(display_name.c_str());

    if (!display) {
        throw std::runtime_error(
            "cannot open X display '" + display_name
            + "': check DISPLAY and XAUTHORITY"
        );
    }

    display_ = display;
    const int screen = DefaultScreen(display);
    screen_width_ = DisplayWidth(display, screen);
    screen_height_ = DisplayHeight(display, screen);

    if (setenv("DISPLAY", display_name.c_str(), 1) != 0) {
        XCloseDisplay(display);
        display_ = nullptr;
        throw std::runtime_error(
            "cannot set DISPLAY='" + display_name + "'"
        );
    }

    try {
        cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);

        if (fullscreen_) {
            cv::setWindowProperty(
                WINDOW_NAME,
                cv::WND_PROP_FULLSCREEN,
                cv::WINDOW_FULLSCREEN
            );
        }
    } catch (...) {
        try {
            cv::destroyWindow(WINDOW_NAME);
        } catch (...) {
        }
        XCloseDisplay(display);
        display_ = nullptr;
        throw;
    }

    if (!window_visible(WINDOW_NAME)) {
        try {
            cv::destroyWindow(WINDOW_NAME);
        } catch (...) {
        }
        XCloseDisplay(display);
        display_ = nullptr;
        throw std::runtime_error(
            "cannot create a visible window '" + display_name
            + "': HighGUI window handle is null"
        );
    }
}

HdmiDisplay::~HdmiDisplay()
{
    try {
        cv::destroyWindow(WINDOW_NAME);
    } catch (...) {
    }

    if (display_) {
        XCloseDisplay(static_cast<Display*>(display_));
    }
}

int HdmiDisplay::target_width() const
{
    return screen_width_;
}

int HdmiDisplay::target_height() const
{
    return screen_height_;
}

bool HdmiDisplay::show(const cv::Mat& image)
{
    if (!window_open_ || image.empty()) {
        return false;
    }

    int canvas_width = screen_width_;
    int canvas_height = screen_height_;

    if (!fullscreen_) {
        try {
            const cv::Rect area =
                cv::getWindowImageRect(WINDOW_NAME);
            if (area.width > 0 && area.height > 0) {
                canvas_width = area.width;
                canvas_height = area.height;
            }
        } catch (...) {
        }
    }

    if (canvas_width <= 0 || canvas_height <= 0) {
        canvas_width = image.cols;
        canvas_height = image.rows;
    }

    const FittedImage fit = fit_image(
        image.cols,
        image.rows,
        canvas_width,
        canvas_height
    );

    cv::Mat canvas(
        canvas_height,
        canvas_width,
        image.type(),
        cv::Scalar(0, 0, 0)
    );
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(fit.width, fit.height));
    resized.copyTo(canvas(cv::Rect(fit.x, fit.y, fit.width, fit.height)));
    cv::imshow(WINDOW_NAME, canvas);

    return poll_key(cv::waitKey(1));
}

bool HdmiDisplay::poll()
{
    return poll_key(cv::waitKey(10));
}

bool HdmiDisplay::poll_key(int key)
{
    if (key_requests_exit(key)) {
        window_open_ = false;
        return false;
    }

    if (!window_visible(WINDOW_NAME)) {
        window_open_ = false;
        return false;
    }

    return true;
}

}
}
