#pragma once

#include <cstdint>
#include <termios.h>

#include <opencv2/opencv.hpp>
#include <xf86drmMode.h>

#include "yolov26_display.hpp"

namespace yolov26 {
namespace hdmi {

class KmsDisplay : public Display {
public:
    explicit KmsDisplay(bool fullscreen);
    ~KmsDisplay() override;

    KmsDisplay(const KmsDisplay&) = delete;
    KmsDisplay& operator=(const KmsDisplay&) = delete;

    int target_width() const override;
    int target_height() const override;
    bool show(const cv::Mat& image) override;
    bool poll() override;

private:
    struct DumbBuffer {
        uint32_t handle = 0;
        uint32_t fb_id = 0;
        uint32_t pitch = 0;
        uint64_t size = 0;
        void* map = nullptr;
        int width = 0;
        int height = 0;
    };

    void open_card();
    void setup_crtc();
    void select_overlay_plane();
    void create_dumb(
        DumbBuffer& buffer,
        int width,
        int height,
        uint32_t bpp,
        uint32_t fourcc);
    void destroy_dumb(DumbBuffer& buffer);
    bool ensure_frame_buffers(int width, int height);
    void poll_keys();
    void setup_terminal();
    void restore_terminal();
    void teardown();

    int fd_ = -1;
    bool fullscreen_;
    bool window_open_ = true;
    bool hw_scaling_ = true;
    int screen_width_ = 0;
    int screen_height_ = 0;
    uint32_t crtc_id_ = 0;
    int crtc_index_ = 0;
    uint32_t connector_id_ = 0;
    uint32_t plane_id_ = 0;
    drmModeCrtc* saved_crtc_ = nullptr;
    drmModeModeInfo mode_{};
    DumbBuffer primary_fb_;
    DumbBuffer frame_fbs_[2];
    int frame_index_ = 0;
    bool stdin_raw_ = false;
    struct termios saved_termios_{};
    int saved_stdin_flags_ = -1;
};

}
}
