#pragma once

#include <cstdint>
#include <termios.h>

#include <opencv2/opencv.hpp>
#include <xf86drmMode.h>

#include "yolov26_display.hpp"

namespace yolov26 {
namespace hdmi {

enum class KmsPlane {
    Primary,
    Overlay,
};

enum class KmsScaling {
    Nearest,
    Linear,
};

namespace detail {

cv::Rect kms_content_rect(
    int image_width,
    int image_height,
    int screen_width,
    int screen_height,
    bool fullscreen);

cv::Rect compose_kms_frame(
    const cv::Mat& image,
    cv::Mat& destination,
    bool fullscreen,
    cv::Mat& scratch,
    KmsScaling scaling = KmsScaling::Nearest);

} // namespace detail

class KmsDisplay : public Display {
public:
    KmsDisplay(bool fullscreen, KmsPlane plane, KmsScaling scaling);
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

    struct Blending {
        bool saved = false;
        uint32_t plane_id = 0;
        uint32_t alpha_prop = 0;
        uint32_t galpha_prop = 0;
        uint64_t alpha = 0;
        uint64_t galpha = 0;
    };

    void open_card();
    void select_planes();
    void allocate_frame_buffers();
    void setup_crtc();
    void save_blending();
    void apply_blending(uint64_t galpha, uint64_t alpha);
    void restore_blending();
    void create_dumb(
        DumbBuffer& buffer,
        int width,
        int height,
        uint32_t bpp,
        uint32_t fourcc);
    void destroy_dumb(DumbBuffer& buffer);
    void poll_keys();
    void setup_terminal();
    void restore_terminal();
    void teardown();

    int fd_ = -1;
    bool fullscreen_;
    KmsPlane plane_;
    KmsScaling scaling_;
    bool window_open_ = true;
    int screen_width_ = 0;
    int screen_height_ = 0;
    uint32_t crtc_id_ = 0;
    int crtc_index_ = 0;
    uint32_t connector_id_ = 0;
    uint32_t primary_plane_id_ = 0;
    uint32_t overlay_plane_id_ = 0;
    uint32_t present_plane_id_ = 0;
    drmModeCrtc* saved_crtc_ = nullptr;
    drmModeModeInfo mode_{};
    Blending blending_;
    DumbBuffer primary_fb_;
    DumbBuffer frame_fbs_[2];
    int frame_index_ = 0;
    int frame_type_ = 0;
    cv::Rect content_rect_;
    cv::Mat resize_scratch_;
    bool stdin_raw_ = false;
    struct termios saved_termios_{};
    int saved_stdin_flags_ = -1;
};

}
}
