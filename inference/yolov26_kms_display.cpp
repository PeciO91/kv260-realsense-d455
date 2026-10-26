#include "yolov26_kms_display.hpp"

#include "yolov26_hdmi_control.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/select.h>
#include <thread>
#include <unistd.h>

#include <drm.h>
#include <drm_mode.h>
#include <drm_fourcc.h>
#include <xf86drm.h>

namespace yolov26 {
namespace hdmi {

namespace {

constexpr int MAX_CARDS = 8;
constexpr int POLL_TIMEOUT_MS = 5;

bool key_requests_exit(int key)
{
    return key == 27 || key == 'q' || key == 'Q';
}

std::string card_path(int index)
{
    return "/dev/dri/card" + std::to_string(index);
}

} // namespace

KmsDisplay::KmsDisplay(bool fullscreen)
    : fullscreen_(fullscreen)
{
    try {
        open_card();
        setup_crtc();
        select_overlay_plane();
        setup_terminal();
    } catch (...) {
        teardown();
        throw;
    }
}

KmsDisplay::~KmsDisplay()
{
    teardown();
}

void KmsDisplay::open_card()
{
    for (int i = 0; i < MAX_CARDS && fd_ < 0; ++i) {
        const std::string path = card_path(i);
        const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            continue;
        }

        drmModeRes* res = drmModeGetResources(fd);
        if (res) {
            for (int c = 0; c < res->count_connectors; ++c) {
                drmModeConnector* conn =
                    drmModeGetConnector(fd, res->connectors[c]);
                if (conn) {
                    if (conn->connection == DRM_MODE_CONNECTED
                        && conn->count_modes > 0) {
                        connector_id_ = conn->connector_id;
                        for (int m = 0; m < conn->count_modes; ++m) {
                            if (conn->modes[m].type
                                & DRM_MODE_TYPE_PREFERRED) {
                                mode_ = conn->modes[m];
                                break;
                            }
                        }
                        if (mode_.hdisplay == 0) {
                            mode_ = conn->modes[0];
                        }
                        screen_width_ = mode_.hdisplay;
                        screen_height_ = mode_.vdisplay;
                    }
                    drmModeFreeConnector(conn);
                }
                if (connector_id_ != 0) {
                    break;
                }
            }

            if (connector_id_ != 0 && res->count_crtcs > 0) {
                crtc_id_ = res->crtcs[0];
                crtc_index_ = 0;
                saved_crtc_ = drmModeGetCrtc(fd, crtc_id_);
            }
            drmModeFreeResources(res);
        }

        if (connector_id_ == 0 || crtc_id_ == 0) {
            ::close(fd);
        } else {
            fd_ = fd;
        }
    }

    if (fd_ < 0) {
        throw std::runtime_error(
            "no DRM card with a connected display found"
        );
    }

    if (drmSetMaster(fd_) != 0) {
        throw std::runtime_error(
            "cannot acquire DRM master: stop the X server"
            " first (systemctl stop xserver-nodm)"
        );
    }
}

void KmsDisplay::setup_crtc()
{
    create_dumb(
        primary_fb_,
        screen_width_,
        screen_height_,
        16,
        DRM_FORMAT_RGB565
    );
    std::memset(primary_fb_.map, 0, primary_fb_.size);

    if (drmModeSetCrtc(
            fd_,
            crtc_id_,
            primary_fb_.fb_id,
            0,
            0,
            &connector_id_,
            1,
            &mode_) != 0) {
        throw std::runtime_error(
            std::string("drmModeSetCrtc failed: ")
            + std::strerror(errno)
        );
    }
}

void KmsDisplay::select_overlay_plane()
{
    drmModePlaneRes* planes = drmModeGetPlaneResources(fd_);
    if (!planes) {
        throw std::runtime_error("no DRM planes available");
    }

    for (uint32_t i = 0; i < planes->count_planes && !plane_id_; ++i) {
        drmModePlane* plane = drmModeGetPlane(fd_, planes->planes[i]);
        if (!plane) {
            continue;
        }

        const uint32_t id = plane->plane_id;
        const bool usable = plane->possible_crtcs & (1u << crtc_index_);
        bool has_xr24 = false;
        for (uint32_t f = 0; f < plane->count_formats; ++f) {
            if (plane->formats[f] == DRM_FORMAT_XRGB8888) {
                has_xr24 = true;
                break;
            }
        }
        drmModeFreePlane(plane);

        if (!usable || !has_xr24) {
            continue;
        }

        drmModeObjectProperties* props =
            drmModeObjectGetProperties(
                fd_, id, DRM_MODE_OBJECT_PLANE
            );
        bool overlay = false;
        if (props) {
            for (uint32_t p = 0; p < props->count_props; ++p) {
                drmModePropertyRes* prop =
                    drmModeGetProperty(fd_, props->props[p]);
                if (prop) {
                    if (std::strcmp(prop->name, "type") == 0
                        && props->prop_values[p]
                            == DRM_PLANE_TYPE_OVERLAY) {
                        overlay = true;
                    }
                    drmModeFreeProperty(prop);
                }
            }
            drmModeFreeObjectProperties(props);
        }

        if (overlay) {
            plane_id_ = id;
        }
    }

    drmModeFreePlaneResources(planes);

    if (!plane_id_) {
        throw std::runtime_error(
            "no usable DRM overlay plane with XR24 support"
        );
    }
}

void KmsDisplay::create_dumb(
    DumbBuffer& buffer,
    int width,
    int height,
    uint32_t bpp,
    uint32_t fourcc)
{
    drm_mode_create_dumb create{};
    create.width = width;
    create.height = height;
    create.bpp = bpp;

    if (drmIoctl(fd_, DRM_IOCTL_MODE_CREATE_DUMB, &create) != 0) {
        throw std::runtime_error(
            std::string("CREATE_DUMB failed: ")
            + std::strerror(errno)
        );
    }

    buffer.handle = create.handle;
    buffer.pitch = create.pitch;
    buffer.size = create.size;
    buffer.width = width;
    buffer.height = height;

    const uint32_t handles[4] = {buffer.handle, 0, 0, 0};
    const uint32_t pitches[4] = {buffer.pitch, 0, 0, 0};
    const uint32_t offsets[4] = {0, 0, 0, 0};

    if (drmModeAddFB2(
            fd_,
            width,
            height,
            fourcc,
            handles,
            pitches,
            offsets,
            &buffer.fb_id,
            0) != 0) {
        throw std::runtime_error(
            std::string("drmModeAddFB2 failed: ")
            + std::strerror(errno)
        );
    }

    drm_mode_map_dumb map{};
    map.handle = buffer.handle;
    if (drmIoctl(fd_, DRM_IOCTL_MODE_MAP_DUMB, &map) != 0) {
        throw std::runtime_error(
            std::string("MAP_DUMB failed: ")
            + std::strerror(errno)
        );
    }

    buffer.map = mmap(
        nullptr,
        buffer.size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd_,
        map.offset
    );
    if (buffer.map == MAP_FAILED) {
        buffer.map = nullptr;
        throw std::runtime_error(
            std::string("mmap dumb buffer failed: ")
            + std::strerror(errno)
        );
    }
}

void KmsDisplay::destroy_dumb(DumbBuffer& buffer)
{
    if (buffer.fb_id) {
        drmModeRmFB(fd_, buffer.fb_id);
        buffer.fb_id = 0;
    }
    if (buffer.map) {
        munmap(buffer.map, buffer.size);
        buffer.map = nullptr;
    }
    if (buffer.handle) {
        drm_mode_destroy_dumb destroy{};
        destroy.handle = buffer.handle;
        drmIoctl(fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
        buffer.handle = 0;
    }
}

bool KmsDisplay::ensure_frame_buffers(int width, int height)
{
    if (frame_fbs_[0].width == width
        && frame_fbs_[0].height == height
        && frame_fbs_[0].map) {
        return true;
    }

    try {
        for (auto& buffer : frame_fbs_) {
            destroy_dumb(buffer);
            create_dumb(buffer, width, height, 32, DRM_FORMAT_XRGB8888);
        }
    } catch (...) {
        return false;
    }
    return true;
}

void KmsDisplay::setup_terminal()
{
    if (!isatty(STDIN_FILENO)) {
        return;
    }

    if (tcgetattr(STDIN_FILENO, &saved_termios_) == 0) {
        struct termios raw = saved_termios_;
        raw.c_lflag &= ~(ICANON | ECHO);
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) == 0) {
            stdin_raw_ = true;
        }
    }

    saved_stdin_flags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(
        STDIN_FILENO,
        F_SETFL,
        saved_stdin_flags_ | O_NONBLOCK
    );
}

void KmsDisplay::restore_terminal()
{
    if (stdin_raw_) {
        tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios_);
        stdin_raw_ = false;
    }
    if (saved_stdin_flags_ >= 0) {
        fcntl(STDIN_FILENO, F_SETFL, saved_stdin_flags_);
        saved_stdin_flags_ = -1;
    }
}

void KmsDisplay::teardown()
{
    if (fd_ >= 0) {
        if (plane_id_) {
            drmModeSetPlane(
                fd_, plane_id_, crtc_id_, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0
            );
        }
        if (saved_crtc_) {
            drmModeSetCrtc(
                fd_,
                saved_crtc_->crtc_id,
                saved_crtc_->buffer_id,
                saved_crtc_->x,
                saved_crtc_->y,
                &connector_id_,
                1,
                &saved_crtc_->mode
            );
            drmModeFreeCrtc(saved_crtc_);
            saved_crtc_ = nullptr;
        }
        for (auto& buffer : frame_fbs_) {
            destroy_dumb(buffer);
        }
        destroy_dumb(primary_fb_);
        drmDropMaster(fd_);
        ::close(fd_);
        fd_ = -1;
    }
    restore_terminal();
}

int KmsDisplay::target_width() const
{
    return screen_width_;
}

int KmsDisplay::target_height() const
{
    return screen_height_;
}

bool KmsDisplay::show(const cv::Mat& image)
{
    if (!window_open_ || image.empty()) {
        return false;
    }

    if (!ensure_frame_buffers(image.cols, image.rows)) {
        throw std::runtime_error(
            "cannot allocate KMS frame buffers"
        );
    }

    DumbBuffer& fb = frame_fbs_[frame_index_];
    cv::Mat dst(
        image.rows,
        image.cols,
        CV_8UC4,
        fb.map,
        fb.pitch
    );
    cv::cvtColor(image, dst, cv::COLOR_BGR2BGRA);

    int dst_w = image.cols;
    int dst_h = image.rows;
    if (hw_scaling_ && fullscreen_) {
        const FittedImage fit = fit_image(
            image.cols,
            image.rows,
            screen_width_,
            screen_height_
        );
        dst_w = fit.width;
        dst_h = fit.height;
    }

    int rc = drmModeSetPlane(
        fd_,
        plane_id_,
        crtc_id_,
        fb.fb_id,
        0,
        (screen_width_ - dst_w) / 2,
        (screen_height_ - dst_h) / 2,
        dst_w,
        dst_h,
        0,
        0,
        static_cast<uint32_t>(image.cols) << 16,
        static_cast<uint32_t>(image.rows) << 16
    );

    if (rc != 0
        && hw_scaling_
        && (dst_w != image.cols || dst_h != image.rows)) {
        hw_scaling_ = false;
        std::fprintf(
            stderr,
            "[kms] plane scaling unsupported;"
            " showing native size\n"
        );
        rc = drmModeSetPlane(
            fd_,
            plane_id_,
            crtc_id_,
            fb.fb_id,
            0,
            (screen_width_ - image.cols) / 2,
            (screen_height_ - image.rows) / 2,
            image.cols,
            image.rows,
            0,
            0,
            static_cast<uint32_t>(image.cols) << 16,
            static_cast<uint32_t>(image.rows) << 16
        );
    }

    if (rc != 0) {
        throw std::runtime_error(
            std::string("drmModeSetPlane failed: ")
            + std::strerror(errno)
        );
    }

    frame_index_ ^= 1;
    poll_keys();
    return window_open_;
}

void KmsDisplay::poll_keys()
{
    char buffer[16];
    while (stdin_raw_) {
        const ssize_t n = ::read(STDIN_FILENO, buffer, sizeof(buffer));
        if (n <= 0) {
            break;
        }
        for (ssize_t i = 0; i < n; ++i) {
            if (key_requests_exit(buffer[i])) {
                window_open_ = false;
            }
        }
        if (n < static_cast<ssize_t>(sizeof(buffer))) {
            break;
        }
    }
}

bool KmsDisplay::poll()
{
    if (stdin_raw_) {
        fd_set set;
        FD_ZERO(&set);
        FD_SET(STDIN_FILENO, &set);
        timeval timeout{};
        timeout.tv_usec = POLL_TIMEOUT_MS * 1000;
        if (::select(
                STDIN_FILENO + 1,
                &set,
                nullptr,
                nullptr,
                &timeout) > 0) {
            poll_keys();
        }
    } else {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(POLL_TIMEOUT_MS)
        );
    }
    return window_open_;
}

}
}
