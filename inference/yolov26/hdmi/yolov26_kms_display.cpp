#include "yolov26_kms_display.hpp"

#include "yolov26_hdmi_control.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
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

[[noreturn]] void throw_errno(const std::string& what, int err)
{
    throw std::runtime_error(what + ": " + std::strerror(err));
}

bool read_prop(
    int fd,
    uint32_t object_id,
    uint32_t object_type,
    const char* name,
    uint32_t* prop_id,
    uint64_t* value)
{
    bool found = false;
    drmModeObjectProperties* props =
        drmModeObjectGetProperties(fd, object_id, object_type);
    if (!props) {
        return false;
    }
    for (uint32_t i = 0; i < props->count_props && !found; ++i) {
        drmModePropertyRes* prop =
            drmModeGetProperty(fd, props->props[i]);
        if (prop) {
            if (std::strcmp(prop->name, name) == 0) {
                found = true;
                if (prop_id) {
                    *prop_id = prop->prop_id;
                }
                if (value) {
                    *value = props->prop_values[i];
                }
            }
            drmModeFreeProperty(prop);
        }
    }
    drmModeFreeObjectProperties(props);
    return found;
}

bool plane_has_format(const drmModePlane* plane, uint32_t fourcc)
{
    for (uint32_t f = 0; f < plane->count_formats; ++f) {
        if (plane->formats[f] == fourcc) {
            return true;
        }
    }
    return false;
}

bool pick_crtc(
    int fd,
    const drmModeRes* res,
    const drmModeConnector* conn,
    uint32_t* crtc_id,
    int* crtc_index)
{
    uint32_t wanted = 0;
    if (conn->encoder_id) {
        drmModeEncoder* encoder =
            drmModeGetEncoder(fd, conn->encoder_id);
        if (encoder) {
            wanted = encoder->crtc_id;
            drmModeFreeEncoder(encoder);
        }
    }

    for (int c = 0; c < res->count_crtcs; ++c) {
        if (wanted && res->crtcs[c] == wanted) {
            *crtc_id = res->crtcs[c];
            *crtc_index = c;
            return true;
        }
    }

    for (int e = 0; e < conn->count_encoders; ++e) {
        drmModeEncoder* encoder =
            drmModeGetEncoder(fd, conn->encoders[e]);
        if (!encoder) {
            continue;
        }
        const uint32_t possible = encoder->possible_crtcs;
        drmModeFreeEncoder(encoder);
        for (int c = 0; c < res->count_crtcs; ++c) {
            if (possible & (1u << c)) {
                *crtc_id = res->crtcs[c];
                *crtc_index = c;
                return true;
            }
        }
    }
    return false;
}

} // namespace

namespace detail {

cv::Rect kms_content_rect(
    int image_width,
    int image_height,
    int screen_width,
    int screen_height,
    bool fullscreen)
{
    if (image_width <= 0 || image_height <= 0
        || screen_width <= 0 || screen_height <= 0) {
        throw std::invalid_argument(
            "kms_content_rect requires positive dimensions"
        );
    }

    if (!fullscreen
        && image_width <= screen_width
        && image_height <= screen_height) {
        return cv::Rect(
            (screen_width - image_width) / 2,
            (screen_height - image_height) / 2,
            image_width,
            image_height
        );
    }

    const FittedImage fit = fit_image(
        image_width, image_height, screen_width, screen_height
    );
    return cv::Rect(fit.x, fit.y, fit.width, fit.height);
}

cv::Rect compose_kms_frame(
    const cv::Mat& image,
    cv::Mat& destination,
    bool fullscreen,
    cv::Mat& scratch,
    KmsScaling scaling)
{
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::invalid_argument(
            "compose_kms_frame requires a non-empty CV_8UC3 image"
        );
    }
    if (destination.empty()
        || (destination.type() != CV_8UC4
            && destination.type() != CV_8UC2)) {
        throw std::invalid_argument(
            "compose_kms_frame requires a CV_8UC4 or CV_8UC2"
            " destination"
        );
    }

    const cv::Rect rect = kms_content_rect(
        image.cols,
        image.rows,
        destination.cols,
        destination.rows,
        fullscreen
    );

    const bool packed16 = destination.type() == CV_8UC2;
    const int code =
        packed16 ? cv::COLOR_BGR2BGR565 : cv::COLOR_BGR2BGRA;
    cv::Mat roi = destination(rect);

    if (rect.width == image.cols && rect.height == image.rows) {
        cv::cvtColor(image, roi, code);
    } else if (scaling == KmsScaling::Linear) {
        cv::resize(
            image,
            scratch,
            rect.size(),
            0,
            0,
            cv::INTER_LINEAR
        );
        cv::cvtColor(scratch, roi, code);
    } else {
        cv::cvtColor(image, scratch, code);
        const int word_type = packed16 ? CV_16UC1 : CV_32SC1;
        const cv::Mat source_words(
            scratch.rows,
            scratch.cols,
            word_type,
            scratch.data,
            scratch.step
        );
        cv::Mat roi_words(
            roi.rows,
            roi.cols,
            word_type,
            roi.data,
            roi.step
        );
        cv::resize(
            source_words,
            roi_words,
            roi.size(),
            0,
            0,
            cv::INTER_NEAREST
        );
    }
    return rect;
}

} // namespace detail

KmsDisplay::KmsDisplay(
    bool fullscreen,
    KmsPlane plane,
    KmsScaling scaling)
    : fullscreen_(fullscreen)
    , plane_(plane)
    , scaling_(scaling)
{
    try {
        open_card();
        select_planes();
        allocate_frame_buffers();
        setup_crtc();
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
            for (int c = 0;
                 c < res->count_connectors && connector_id_ == 0;
                 ++c) {
                drmModeConnector* conn =
                    drmModeGetConnector(fd, res->connectors[c]);
                if (!conn) {
                    continue;
                }
                if (conn->connection == DRM_MODE_CONNECTED
                    && conn->count_modes > 0
                    && pick_crtc(
                        fd, res, conn, &crtc_id_, &crtc_index_)) {
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

    if (drmSetClientCap(fd_, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1)
        != 0) {
        throw_errno("DRM universal planes unavailable", errno);
    }

    if (drmSetMaster(fd_) != 0) {
        throw std::runtime_error(
            "cannot acquire DRM master: stop the X server"
            " first (systemctl stop xserver-nodm)"
        );
    }
}

void KmsDisplay::select_planes()
{
    drmModePlaneRes* planes = drmModeGetPlaneResources(fd_);
    if (!planes) {
        throw std::runtime_error("no DRM planes available");
    }

    const uint32_t crtc_mask = 1u << crtc_index_;
    const uint32_t primary_format = plane_ == KmsPlane::Overlay
        ? DRM_FORMAT_ARGB8888
        : DRM_FORMAT_RGB565;

    for (uint32_t i = 0; i < planes->count_planes; ++i) {
        drmModePlane* plane = drmModeGetPlane(fd_, planes->planes[i]);
        if (!plane) {
            continue;
        }
        const uint32_t id = plane->plane_id;
        const bool usable = plane->possible_crtcs & crtc_mask;
        const bool has_primary_format =
            plane_has_format(plane, primary_format);
        const bool has_xr24 =
            plane_has_format(plane, DRM_FORMAT_XRGB8888);
        drmModeFreePlane(plane);
        if (!usable) {
            continue;
        }

        uint64_t type = 0;
        if (!read_prop(
                fd_, id, DRM_MODE_OBJECT_PLANE, "type", nullptr,
                &type)) {
            continue;
        }

        if (type == DRM_PLANE_TYPE_PRIMARY
            && !primary_plane_id_
            && has_primary_format) {
            primary_plane_id_ = id;
        } else if (type == DRM_PLANE_TYPE_OVERLAY
                   && !overlay_plane_id_
                   && (has_xr24 || plane_ == KmsPlane::Primary)) {
            overlay_plane_id_ = id;
        }
    }

    drmModeFreePlaneResources(planes);

    if (!primary_plane_id_) {
        throw std::runtime_error(
            plane_ == KmsPlane::Overlay
                ? "no primary DRM plane with ARGB8888 support"
                : "no primary DRM plane with RGB565 support"
        );
    }
    if (plane_ == KmsPlane::Overlay && !overlay_plane_id_) {
        throw std::runtime_error(
            "no usable DRM overlay plane with XR24 support"
        );
    }

    present_plane_id_ = plane_ == KmsPlane::Overlay
        ? overlay_plane_id_
        : primary_plane_id_;
}

void KmsDisplay::allocate_frame_buffers()
{
    const bool overlay = plane_ == KmsPlane::Overlay;
    frame_type_ = overlay ? CV_8UC4 : CV_8UC2;

    for (auto& buffer : frame_fbs_) {
        create_dumb(
            buffer,
            screen_width_,
            screen_height_,
            overlay ? 32 : 16,
            overlay ? DRM_FORMAT_XRGB8888 : DRM_FORMAT_RGB565
        );
        std::memset(buffer.map, 0, buffer.size);
    }
}

void KmsDisplay::save_blending()
{
    blending_.plane_id = primary_plane_id_;
    const bool alpha = read_prop(
        fd_, primary_plane_id_, DRM_MODE_OBJECT_PLANE, "alpha",
        &blending_.alpha_prop, &blending_.alpha
    );
    const bool galpha = read_prop(
        fd_, primary_plane_id_, DRM_MODE_OBJECT_PLANE, "g_alpha_en",
        &blending_.galpha_prop, &blending_.galpha
    );
    if (!alpha) {
        blending_.alpha_prop = 0;
    }
    if (!galpha) {
        blending_.galpha_prop = 0;
    }
    blending_.saved = alpha || galpha;

    std::fprintf(
        stderr,
        "[kms] graphics plane %u blending before: g_alpha_en=%s%llu"
        " alpha=%s%llu\n",
        primary_plane_id_,
        galpha ? "" : "n/a:",
        static_cast<unsigned long long>(blending_.galpha),
        alpha ? "" : "n/a:",
        static_cast<unsigned long long>(blending_.alpha)
    );
}

void KmsDisplay::apply_blending(uint64_t galpha, uint64_t alpha)
{
    if (!blending_.saved) {
        std::fprintf(
            stderr,
            "[kms] graphics plane has no alpha properties;"
            " leaving blending unchanged\n"
        );
        return;
    }

    if (blending_.galpha_prop
        && drmModeObjectSetProperty(
               fd_, primary_plane_id_, DRM_MODE_OBJECT_PLANE,
               blending_.galpha_prop, galpha) != 0) {
        throw_errno("set g_alpha_en failed", errno);
    }
    if (blending_.alpha_prop
        && drmModeObjectSetProperty(
               fd_, primary_plane_id_, DRM_MODE_OBJECT_PLANE,
               blending_.alpha_prop, alpha) != 0) {
        throw_errno("set alpha failed", errno);
    }

    uint64_t read_galpha = 0;
    uint64_t read_alpha = 0;
    read_prop(
        fd_, primary_plane_id_, DRM_MODE_OBJECT_PLANE, "g_alpha_en",
        nullptr, &read_galpha
    );
    read_prop(
        fd_, primary_plane_id_, DRM_MODE_OBJECT_PLANE, "alpha",
        nullptr, &read_alpha
    );
    std::fprintf(
        stderr,
        "[kms] graphics plane blending now: g_alpha_en=%llu"
        " alpha=%llu\n",
        static_cast<unsigned long long>(read_galpha),
        static_cast<unsigned long long>(read_alpha)
    );
}

void KmsDisplay::restore_blending()
{
    if (!blending_.saved) {
        return;
    }
    if (blending_.galpha_prop
        && drmModeObjectSetProperty(
               fd_, blending_.plane_id, DRM_MODE_OBJECT_PLANE,
               blending_.galpha_prop, blending_.galpha) != 0) {
        std::fprintf(
            stderr, "[kms] restore g_alpha_en failed: %s\n",
            std::strerror(errno)
        );
    }
    if (blending_.alpha_prop
        && drmModeObjectSetProperty(
               fd_, blending_.plane_id, DRM_MODE_OBJECT_PLANE,
               blending_.alpha_prop, blending_.alpha) != 0) {
        std::fprintf(
            stderr, "[kms] restore alpha failed: %s\n",
            std::strerror(errno)
        );
    }
    blending_.saved = false;
}

void KmsDisplay::setup_crtc()
{
    save_blending();

    uint32_t bound_fb = frame_fbs_[0].fb_id;
    if (plane_ == KmsPlane::Overlay) {
        create_dumb(
            primary_fb_,
            screen_width_,
            screen_height_,
            32,
            DRM_FORMAT_ARGB8888
        );
        std::memset(primary_fb_.map, 0, primary_fb_.size);
        bound_fb = primary_fb_.fb_id;
    }

    if (drmModeSetCrtc(
            fd_,
            crtc_id_,
            bound_fb,
            0,
            0,
            &connector_id_,
            1,
            &mode_) != 0) {
        throw_errno("drmModeSetCrtc failed", errno);
    }

    if (plane_ == KmsPlane::Overlay) {
        apply_blending(1, 0);
    } else {
        apply_blending(1, 255);
        frame_index_ = 1;
        if (overlay_plane_id_
            && drmModeSetPlane(
                   fd_, overlay_plane_id_, crtc_id_, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0) != 0) {
            std::fprintf(
                stderr, "[kms] overlay disable failed: %s\n",
                std::strerror(errno)
            );
        }
    }

    std::fprintf(
        stderr,
        "[kms] %s plane=%u crtc=%u connector=%u mode=%dx%d"
        " graphics_plane=%u overlay_plane=%u fb_pitch=%u"
        " scaling=%s\n",
        plane_ == KmsPlane::Overlay ? "overlay" : "primary",
        present_plane_id_,
        crtc_id_,
        connector_id_,
        screen_width_,
        screen_height_,
        primary_plane_id_,
        overlay_plane_id_,
        frame_fbs_[0].pitch,
        scaling_ == KmsScaling::Linear ? "linear" : "nearest"
    );
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
        throw_errno("CREATE_DUMB failed", errno);
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
        throw_errno("drmModeAddFB2 failed", errno);
    }

    drm_mode_map_dumb map{};
    map.handle = buffer.handle;
    if (drmIoctl(fd_, DRM_IOCTL_MODE_MAP_DUMB, &map) != 0) {
        throw_errno("MAP_DUMB failed", errno);
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
        const int err = errno;
        buffer.map = nullptr;
        throw_errno("mmap dumb buffer failed", err);
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
        if (overlay_plane_id_
            && drmModeSetPlane(
                   fd_, overlay_plane_id_, crtc_id_, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0) != 0) {
            std::fprintf(
                stderr, "[kms] overlay disable failed: %s\n",
                std::strerror(errno)
            );
        }
        restore_blending();
        if (saved_crtc_) {
            if (saved_crtc_->mode_valid && saved_crtc_->buffer_id) {
                if (drmModeSetCrtc(
                        fd_,
                        saved_crtc_->crtc_id,
                        saved_crtc_->buffer_id,
                        saved_crtc_->x,
                        saved_crtc_->y,
                        &connector_id_,
                        1,
                        &saved_crtc_->mode) != 0) {
                    std::fprintf(
                        stderr, "[kms] restore CRTC failed: %s\n",
                        std::strerror(errno)
                    );
                }
            }
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

    const cv::Rect rect = detail::kms_content_rect(
        image.cols,
        image.rows,
        screen_width_,
        screen_height_,
        fullscreen_
    );
    if (rect != content_rect_) {
        for (auto& buffer : frame_fbs_) {
            std::memset(buffer.map, 0, buffer.size);
        }
        content_rect_ = rect;
    }

    DumbBuffer& fb = frame_fbs_[frame_index_];
    cv::Mat dst(
        screen_height_,
        screen_width_,
        frame_type_,
        fb.map,
        fb.pitch
    );
    detail::compose_kms_frame(
        image, dst, fullscreen_, resize_scratch_, scaling_
    );

    if (drmModeSetPlane(
            fd_,
            present_plane_id_,
            crtc_id_,
            fb.fb_id,
            0,
            0,
            0,
            screen_width_,
            screen_height_,
            0,
            0,
            static_cast<uint32_t>(screen_width_) << 16,
            static_cast<uint32_t>(screen_height_) << 16) != 0) {
        throw_errno("drmModeSetPlane failed", errno);
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
