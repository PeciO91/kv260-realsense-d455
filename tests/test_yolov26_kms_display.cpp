#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolov26_kms_display.hpp"

using namespace yolov26::hdmi;
using namespace yolov26::hdmi::detail;

namespace {

int failures = 0;
int checks = 0;

void check(bool condition, const char* name)
{
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", name);
    }
}

bool rect_is(const cv::Rect& r, int x, int y, int w, int h)
{
    return r.x == x && r.y == y && r.width == w && r.height == h;
}

template <typename Fn>
bool throws_invalid(Fn fn)
{
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_content_rect()
{
    check(rect_is(kms_content_rect(848, 480, 1920, 1080, false),
                  536, 300, 848, 480),
          "native 848x480 centered in 1920x1080");
    check(rect_is(kms_content_rect(848, 480, 1920, 1080, true),
                  6, 0, 1908, 1080),
          "fullscreen fit 1908x1080 at x=6");
    check(rect_is(kms_content_rect(848, 480, 1024, 768, true),
                  0, 94, 1024, 579),
          "fullscreen fit 1024x579 at y=94");
    check(rect_is(kms_content_rect(2000, 1200, 1920, 1080, false),
                  60, 0, 1800, 1080),
          "oversized native input fit down");
    check(rect_is(kms_content_rect(1920, 1080, 1920, 1080, false),
                  0, 0, 1920, 1080),
          "exact size native fills screen");
    check(throws_invalid([] {
              kms_content_rect(0, 480, 1920, 1080, false);
          }),
          "zero image width rejected");
    check(throws_invalid([] {
              kms_content_rect(848, 480, 1920, 0, true);
          }),
          "zero screen height rejected");
}

cv::Mat solid_bgr(int width, int height, cv::Scalar color)
{
    return cv::Mat(height, width, CV_8UC3, color);
}

void test_compose_xr24()
{
    const int sw = 64;
    const int sh = 48;
    const int pitch = sw * 4 + 32;
    std::vector<unsigned char> memory(pitch * sh, 0xAB);
    cv::Mat dst(sh, sw, CV_8UC4, memory.data(), pitch);
    for (int y = 0; y < sh; ++y) {
        std::memset(memory.data() + y * pitch, 0, sw * 4);
    }

    cv::Mat image = solid_bgr(16, 8, cv::Scalar(10, 20, 30));
    cv::Mat before = image.clone();
    cv::Mat scratch;
    const cv::Rect rect = compose_kms_frame(image, dst, false, scratch);

    check(rect_is(rect, 24, 20, 16, 8), "xr24 native rect centered");
    const cv::Vec4b inside = dst.at<cv::Vec4b>(24, 30);
    check(inside[0] == 10 && inside[1] == 20 && inside[2] == 30,
          "xr24 pixel carries BGR order");
    check(dst.at<cv::Vec4b>(0, 0) == cv::Vec4b(0, 0, 0, 0),
          "xr24 margin untouched");
    check(dst.at<cv::Vec4b>(19, 30) == cv::Vec4b(0, 0, 0, 0),
          "xr24 row above content untouched");
    bool padding_intact = true;
    for (int y = 0; y < sh; ++y) {
        for (int b = sw * 4; b < pitch; ++b) {
            if (memory[y * pitch + b] != 0xAB) {
                padding_intact = false;
            }
        }
    }
    check(padding_intact, "xr24 row padding sentinel unchanged");
    check(cv::norm(image, before, cv::NORM_INF) == 0,
          "source image unchanged");
    check(scratch.empty(), "native placement does not resize");
}

void test_compose_fullscreen_and_rgb565()
{
    const int sw = 64;
    const int sh = 48;
    const int pitch = sw * 2 + 16;
    std::vector<unsigned char> memory(pitch * sh, 0);
    cv::Mat dst(sh, sw, CV_8UC2, memory.data(), pitch);

    cv::Mat image = solid_bgr(16, 8, cv::Scalar(255, 255, 255));
    cv::Mat scratch;
    const cv::Rect rect = compose_kms_frame(image, dst, true, scratch);

    check(rect_is(rect, 0, 8, 64, 32), "rgb565 fullscreen fit rect");
    check(scratch.cols == 64 && scratch.rows == 32,
          "fullscreen resizes into scratch");
    check(dst.at<cv::Vec2b>(24, 32) == cv::Vec2b(0xFF, 0xFF),
          "rgb565 white pixel");
    check(dst.at<cv::Vec2b>(0, 0) == cv::Vec2b(0, 0),
          "rgb565 margin untouched");

    const unsigned char* scratch_data = scratch.data;
    compose_kms_frame(image, dst, true, scratch);
    check(scratch.data == scratch_data, "scratch storage reused");
}

void test_compose_rejects_invalid()
{
    std::vector<unsigned char> memory(64 * 48 * 4, 0);
    cv::Mat dst(48, 64, CV_8UC4, memory.data());
    cv::Mat scratch;
    cv::Mat empty;
    cv::Mat gray(8, 16, CV_8UC1, cv::Scalar(1));
    cv::Mat image = solid_bgr(16, 8, cv::Scalar(1, 2, 3));
    cv::Mat bad_dst(48, 64, CV_8UC3);
    cv::Mat empty_dst;

    check(throws_invalid([&] {
              compose_kms_frame(empty, dst, false, scratch);
          }),
          "empty input rejected");
    check(throws_invalid([&] {
              compose_kms_frame(gray, dst, false, scratch);
          }),
          "non-BGR input rejected");
    check(throws_invalid([&] {
              compose_kms_frame(image, bad_dst, false, scratch);
          }),
          "unsupported destination type rejected");
    check(throws_invalid([&] {
              compose_kms_frame(image, empty_dst, false, scratch);
          }),
          "empty destination rejected");
}

} // namespace

int main()
{
    test_content_rect();
    test_compose_xr24();
    test_compose_fullscreen_and_rgb565();
    test_compose_rejects_invalid();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
