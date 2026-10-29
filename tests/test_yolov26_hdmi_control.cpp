#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolov26_hdmi_control.hpp"

using namespace yolov26;
using namespace yolov26::hdmi;

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

std::vector<std::string> strings(
    std::initializer_list<const char*> items)
{
    std::vector<std::string> result;
    for (const char* item : items) {
        result.emplace_back(item);
    }
    return result;
}

std::vector<char*> argv_of(std::vector<std::string>& items)
{
    std::vector<char*> argv;
    argv.reserve(items.size());
    for (std::string& item : items) {
        argv.push_back(item.data());
    }
    return argv;
}

ParseResult parse(std::initializer_list<const char*> items)
{
    std::vector<std::string> storage = strings(items);
    std::vector<char*> argv = argv_of(storage);
    return parse_options(
        static_cast<int>(argv.size()),
        argv.data()
    );
}

bool parse_ok(std::initializer_list<const char*> items,
              Options& out)
{
    try {
        const ParseResult parsed = parse(items);
        out = parsed.options;
        return !parsed.show_help;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

bool parse_fails(std::initializer_list<const char*> items)
{
    Options out;
    return !parse_ok(items, out);
}

void test_parsing()
{
    Options o;
    check(parse_ok({"app"}, o), "no arguments accepted");
    check(std::fabs(o.inference.confidence_threshold - 0.30)
              < 1e-4,
          "confidence default 0.30");
    check(o.inference.max_detections == 50,
          "max detections default 50");
    check(!o.inference.show_depth_roi,
          "show depth roi default off");
    check(o.inference.show_confidence,
          "show confidence default on");
    check(o.display == ":0", "display default :0");
    check(o.fullscreen, "fullscreen default on");
    check(!o.telemetry, "telemetry default off");
    check(!o.test_pattern, "test pattern default off");
    check(!o.duration_set, "duration unset default");

    check(o.renderer == "x11", "renderer default x11");
    check(o.kms_plane == "primary", "kms plane default primary");

    check(parse_ok({"app", "--renderer", "kms"}, o),
          "kms renderer accepted");
    check(o.kms_plane == "primary",
          "kms plane stays primary by default");
    check(parse_ok({"app", "--renderer", "kms",
                    "--kms-plane", "overlay"}, o),
          "kms overlay plane accepted");
    check(o.kms_plane == "overlay", "kms overlay plane stored");
    check(parse_ok({"app", "--kms-plane", "overlay",
                    "--renderer", "kms"}, o),
          "kms plane before renderer accepted");
    check(o.kms_plane == "overlay",
          "kms plane order independent");
    check(parse_fails({"app", "--renderer", "kms",
                       "--kms-plane", "cursor"}),
          "invalid kms plane rejected");
    check(parse_fails({"app", "--kms-plane", "overlay"}),
          "kms plane rejected with default x11 renderer");
    check(parse_fails({"app", "--renderer", "x11",
                       "--kms-plane", "primary"}),
          "kms plane rejected with explicit x11 renderer");
    check(parse_fails({"app", "--renderer", "kms",
                       "--kms-plane"}),
          "bare --kms-plane rejected");

    check(parse_ok({"app", "--display", ":1"}, o),
          "display override accepted");
    check(o.display == ":1", "display stored");

    check(parse_ok({"app", "--fullscreen", "false"}, o),
          "fullscreen false accepted");
    check(!o.fullscreen, "fullscreen disabled");
    check(parse_ok({"app", "--fullscreen", "true"}, o),
          "fullscreen true accepted");
    check(o.fullscreen, "fullscreen enabled");
    check(parse_fails({"app", "--fullscreen"}),
          "bare --fullscreen rejected");
    check(parse_fails({"app", "--fullscreen", "yes"}),
          "--fullscreen yes rejected");

    check(parse_ok({"app", "--telemetry", "true"}, o),
          "telemetry true accepted");
    check(o.telemetry, "telemetry enabled");
    check(parse_ok({"app", "--telemetry", "false"}, o),
          "telemetry false accepted");
    check(!o.telemetry, "telemetry stays off");
    check(parse_fails({"app", "--telemetry"}),
          "bare --telemetry rejected");
    check(parse_fails({"app", "--telemetry", "1"}),
          "--telemetry 1 rejected");

    check(parse_ok({"app", "--test-pattern"}, o),
          "test pattern accepted");
    check(o.test_pattern, "test pattern stored");
    check(DEFAULT_TEST_PATTERN_SECONDS == 10.0,
          "test pattern default 10 s");

    check(parse_ok({"app", "--show-depth-roi", "true"}, o),
          "show-depth-roi true accepted");
    check(o.inference.show_depth_roi,
          "show depth roi enabled");
    check(parse_fails({"app", "--show-depth-roi"}),
          "bare --show-depth-roi rejected");
    check(parse_fails({"app", "--show-depth-roi", "yes"}),
          "--show-depth-roi yes rejected");
    check(parse_ok({"app", "--show-confidence", "false"}, o),
          "show-confidence false accepted");
    check(!o.inference.show_confidence,
          "show confidence disabled");
    check(parse_fails({"app", "--show-confidence"}),
          "bare --show-confidence rejected");

    check(parse_ok({"app", "--duration-seconds", "2.5"}, o),
          "duration accepted");
    check(o.duration_set && o.duration_seconds == 2.5,
          "duration stored");
    check(parse_fails({"app", "--duration-seconds", "0"}),
          "zero duration rejected");
    check(parse_fails({"app", "--duration-seconds", "-1"}),
          "negative duration rejected");
    check(parse_fails({"app", "--duration-seconds", "nan"}),
          "nan duration rejected");
    check(parse_fails({"app", "--duration-seconds", "inf"}),
          "inf duration rejected");
    check(parse_fails({"app", "--duration-seconds", "1e30"}),
          "unrepresentable duration rejected");
    check(parse_fails({"app", "--duration-seconds"}),
          "bare --duration-seconds rejected");

    check(parse_ok({"app", "--model", "m.xmodel"}, o),
          "model accepted");
    check(o.inference.model_path == "m.xmodel", "model stored");
    check(parse_fails({"app", "--model", ""}),
          "empty model rejected");
    check(parse_ok({"app", "--confidence", "0.25"}, o),
          "confidence accepted");
    check(std::fabs(o.inference.confidence_threshold - 0.25)
              < 1e-4,
          "confidence stored");
    check(parse_fails({"app", "--confidence", "1.2"}),
          "confidence >1 rejected");
    check(parse_fails({"app", "--confidence", "nan"}),
          "nan confidence rejected");
    check(parse_fails({"app", "--confidence", "inf"}),
          "inf confidence rejected");
    check(parse_ok({"app", "--max-detections", "7"}, o),
          "max detections accepted");
    check(o.inference.max_detections == 7,
          "max detections stored");
    check(parse_fails({"app", "--max-detections", "0"}),
          "zero max detections rejected");

    check(parse_fails({"app", "--depth-width", "1280"}),
          "depth width alone rejected");
    check(parse_fails({"app", "--depth-height", "720"}),
          "depth height alone rejected");
    check(parse_ok({"app", "--depth-width", "1280",
                    "--depth-height", "720"}, o),
          "depth pair accepted");
    check(o.inference.depth_width == 1280
          && o.inference.depth_height == 720,
          "depth pair stored");

    check(parse_fails({"app", "--confidence", "-0.1"}),
          "negative confidence rejected");
    check(parse_fails({"app", "--max-detections", "-1"}),
          "negative max detections rejected");
    check(parse_fails({"app", "--duration-seconds", "abc"}),
          "nonnumeric duration rejected");
    check(parse_fails({"app", "--display", ""}),
          "empty display rejected");
    check(parse_fails({"app", "--fullscreen", "TRUE"}),
          "uppercase bool rejected");
    check(parse_fails({"app", "--test-pattern", "true"}),
          "valued flag rejected");

    check(parse_fails({"app", "--bogus"}),
          "unknown option rejected");
    check(parse_fails({"app", "extra"}),
          "unexpected positional rejected");
    check(parse_fails({"app", "--display"}),
          "bare --display rejected");
    check(parse_fails({"app", "--list-profiles"}),
          "--list-profiles rejected");

    {
        const ParseResult help = parse({"app", "--help"});
        check(help.show_help, "help flag");
    }
    {
        bool threw = false;
        try {
            parse({"app", "--help", "--fullscreen", "bogus"});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "help still validates bad values");
    }
}

void test_fit_image()
{
    {
        const FittedImage fit = fit_image(848, 480, 1024, 768);
        check(fit.width == 1024 && fit.height == 579,
              "848x480 into 1024x768 scales to 1024x579");
        check(fit.x == 0 && fit.y == 94,
              "1024x768 fit centered vertically");
    }
    {
        const FittedImage fit = fit_image(848, 480, 1920, 1080);
        check(fit.width == 1908 && fit.height == 1080,
              "848x480 into 1920x1080 scales to 1908x1080");
        check(fit.x == 6 && fit.y == 0,
              "1920x1080 fit centered horizontally");
    }
    {
        const FittedImage fit = fit_image(848, 480, 480, 848);
        check(fit.width == 480 && fit.height <= 848,
              "portrait canvas fits width");
        check(fit.y >= 0
              && fit.y + fit.height <= 848,
              "portrait fit in bounds");
    }
    {
        const FittedImage fit = fit_image(848, 480, 3, 3);
        check(fit.width >= 1 && fit.height >= 1
              && fit.width <= 3 && fit.height <= 3,
              "tiny canvas produces positive in-bounds fit");
    }
    {
        const FittedImage fit = fit_image(848, 480, 640, 480);
        check(fit.width == 640 && fit.height == 362
              && fit.x == 0 && fit.y == 59,
              "non-integer scale floors into bounds");
    }
    {
        const FittedImage fit = fit_image(640, 480, 640, 480);
        check(fit.width == 640 && fit.height == 480
              && fit.x == 0 && fit.y == 0,
              "same aspect fills canvas");
    }

    const int bad_cases[][4] = {
        {0, 480, 640, 480},
        {848, 0, 640, 480},
        {848, 480, 0, 480},
        {848, 480, 640, 0},
        {-1, 480, 640, 480},
        {848, 480, -640, 480},
    };
    for (const auto& bad : bad_cases) {
        bool threw = false;
        try {
            fit_image(bad[0], bad[1], bad[2], bad[3]);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "nonpositive geometry rejected");
    }
}

void test_display_telemetry()
{
    const auto t0 = Clock::now();
    DisplayTelemetry telemetry;

    check(!telemetry.snapshot_if_due(
              t0 + std::chrono::milliseconds(500)),
          "no snapshot before interval");

    telemetry.record(10.0, 20.0, 30.0, 5.0, 100.0);
    telemetry.record(20.0, 40.0, 50.0, 15.0, 300.0);

    const auto due = telemetry.snapshot_if_due(
        t0 + std::chrono::milliseconds(1200));
    check(due.has_value(), "snapshot due after interval");
    check(due->submissions == 2, "two submissions counted");
    check(std::fabs(due->fps - 2.0 / 1.2) < 0.01,
          "fps from windowed count");
    check(std::fabs(due->acquire_ms - 15.0) < 1e-9,
          "acquire mean");
    check(std::fabs(due->preprocess_ms - 30.0) < 1e-9,
          "preprocess mean");
    check(std::fabs(due->dpu_ms - 40.0) < 1e-9,
          "dpu mean");
    check(std::fabs(due->depth_ms - 10.0) < 1e-9,
          "depth mean");
    check(std::fabs(due->e2e_mean_ms - 200.0) < 1e-9,
          "e2e mean");
    check(std::fabs(due->e2e_max_ms - 300.0) < 1e-9,
          "e2e max");

    const auto next = telemetry.snapshot_if_due(
        t0 + std::chrono::milliseconds(2500));
    check(next.has_value(), "second window due");
    check(next->submissions == 0, "counters reset");
    check(next->e2e_max_ms == 0.0, "e2e max reset");
}

}

int main()
{
    test_parsing();
    test_fit_image();
    test_display_telemetry();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
