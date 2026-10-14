#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolov26_rpi_control.hpp"

using namespace yolov26;
using namespace yolov26::rpi;

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

void check_near(double actual, double expected, const char* name)
{
    ++checks;
    if (!std::isfinite(actual)
        || std::fabs(actual - expected) > 1e-9) {
        ++failures;
        std::fprintf(
            stderr,
            "FAIL: %s (expected %g, got %g)\n",
            name,
            expected,
            actual
        );
    }
}

double seconds_between(
    Clock::time_point from,
    Clock::time_point to)
{
    return std::chrono::duration<double>(to - from).count();
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
    return parse_sender_options(
        static_cast<int>(argv.size()),
        argv.data()
    );
}

bool parse_ok(std::initializer_list<const char*> items,
              SenderOptions& out)
{
    try {
        const ParseResult parsed = parse(items);
        out = parsed.options;
        return !parsed.show_help && !parsed.options.list_profiles;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

bool parse_fails(std::initializer_list<const char*> items)
{
    SenderOptions out;
    return !parse_ok(items, out);
}

DetectionAnnotation face_box()
{
    DetectionAnnotation a;
    a.class_id = 1;
    a.score = 0.5f;
    a.x = 10;
    a.y = 10;
    a.width = 20;
    a.height = 20;
    return a;
}

void test_parsing()
{
    SenderOptions o;
    check(parse_fails({"app"}), "missing --pi-host rejected");
    check(parse_ok({"app", "--pi-host", "192.168.1.5"}, o),
          "ipv4 host accepted");
    check_near(o.redis_interval_seconds, 1.0,
               "interval default 1.0");
    check_near(o.face_stale_seconds, 2.0, "stale default 2.0");
    check(o.redis_timeout_ms == 200, "timeout default 200ms");
    check(o.redis_channel == "face_detection",
          "channel default");
    check(o.redis_host == "192.168.1.5",
          "redis host defaults to pi host");
    check(o.video_port == 5000, "video port default");
    check(o.redis_port == 6379, "redis port default");
    check(o.bitrate_kbps == 4000, "bitrate default");
    check(o.encoder == "auto", "encoder default");
    check(!o.telemetry, "telemetry default off");
    check(!o.duration_set, "duration unset default");

    check(parse_ok({"app", "--pi-host", "pi.local"}, o),
          "dns host accepted");
    check(parse_ok({"app", "--pi-host", "pi.local."}, o),
          "trailing-dot dns accepted");
    check(parse_ok({"app", "--pi-host", "[fe80::1%eth0]"}, o),
          "scoped bracket ipv6 accepted");
    check(parse_ok({"app", "--pi-host", "fe80::1"}, o),
          "ipv6 accepted");
    check(parse_fails({"app", "--pi-host", "pi.local\n"}),
          "trailing newline rejected");
    check(parse_fails({"app", "--pi-host", "pi.local\r\n"}),
          "trailing crlf rejected");
    check(parse_fails({"app", "--pi-host", "fe80::1%"}),
          "empty scope rejected");
    check(parse_fails({"app", "--pi-host", "fe80::1%a%b"}),
          "double percent rejected");
    check(parse_fails({"app", "--pi-host", "999.1.1.1"}),
          "invalid ipv4 rejected");
    check(parse_fails({"app", "--pi-host", "a|b"}),
          "pipe char rejected");
    check(parse_fails({"app", "--pi-host", "-bad.local"}),
          "leading-dash label rejected");
    check(parse_fails({"app", "--pi-host", "a..b"}),
          "empty label rejected");
    check(parse_fails({"app", "--pi-host", ""}),
          "empty pi host rejected");

    check(parse_ok(
              {"app", "--pi-host", "h",
               "--redis-host", "redis.local"}, o),
          "redis host override");
    check(o.redis_host == "redis.local", "redis host stored");
    check(parse_fails(
              {"app", "--pi-host", "h",
               "--redis-host", "bad|host"}),
          "bad redis host rejected");
    check(parse_fails(
              {"app", "--pi-host", "h", "--redis-host", ""}),
          "empty redis host rejected");

    check(parse_fails({"app", "--pi-host", "h",
                       "--video-port", "0"}),
          "port 0 rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--video-port", "65536"}),
          "port 65536 rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--video-port", "1.5"}),
          "non-integer port rejected");
    check(parse_ok({"app", "--pi-host", "h",
                    "--video-port", "5002"}, o),
          "port 5002 accepted");
    check(o.video_port == 5002, "port stored");

    check(parse_fails({"app", "--pi-host", "h",
                       "--encoder", "h264"}),
          "bad encoder rejected");
    check(parse_ok({"app", "--pi-host", "h",
                    "--encoder", "omx"}, o),
          "encoder omx");
    check(o.encoder == "omx", "encoder stored");

    check(parse_fails({"app", "--pi-host", "h",
                       "--bitrate-kbps", "0"}),
          "zero bitrate rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--bitrate-kbps", "-5"}),
          "negative bitrate rejected");

    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-interval-seconds", "0"}),
          "zero interval rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-interval-seconds", "0.05"}),
          "interval below 0.1 rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-interval-seconds", "nan"}),
          "nan interval rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-interval-seconds", "inf"}),
          "inf interval rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-interval-seconds", "1e30"}),
          "unrepresentable interval rejected");
    check(parse_ok({"app", "--pi-host", "h",
                    "--redis-interval-seconds", "0.5"}, o),
          "interval 0.5 accepted");
    check_near(o.redis_interval_seconds, 0.5,
               "interval stored");

    check(parse_fails({"app", "--pi-host", "h",
                       "--face-stale-seconds", "-1"}),
          "negative stale rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--face-stale-seconds", "0"}),
          "zero stale rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-timeout-ms", "0"}),
          "zero timeout rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-timeout-ms", "10001"}),
          "timeout >10000 rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-timeout-ms", "1.5"}),
          "noninteger timeout rejected");
    check(parse_ok({"app", "--pi-host", "h",
                    "--redis-timeout-ms", "5000"}, o),
          "timeout 5000 accepted");
    check(o.redis_timeout_ms == 5000, "timeout stored");

    check(parse_fails({"app", "--pi-host", "h",
                       "--duration-seconds", "abc"}),
          "bad duration rejected");
    check(parse_ok({"app", "--pi-host", "h",
                    "--duration-seconds", "2.5"}, o),
          "duration accepted");
    check(o.duration_set && o.duration_seconds == 2.5,
          "duration stored");

    check(parse_fails({"app", "--pi-host", "h",
                       "--redis-channel", ""}),
          "empty channel rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--model", ""}),
          "empty model rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--confidence", "1.2"}),
          "confidence >1 rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--confidence", "nan"}),
          "nan confidence rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--max-detections", "0"}),
          "zero max detections rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--depth-width", "1280"}),
          "depth width alone rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--depth-width", "0",
                       "--depth-height", "720"}),
          "depth width 0 rejected");
    check(parse_ok({"app", "--pi-host", "h",
                    "--depth-width", "1280",
                    "--depth-height", "720"}, o),
          "depth pair accepted");
    check(o.inference.depth_width == 1280
          && o.inference.depth_height == 720,
          "depth pair stored");
    check(o.inference.confidence_threshold
              == DEFAULT_CONFIDENCE_THRESHOLD,
          "confidence default");
    check(o.inference.max_detections == 300,
          "max detections default");
    check(o.inference.model_path
          == "yolov26n_facehuman_kria.xmodel",
          "model default");

    check(parse_ok({"app", "--pi-host", "h",
                    "--telemetry", "true"}, o),
          "telemetry true accepted");
    check(o.telemetry, "telemetry enabled");
    check(parse_ok({"app", "--pi-host", "h",
                    "--telemetry", "false"}, o),
          "telemetry false accepted");
    check(!o.telemetry, "telemetry stays off");
    check(parse_fails({"app", "--pi-host", "h",
                       "--telemetry"}),
          "bare --telemetry rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--telemetry", "yes"}),
          "--telemetry yes rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--no-telemetry"}),
          "old --no-telemetry rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--report-interval-seconds", "0.3"}),
          "old interval spelling rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--stale-seconds", "1.5"}),
          "old stale spelling rejected");
    check(parse_fails({"app", "--pi-host", "h",
                       "--timeout-seconds", "5"}),
          "old timeout spelling rejected");

    check(parse_fails({"app", "--pi-host", "h",
                       "--bogus"}),
          "unknown option rejected");
    check(parse_fails({"app", "--pi-host"}),
          "missing value rejected");

    {
        const ParseResult help = parse({"app", "--help"});
        check(help.show_help, "help flag");
    }
    {
        bool threw = false;
        try {
            parse({"app", "--help",
                   "--pi-host", "bad|host"});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "help still validates bad values");
    }
    {
        bool threw = false;
        try {
            parse({"app", "--list-profiles",
                   "--encoder", "bogus"});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "list-profiles still validates values");
    }
    {
        const ParseResult lp = parse({"app", "--list-profiles"});
        check(lp.options.list_profiles && !lp.show_help,
              "list-profiles parses without pi-host");
    }
}

void test_face_predicate()
{
    const int w = 848;
    const int h = 480;
    std::vector<DetectionAnnotation> none;
    check(!face_detected(none, w, h), "no annotations false");

    DetectionAnnotation person;
    person.class_id = 0;
    person.score = 0.9f;
    person.x = 5;
    person.y = 5;
    person.width = 50;
    person.height = 50;
    check(!face_detected({person}, w, h),
          "person class not face");

    check(face_detected({face_box()}, w, h),
          "valid face true");

    {
        DetectionAnnotation a = face_box();
        a.score = -0.1f;
        check(!face_detected({a}, w, h), "negative score");
    }
    {
        DetectionAnnotation a = face_box();
        a.score = 1.01f;
        check(!face_detected({a}, w, h), "score >1");
    }
    {
        DetectionAnnotation a = face_box();
        a.score = std::nanf("");
        check(!face_detected({a}, w, h), "nan score");
    }
    {
        DetectionAnnotation a = face_box();
        a.x = -1;
        check(!face_detected({a}, w, h), "negative x");
    }
    {
        DetectionAnnotation a = face_box();
        a.y = -1;
        check(!face_detected({a}, w, h), "negative y");
    }
    {
        DetectionAnnotation a = face_box();
        a.width = 0;
        check(!face_detected({a}, w, h), "zero width");
    }
    {
        DetectionAnnotation a = face_box();
        a.height = -3;
        check(!face_detected({a}, w, h), "negative height");
    }
    {
        DetectionAnnotation a = face_box();
        a.x = w;
        a.width = 1;
        check(!face_detected({a}, w, h), "x beyond width");
    }
    {
        DetectionAnnotation a = face_box();
        a.y = h;
        a.height = 1;
        check(!face_detected({a}, w, h), "y beyond height");
    }
    {
        DetectionAnnotation a = face_box();
        a.x = w - 20;
        a.width = 20;
        a.y = h - 20;
        a.height = 20;
        check(face_detected({a}, w, h),
              "corner box at exact edge ok");
    }
    {
        DetectionAnnotation a = face_box();
        a.distance = -1.0f;
        check(face_detected({a}, w, h),
              "distance ignored");
    }
    {
        DetectionAnnotation a = face_box();
        a.score = std::numeric_limits<float>::infinity();
        check(!face_detected({a}, w, h), "inf score");
    }
}

void test_payload_and_state()
{
    check(face_payload(true) == "{\"face_detected\":true}",
          "true payload exact");
    check(face_payload(false) == "{\"face_detected\":false}",
          "false payload exact");

    LatestFaceState state;
    check(!state.snapshot(), "no initial state");

    const Clock::time_point t0 = Clock::now();
    state.update(true, t0);
    const auto snap = state.snapshot();
    check(snap && snap->first && snap->second == t0,
          "snapshot has state+timestamp");

    check(state_fresh(t0, t0, 2.0), "age 0 fresh");
    check(state_fresh(
              t0,
              t0 + std::chrono::milliseconds(1999),
              2.0),
          "inside window fresh");
    check(!state_fresh(
              t0,
              t0 + std::chrono::milliseconds(2000),
              2.0),
          "exact boundary stale");
    check(!state_fresh(
              t0,
              t0 + std::chrono::milliseconds(2500),
              2.0),
          "past window stale");
    check(!state_fresh(
              t0 + std::chrono::milliseconds(100),
              t0,
              2.0),
          "negative age stale");

    state.update(false, t0);
    check(state.snapshot()->first == false,
          "update replaces");
    state.invalidate();
    check(!state.snapshot(), "invalidate clears");
}

void test_deadlines_and_backoff()
{
    const Clock::time_point origin{};

    const Clock::time_point first = next_report_deadline(
        origin, 0.5, origin
    );
    check_near(seconds_between(origin, first), 0.5,
               "first deadline at interval");

    const Clock::time_point second = next_report_deadline(
        origin, 0.5, first
    );
    check_near(seconds_between(origin, second), 1.0,
               "second deadline anchored");

    const Clock::time_point just_before = next_report_deadline(
        origin, 0.5,
        origin + Clock::duration(std::chrono::milliseconds(499))
    );
    check_near(seconds_between(origin, just_before), 0.5,
               "just before tick keeps deadline");

    const Clock::time_point just_after = next_report_deadline(
        origin, 0.5,
        origin + Clock::duration(std::chrono::milliseconds(501))
    );
    check_near(seconds_between(origin, just_after), 1.0,
               "just after tick advances");

    const Clock::time_point after_gap = next_report_deadline(
        origin,
        0.5,
        origin + std::chrono::milliseconds(2600)
    );
    check_near(seconds_between(origin, after_gap), 3.0,
               "missed ticks skipped");

    const Clock::time_point tenth = next_report_deadline(
        origin, 0.1,
        origin + std::chrono::milliseconds(900)
    );
    check_near(seconds_between(origin, tenth), 1.0,
               "0.1 interval ticks");

    const Clock::time_point now = Clock::now();
    const Clock::time_point far = next_report_deadline(
        now, MAX_SECONDS, now
    );
    check(far > now, "max interval deadline future");

    const Clock::time_point saturated = next_report_deadline(
        now - std::chrono::hours(24),
        MAX_SECONDS,
        now
    );
    check(saturated == Clock::time_point::max()
              || seconds_between(now, saturated) > 0,
          "overflow saturates or stays future");

    bool threw = false;
    try {
        next_report_deadline(origin, 1e-12, origin);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "sub-nanosecond interval rejected");

    check_near(reconnect_backoff_seconds(0), 1.0, "backoff 0");
    check_near(reconnect_backoff_seconds(1), 2.0, "backoff 1");
    check_near(reconnect_backoff_seconds(2), 4.0, "backoff 2");
    check_near(reconnect_backoff_seconds(3), 5.0, "backoff 3");
    check_near(reconnect_backoff_seconds(4), 5.0, "backoff cap");
    check_near(reconnect_backoff_seconds(10), 5.0,
               "backoff capped");
}

void test_telemetry()
{
    const Clock::time_point origin{};
    SenderTelemetry telemetry(origin);

    telemetry.record_inference(10, 20, 30, 40);
    telemetry.record_inference(30, 40, 50, 60);
    telemetry.record_submission(6, 8, 100);
    telemetry.record_replacement();
    telemetry.record_replacement();

    const auto early = telemetry.snapshot_if_due(
        3,
        origin + std::chrono::milliseconds(500)
    );
    check(!early, "early snapshot empty");

    const auto first = telemetry.snapshot_if_due(
        3,
        origin + std::chrono::seconds(2)
    );
    check(first && first->available, "first snapshot available");
    check_near(first->window_seconds, 2.0, "window 2s");
    check_near(first->inference_fps, 1.0, "inference 1/s");
    check_near(first->appsrc_submissions_per_second, 0.5,
               "submissions 0.5/s");
    check_near(first->encoded_au_per_second, 1.5,
               "AUs 1.5/s");
    check(first->replaced_frames == 2, "replaced 2");
    const double expected_means[7] = {20, 30, 40, 50, 6, 8, 100};
    for (int i = 0; i < 7; ++i) {
        check_near(first->mean_ms[i], expected_means[i],
                   "mean first window");
    }

    telemetry.record_inference(2, 4, 6, 8);

    const auto second = telemetry.snapshot_if_due(
        5,
        origin + std::chrono::seconds(3)
    );
    check(second && second->available, "second snapshot");
    check_near(second->inference_fps, 1.0, "inference 1/s 2nd");
    check_near(second->appsrc_submissions_per_second, 0.0,
               "submissions 0 2nd");
    check_near(second->encoded_au_per_second, 2.0,
               "AUs 2/s 2nd");
    check(second->replaced_frames == 2,
          "replaced cumulative 2");
    const double expected2[7] = {2, 4, 6, 8, 0, 0, 0};
    for (int i = 0; i < 7; ++i) {
        check_near(second->mean_ms[i], expected2[i],
                   "mean second window");
    }

    const auto third = telemetry.snapshot_if_due(
        5,
        origin + std::chrono::seconds(4)
    );
    check(third && third->available, "third snapshot");
    check_near(third->inference_fps, 0.0, "inference 0 3rd");
    check_near(third->appsrc_submissions_per_second, 0.0,
               "submissions 0 3rd");
    check_near(third->encoded_au_per_second, 0.0,
               "AUs 0 3rd");
    check(third->replaced_frames == 2,
          "replaced cumulative still 2");
    for (int i = 0; i < 7; ++i) {
        check_near(third->mean_ms[i], 0.0, "mean third window");
    }

    bool threw = false;
    try {
        telemetry.snapshot_if_due(
            4,
            origin + std::chrono::seconds(5)
        );
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw, "regression throws logic_error");
}

}

int main()
{
    test_parsing();
    test_face_predicate();
    test_payload_and_state();
    test_deadlines_and_backoff();
    test_telemetry();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
