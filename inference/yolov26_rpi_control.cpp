#include "yolov26_rpi_control.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include <arpa/inet.h>

namespace yolov26 {
namespace rpi {

namespace {

bool valid_host_characters(const std::string& host)
{
    for (char c : host) {
        const unsigned char u = static_cast<unsigned char>(c);
        const bool ok = std::isalnum(u)
            || c == '.' || c == ':' || c == '%'
            || c == '[' || c == ']'
            || c == '_' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

bool valid_ipv6_scope(const std::string& scope)
{
    if (scope.empty() || scope.size() > 15) {
        return false;
    }
    for (char c : scope) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!std::isalnum(u) && c != '.' && c != '_' && c != '-') {
            return false;
        }
    }
    return true;
}

bool valid_dns_name(const std::string& host)
{
    std::string name = host;
    if (name.size() > 1 && name.back() == '.') {
        name.pop_back();
    }
    if (name.empty() || name.size() > 253
        || name.front() == '.' || name.back() == '.') {
        return false;
    }

    std::size_t start = 0;

    while (start < name.size()) {
        const std::size_t dot = name.find('.', start);
        const std::size_t end =
            dot == std::string::npos ? name.size() : dot;
        const std::size_t length = end - start;

        if (length == 0 || length > 63) {
            return false;
        }

        if (name[start] == '-' || name[end - 1] == '-') {
            return false;
        }

        for (std::size_t i = start; i < end; ++i) {
            const unsigned char u =
                static_cast<unsigned char>(name[i]);
            if (!std::isalnum(u) && name[i] != '-' && name[i] != '_') {
                return false;
            }
        }

        start = end + 1;
    }

    return true;
}

bool validate_host(const std::string& host)
{
    if (host.empty() || host.size() > 253
        || !valid_host_characters(host)) {
        return false;
    }

    in_addr ipv4{};
    if (inet_pton(AF_INET, host.c_str(), &ipv4) == 1) {
        return true;
    }

    const bool looks_ipv4 =
        std::all_of(host.begin(), host.end(), [](char c) {
            return std::isdigit(
                static_cast<unsigned char>(c)
            ) || c == '.';
        });

    if (looks_ipv4) {
        return false;
    }

    std::string candidate = host;

    if (candidate.front() == '[') {
        if (candidate.back() != ']') {
            return false;
        }
        candidate = candidate.substr(1, candidate.size() - 2);
        if (candidate.empty()) {
            return false;
        }
    }

    const std::size_t percent = candidate.find('%');
    std::string address = candidate;

    if (percent != std::string::npos) {
        if (candidate.find('%', percent + 1) != std::string::npos) {
            return false;
        }
        address = candidate.substr(0, percent);
        const std::string scope = candidate.substr(percent + 1);
        if (!valid_ipv6_scope(scope)) {
            return false;
        }
    }

    if (address.find(':') != std::string::npos) {
        in6_addr ipv6{};
        return inet_pton(AF_INET6, address.c_str(), &ipv6) == 1;
    }

    return valid_dns_name(host);
}

int parse_int(
    const char* name,
    const std::string& text,
    long long minimum,
    long long maximum)
{
    if (text.empty()) {
        throw std::invalid_argument(
            std::string("invalid ") + name + " '" + text + "'"
        );
    }

    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(text.c_str(), &end, 10);

    if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        throw std::invalid_argument(
            std::string("invalid integer ") + name
            + " '" + text + "'"
        );
    }

    if (value < minimum || value > maximum) {
        throw std::invalid_argument(
            std::string(name) + " out of range '" + text + "'"
        );
    }

    return static_cast<int>(value);
}

double parse_seconds(
    const char* name,
    const std::string& text,
    double minimum)
{
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text.c_str(), &end);

    if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
        throw std::invalid_argument(
            std::string("invalid number ") + name
            + " '" + text + "'"
        );
    }

    if (!std::isfinite(value) || value < minimum) {
        throw std::invalid_argument(
            std::string(name) + " out of range '" + text
            + "' (min " + std::to_string(minimum) + " s)"
        );
    }

    if (value > MAX_SECONDS) {
        throw std::invalid_argument(
            std::string(name) + " too large '" + text
            + "' (max " + std::to_string(MAX_SECONDS) + " s)"
        );
    }

    return value;
}

void validate_common(const SenderOptions& options)
{
    if (!options.pi_host.empty()
        && !validate_host(options.pi_host)) {
        throw std::invalid_argument(
            "invalid host '" + options.pi_host + "'"
        );
    }

    if (options.redis_host_set
        && !validate_host(options.redis_host)) {
        throw std::invalid_argument(
            "invalid host '" + options.redis_host + "'"
        );
    }

    if ((options.inference.depth_width == 0)
        != (options.inference.depth_height == 0)) {
        throw std::invalid_argument(
            "--depth-width and --depth-height must be set together"
        );
    }
}

}

void print_sender_usage(const char* argv0)
{
    std::cout
        << "Usage: " << argv0
        << " --pi-host HOST [options]" << std::endl
        << std::endl
        << "Send annotated D455 + DPU YOLOv26 video to a Pi over"
        << " RTP/UDP" << std::endl
        << "and publish face detection state on Redis Pub/Sub."
        << std::endl
        << std::endl
        << "Options:" << std::endl
        << "  --pi-host HOST            RTP destination host"
        << " (required)" << std::endl
        << "  --video-port PORT         UDP destination port 1..65535"
        << " (default " << DEFAULT_VIDEO_PORT << ")" << std::endl
        << "  --encoder auto|omx|x264   H.264 encoder "
        << "(default auto)" << std::endl
        << "  --bitrate-kbps KBPS       H.264 bitrate >0 kbps"
        << " (default " << DEFAULT_BITRATE_KBPS << ")" << std::endl
        << "  --redis-host HOST         Redis host"
        << " (default: same as --pi-host)" << std::endl
        << "  --redis-port PORT         Redis port 1..65535"
        << " (default " << DEFAULT_REDIS_PORT << ")" << std::endl
        << "  --redis-channel NAME      Pub/Sub channel"
        << " (default face_detection)" << std::endl
        << "  --redis-interval-seconds SECONDS  Face report"
        << " interval " << MIN_REDIS_INTERVAL_SECONDS << ".."
        << MAX_SECONDS << " s" << std::endl
        << "                            (default "
        << DEFAULT_REDIS_INTERVAL_SECONDS << ")" << std::endl
        << "  --face-stale-seconds SECONDS  Suppress reports when"
        << " the latest" << std::endl
        << "                            result is >= this age"
        << " (default " << DEFAULT_FACE_STALE_SECONDS
        << ")" << std::endl
        << "  --redis-timeout-ms MS     Connect/read/write deadline"
        << " 1..10000 ms" << std::endl
        << "                            (default "
        << DEFAULT_REDIS_TIMEOUT_MS << ")" << std::endl
        << "  --duration-seconds SECONDS  Stop after SECONDS of"
        << " main capture" << std::endl
        << "                            (default: run until SIGINT)"
        << std::endl
        << "  --model PATH              Model xmodel path"
        << " (default " << DEFAULT_MODEL_PATH << ")" << std::endl
        << "  --confidence VALUE        Detection confidence 0..1"
        << " (default " << DEFAULT_CONFIDENCE_THRESHOLD << ")"
        << std::endl
        << "  --max-detections COUNT    Maximum detections per"
        << " frame (default " << DEFAULT_MAX_DETECTIONS << ")"
        << std::endl
        << "  --depth-width WIDTH       Depth stream width"
        << std::endl
        << "  --depth-height HEIGHT     Depth stream height"
        << std::endl
        << "  --telemetry true|false    Enable metrics"
        << " (default false)" << std::endl
        << "  --list-profiles           List supported camera"
        << " profiles and exit" << std::endl
        << std::endl
        << "Video is application/x-rtp H.264 PT "
        << RTP_PAYLOAD_TYPE
        << ", 90000 Hz, MTU " << RTP_MTU << " byte."
        << std::endl
        << "Redis PUBLISH payloads are exactly"
        << " {\"face_detected\":true} or false. Duration starts"
        << " after" << std::endl
        << "encoder preflight, camera start and warmup complete."
        << std::endl;
}

ParseResult parse_sender_options(int argc, char* argv[])
{
    ParseResult result;
    SenderOptions& options = result.options;
    bool pi_host_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);

        auto require_value = [&](const char* name) {
            if (++i >= argc) {
                throw std::invalid_argument(
                    std::string("Missing value for ") + name
                );
            }
            return std::string(argv[i]);
        };

        if (argument == "--pi-host") {
            options.pi_host = require_value("--pi-host");
            pi_host_set = true;
        } else if (argument == "--video-port") {
            options.video_port = parse_int(
                "--video-port",
                require_value("--video-port"),
                1,
                65535
            );
        } else if (argument == "--encoder") {
            options.encoder = require_value("--encoder");
            if (options.encoder != "auto"
                && options.encoder != "omx"
                && options.encoder != "x264") {
                throw std::invalid_argument(
                    "--encoder must be auto, omx or x264"
                );
            }
        } else if (argument == "--bitrate-kbps") {
            options.bitrate_kbps = parse_int(
                "--bitrate-kbps",
                require_value("--bitrate-kbps"),
                1,
                std::numeric_limits<int>::max()
            );
        } else if (argument == "--redis-host") {
            options.redis_host = require_value("--redis-host");
            options.redis_host_set = true;
        } else if (argument == "--redis-port") {
            options.redis_port = parse_int(
                "--redis-port",
                require_value("--redis-port"),
                1,
                65535
            );
        } else if (argument == "--redis-channel") {
            options.redis_channel =
                require_value("--redis-channel");
            if (options.redis_channel.empty()) {
                throw std::invalid_argument(
                    "--redis-channel must not be empty"
                );
            }
        } else if (argument == "--redis-interval-seconds") {
            options.redis_interval_seconds = parse_seconds(
                "--redis-interval-seconds",
                require_value("--redis-interval-seconds"),
                MIN_REDIS_INTERVAL_SECONDS
            );
        } else if (argument == "--face-stale-seconds") {
            options.face_stale_seconds = parse_seconds(
                "--face-stale-seconds",
                require_value("--face-stale-seconds"),
                1e-9
            );
        } else if (argument == "--redis-timeout-ms") {
            options.redis_timeout_ms = parse_int(
                "--redis-timeout-ms",
                require_value("--redis-timeout-ms"),
                1,
                10000
            );
        } else if (argument == "--duration-seconds") {
            options.duration_seconds = parse_seconds(
                "--duration-seconds",
                require_value("--duration-seconds"),
                1e-9
            );
            options.duration_set = true;
        } else if (argument == "--model") {
            options.inference.model_path =
                require_value("--model");
            if (options.inference.model_path.empty()) {
                throw std::invalid_argument(
                    "--model must not be empty"
                );
            }
        } else if (argument == "--confidence") {
            char* end = nullptr;
            const std::string text =
                require_value("--confidence");
            const float value = std::strtof(text.c_str(), &end);
            if (end == text.c_str() || *end != '\0'
                || !std::isfinite(value)
                || value < 0.0f || value > 1.0f) {
                throw std::invalid_argument(
                    "--confidence must be a finite number 0..1"
                );
            }
            options.inference.confidence_threshold = value;
        } else if (argument == "--max-detections") {
            options.inference.max_detections = parse_int(
                "--max-detections",
                require_value("--max-detections"),
                1,
                std::numeric_limits<int>::max()
            );
        } else if (argument == "--depth-width") {
            options.inference.depth_width = parse_int(
                "--depth-width",
                require_value("--depth-width"),
                1,
                4096
            );
        } else if (argument == "--depth-height") {
            options.inference.depth_height = parse_int(
                "--depth-height",
                require_value("--depth-height"),
                1,
                4096
            );
        } else if (argument == "--telemetry") {
            const std::string value =
                require_value("--telemetry");
            if (value == "true") {
                options.telemetry = true;
            } else if (value == "false") {
                options.telemetry = false;
            } else {
                throw std::invalid_argument(
                    "--telemetry must be true or false"
                );
            }
        } else if (argument == "--list-profiles") {
            options.list_profiles = true;
        } else if (argument == "--help" || argument == "-h") {
            result.show_help = true;
        } else {
            throw std::invalid_argument(
                "Unknown option: " + argument
            );
        }
    }

    if (pi_host_set && options.pi_host.empty()) {
        throw std::invalid_argument("--pi-host is empty");
    }

    if (options.redis_host_set && options.redis_host.empty()) {
        throw std::invalid_argument("--redis-host is empty");
    }

    validate_common(options);

    if (result.show_help || options.list_profiles) {
        return result;
    }

    if (!pi_host_set) {
        throw std::invalid_argument("--pi-host is required");
    }

    if (!options.redis_host_set) {
        options.redis_host = options.pi_host;
    }

    return result;
}

bool face_detected(
    const std::vector<DetectionAnnotation>& annotations,
    int image_width,
    int image_height)
{
    for (const DetectionAnnotation& a : annotations) {
        if (a.class_id != 1) {
            continue;
        }
        if (!std::isfinite(a.score)
            || a.score < 0.0f || a.score > 1.0f) {
            continue;
        }
        if (a.x < 0 || a.y < 0
            || a.width <= 0 || a.height <= 0) {
            continue;
        }
        if (static_cast<int64_t>(a.x) + a.width > image_width
            || static_cast<int64_t>(a.y) + a.height
                > image_height) {
            continue;
        }
        return true;
    }
    return false;
}

std::string face_payload(bool detected)
{
    return detected
        ? "{\"face_detected\":true}"
        : "{\"face_detected\":false}";
}

bool state_fresh(
    Clock::time_point completed,
    Clock::time_point now,
    double stale_seconds)
{
    const double age =
        std::chrono::duration<double>(now - completed).count();
    return age >= 0.0 && age < stale_seconds;
}

Clock::time_point next_report_deadline(
    Clock::time_point origin,
    double interval_seconds,
    Clock::time_point now)
{
    const Clock::duration period =
        std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(interval_seconds)
        );

    if (period <= Clock::duration::zero()) {
        throw std::invalid_argument(
            "report interval below representable clock period"
        );
    }

    const Clock::duration elapsed =
        now > origin ? now - origin : Clock::duration::zero();
    const int64_t ticks = elapsed / period + 1;
    const Clock::time_point max_point =
        Clock::time_point::max();
    const Clock::duration headroom = max_point - origin;

    if (ticks <= 0
        || static_cast<__int128>(ticks) * period.count()
            > headroom.count()) {
        return max_point;
    }

    return origin + period * ticks;
}

double reconnect_backoff_seconds(int attempt_index)
{
    const int clamped = std::clamp(
        attempt_index,
        0,
        RECONNECT_BACKOFF_STEPS - 1
    );
    return RECONNECT_BACKOFF_SECONDS[clamped];
}

}
}
