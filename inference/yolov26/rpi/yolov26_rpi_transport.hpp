#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <sys/socket.h>

#include "yolov26_common.hpp"
#include "yolov26_rpi_control.hpp"

namespace yolov26 {
namespace rpi {

struct ResolvedAddress {
    sockaddr_storage addr{};
    socklen_t length = 0;
};

std::vector<ResolvedAddress> resolve_tcp_addresses(
    const std::string& host,
    int port,
    std::string& error);

bool resolve_tcp_address(
    const std::string& host,
    int port,
    ResolvedAddress& result,
    std::string& error);

bool resolve_numeric_host(
    const std::string& host,
    std::string& numeric,
    std::string& error);

class StopToken {
public:
    virtual ~StopToken() = default;
    virtual bool requested() const = 0;
};

class RedisPublisher {
public:
    RedisPublisher();
    ~RedisPublisher();

    RedisPublisher(const RedisPublisher&) = delete;
    RedisPublisher& operator=(const RedisPublisher&) = delete;

    bool connect_deadline(
        const std::vector<ResolvedAddress>& candidates,
        double timeout_seconds,
        const StopToken& stop,
        std::string& error);

    int64_t publish(
        const std::string& channel,
        const std::string& payload,
        double timeout_seconds,
        const StopToken& stop,
        std::string& error);

    void close();
    bool connected() const;

private:
    int fd_ = -1;
    void* reader_ = nullptr;
    std::string pending_;
};

struct PublisherParams {
    std::vector<ResolvedAddress> candidates;
    std::string channel;
    double interval_seconds = DEFAULT_REDIS_INTERVAL_SECONDS;
    double stale_seconds = DEFAULT_FACE_STALE_SECONDS;
    double timeout_seconds =
        DEFAULT_REDIS_TIMEOUT_MS / 1000.0;
};

struct PublisherHooks {
    std::function<bool(
        const std::vector<ResolvedAddress>&,
        double,
        const StopToken&,
        std::string&)> connect;
    std::function<int64_t(
        const std::string&,
        const std::string&,
        double,
        const StopToken&,
        std::string&)> publish;
    std::function<void()> close;
    std::function<bool()> connected;
    std::function<void(const std::string&)> log_error;
    std::function<void(const std::string&)> log_info;
};

void run_face_publisher(
    const PublisherParams& params,
    const LatestFaceState& state,
    const StopToken& stop,
    PublisherHooks hooks);

struct EncoderPreflightResult {
    bool ok = false;
    std::string detail;
};

bool gstreamer_element_available(const std::string& element);

EncoderPreflightResult smoke_encoder(
    const std::string& encoder,
    int width,
    int height,
    int fps,
    int bitrate_kbps,
    const StopToken& stop);

struct VideoSenderConfig {
    std::string host;
    int port = 5000;
    int width = 848;
    int height = 480;
    int fps = 30;
    int bitrate_kbps = 4000;
    std::string encoder = "x264";
    bool count_access_units = false;
};

class GstVideoSender {
public:
    explicit GstVideoSender(const VideoSenderConfig& config);
    ~GstVideoSender();

    GstVideoSender(const GstVideoSender&) = delete;
    GstVideoSender& operator=(const GstVideoSender&) = delete;

    void start();
    bool push(
        const void* bgr_data,
        std::size_t row_stride_bytes,
        Clock::time_point acquired_at,
        uint64_t duration_ns);
    void stop();
    bool running() const;
    bool poll_bus_error(std::string& error);
    uint64_t access_unit_count() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
}
