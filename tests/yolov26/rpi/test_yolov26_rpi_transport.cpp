#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <netinet/in.h>
#include <optional>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <sys/socket.h>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include "yolov26_rpi_transport.hpp"

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

class TestStop : public StopToken {
public:
    bool requested() const override
    {
        return flag.load(std::memory_order_relaxed);
    }
    void request() { flag.store(true); }
    std::atomic<bool> flag{false};
};

TestStop never_stop;

class UdpReceiver {
public:
    bool open()
    {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) return false;
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(0);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) < 0) {
            close_fd();
            return false;
        }
        sockaddr_in bound{};
        socklen_t len = sizeof(bound);
        if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&bound),
                          &len) < 0) {
            close_fd();
            return false;
        }
        port_ = ntohs(bound.sin_port);
        return true;
    }
    ~UdpReceiver() { close_fd(); }
    int port() const { return port_; }

    void release()
    {
        close_fd();
    }

    bool recv(std::vector<uint8_t>& out, int timeout_ms)
    {
        pollfd pfd{};
        pfd.fd = fd_;
        pfd.events = POLLIN;
        if (::poll(&pfd, 1, timeout_ms) <= 0) return false;
        out.resize(2048);
        const ssize_t n = ::recv(fd_, out.data(), out.size(), 0);
        if (n < 0) return false;
        out.resize(static_cast<size_t>(n));
        return true;
    }

private:
    void close_fd()
    {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }
    int fd_ = -1;
    int port_ = 0;
};

class FakeRedis {
public:
    bool open()
    {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) return false;
        int one = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR,
                     &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(0);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(fd_, reinterpret_cast<sockaddr*>(&addr),
                   sizeof(addr)) < 0
            || ::listen(fd_, 4) < 0) {
            close_fd();
            return false;
        }
        sockaddr_in bound{};
        socklen_t len = sizeof(bound);
        if (::getsockname(fd_, reinterpret_cast<sockaddr*>(&bound),
                          &len) < 0) {
            close_fd();
            return false;
        }
        port_ = ntohs(bound.sin_port);
        return true;
    }
    ~FakeRedis() { close_fd(); }
    int port() const { return port_; }

    void set_reply(const std::string& reply)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reply_ = reply;
        silent_ = false;
        fragment_ = false;
    }
    void set_fragmented(const std::string& reply)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reply_ = reply;
        silent_ = false;
        fragment_ = true;
    }
    void set_silent()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        silent_ = true;
    }
    void set_close_on_read()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        close_on_read_ = true;
    }
    int accept_count() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return accepts_;
    }
    std::vector<std::vector<std::string>> publishes() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return publishes_;
    }
    void stop()
    {
        done_.store(true);
    }

    bool serve_one(int timeout_ms)
    {
        pollfd pfd{};
        pfd.fd = fd_;
        pfd.events = POLLIN;
        if (::poll(&pfd, 1, timeout_ms) <= 0) return false;
        const int client = ::accept(fd_, nullptr, nullptr);
        if (client < 0) return false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++accepts_;
        }
        std::string buffer;
        bool got_command = false;
        while (!done_.load()) {
            pollfd cfd{};
            cfd.fd = client;
            cfd.events = POLLIN;
            if (::poll(&cfd, 1, 100) <= 0) {
                if (!got_command) continue;
                continue;
            }
            char chunk[512];
            const ssize_t n = ::recv(client, chunk, sizeof(chunk), 0);
            if (n <= 0) break;
            buffer.append(chunk, static_cast<size_t>(n));
            std::vector<std::string> args;
            while (parse_command(buffer, args)) {
                got_command = true;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    publishes_.push_back(args);
                }
                std::string reply;
                bool silent = false;
                bool fragment = false;
                bool close_read = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    reply = reply_;
                    silent = silent_;
                    fragment = fragment_;
                    close_read = close_on_read_;
                }
                if (close_read || silent || reply.empty()) {
                    continue;
                }
                if (fragment) {
                    for (char c : reply) {
                        if (::send(client, &c, 1,
                                   MSG_NOSIGNAL) < 0) break;
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(2)
                        );
                    }
                } else if (::send(
                               client, reply.data(),
                               reply.size(), MSG_NOSIGNAL) < 0) {
                    break;
                }
            }
        }
        ::close(client);
        return got_command;
    }

    void serve(int connections)
    {
        for (int i = 0; i < connections && !done_.load(); ++i) {
            serve_one(300);
        }
    }

private:
    static bool parse_command(
        std::string& buffer,
        std::vector<std::string>& args)
    {
        if (buffer.empty() || buffer[0] != '*') return false;
        const auto eol = buffer.find("\r\n");
        if (eol == std::string::npos) return false;
        int count = 0;
        try {
            count = std::stoi(buffer.substr(1, eol - 1));
        } catch (...) {
            return false;
        }
        if (count < 1 || count > 16) return false;
        size_t pos = eol + 2;
        args.clear();
        for (int i = 0; i < count; ++i) {
            if (pos >= buffer.size()) return false;
            if (buffer[pos] != '$') return false;
            const auto next = buffer.find("\r\n", pos);
            if (next == std::string::npos) return false;
            std::size_t len = 0;
            try {
                len = std::stoul(
                    buffer.substr(pos + 1, next - pos - 1)
                );
            } catch (...) {
                return false;
            }
            if (len > 1u << 20) return false;
            pos = next + 2;
            if (buffer.size() < pos + len + 2) return false;
            if (buffer.compare(pos + len, 2, "\r\n") != 0) {
                return false;
            }
            args.push_back(buffer.substr(pos, len));
            pos += len + 2;
        }
        buffer.erase(0, pos);
        return true;
    }
    void close_fd()
    {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }
    int fd_ = -1;
    int port_ = 0;
    int accepts_ = 0;
    bool silent_ = false;
    bool fragment_ = false;
    bool close_on_read_ = false;
    std::atomic<bool> done_{false};
    std::string reply_ = ":0\r\n";
    std::vector<std::vector<std::string>> publishes_;
    mutable std::mutex mutex_;
};

bool gst_has(const std::string& element)
{
    return gstreamer_element_available(element);
}

std::vector<uint8_t> make_bgr(int width, int height)
{
    std::vector<uint8_t> frame(
        static_cast<size_t>(width)
            * static_cast<size_t>(height) * 3
    );
    for (size_t i = 0; i < frame.size(); ++i) {
        frame[i] = static_cast<uint8_t>(i % 251);
    }
    return frame;
}

void test_resolution()
{
    std::string error;
    std::string numeric;

    check(resolve_numeric_host("127.0.0.1", numeric, error)
              && numeric == "127.0.0.1",
          "ipv4 numeric resolves");

    check(resolve_numeric_host("::1", numeric, error)
              && numeric == "::1",
          "ipv6 numeric resolves");

    check(resolve_numeric_host("[::1]", numeric, error)
              && numeric == "::1",
          "bracketed ipv6 resolves");

    check(resolve_numeric_host("localhost", numeric, error)
              && !numeric.empty(),
          "localhost resolves numeric");

    check(resolve_numeric_host("localhost.", numeric, error)
              && !numeric.empty(),
          "trailing-dot hostname resolves");

    check(!resolve_numeric_host(
              "no.such.host.invalid.", numeric, error)
          && !error.empty(),
          "unresolvable host errors");

    const auto candidates = resolve_tcp_addresses(
        "localhost", 6379, error
    );
    check(!candidates.empty(), "localhost tcp candidates");
    bool saw_v4 = false;
    for (const auto& c : candidates) {
        if (c.addr.ss_family == AF_INET) saw_v4 = true;
    }
    check(saw_v4, "localhost candidates include ipv4");

    ResolvedAddress first;
    check(resolve_tcp_address("localhost", 6379, first, error),
          "single resolve helper");

    check(!resolve_tcp_addresses(
              "no.such.host.invalid.", 6379, error).size()
          && !error.empty(),
          "tcp resolve failure reports error");
}

void test_sender_config()
{
    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = 5000;
    config.width = 848;
    config.height = 480;

    {
        VideoSenderConfig bad = config;
        bad.width = 0;
        bool threw = false;
        try { GstVideoSender s(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "zero width rejected");
    }
    {
        VideoSenderConfig bad = config;
        bad.height = 0;
        bool threw = false;
        try { GstVideoSender s(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "zero height rejected");
    }
    {
        VideoSenderConfig bad = config;
        bad.fps = 0;
        bool threw = false;
        try { GstVideoSender s(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "zero fps rejected");
    }
    {
        VideoSenderConfig bad = config;
        bad.bitrate_kbps = 0;
        bool threw = false;
        try { GstVideoSender s(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "zero bitrate rejected");
    }
    {
        VideoSenderConfig bad = config;
        bad.port = 70000;
        bool threw = false;
        try { GstVideoSender s(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "port out of range rejected");
    }
    {
        VideoSenderConfig bad = config;
        bad.host.clear();
        bool threw = false;
        try { GstVideoSender s(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "empty host rejected");
    }
    {
        VideoSenderConfig bad = config;
        bad.encoder = "bogus";
        bool threw = false;
        try { GstVideoSender s(bad); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "unknown encoder rejected");
    }
}

void test_encoded_stream()
{
    if (!gst_has("x264enc") || !gst_has("udpsink")
        || !gst_has("rtph264pay")) {
        std::fprintf(stderr, "SKIP encoded: missing elements\n");
        return;
    }

    UdpReceiver receiver;
    check(receiver.open(), "udp receiver binds");

    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = receiver.port();
    config.width = 640;
    config.height = 360;
    config.fps = 30;
    config.bitrate_kbps = 800;
    config.count_access_units = true;

    GstVideoSender sender(config);
    bool started = false;
    try {
        sender.start();
        started = true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "start failed: %s\n", e.what());
        check(false, "sender start with available factories");
    }
    if (!started) return;

    const auto frame = make_bgr(config.width, config.height);
    const auto t0 = Clock::now();
    int pushed = 0;
    const int frames_sent = 30;
    for (int i = 0; i < frames_sent; ++i) {
        const int offset_ms =
            i < 10 ? i * 33 : 33 * (i + 2);
        if (sender.push(
                frame.data(),
                static_cast<size_t>(config.width) * 3,
                t0 + std::chrono::milliseconds(offset_ms),
                33000000)) {
            ++pushed;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(33)
        );
    }
    sender.push(
        frame.data(),
        static_cast<size_t>(config.width) * 3,
        t0 + std::chrono::milliseconds(33 * (frames_sent + 2)),
        33000000
    );
    sender.push(
        frame.data(),
        static_cast<size_t>(config.width) * 3,
        t0 + std::chrono::milliseconds(33 * (frames_sent + 2)),
        33000000
    );
    check(pushed == frames_sent, "all pushes accepted");

    std::vector<uint8_t> packet;
    int rtp_packets = 0;
    bool saw_pt96 = false;
    bool saw_v2 = false;
    bool mtu_ok = true;
    bool saw_ts = false;
    bool ts_increasing = true;
    bool saw_gap = false;
    uint32_t prev_ts = 0;
    bool have_prev = false;
    const auto deadline =
        std::chrono::steady_clock::now()
        + std::chrono::seconds(8);
    while (std::chrono::steady_clock::now() < deadline
           && rtp_packets < 400) {
        if (!receiver.recv(packet, 200)) continue;
        ++rtp_packets;
        if (packet.size() > 1200) mtu_ok = false;
        if (packet.size() < 12) continue;
        if ((packet[0] >> 6) == 2) saw_v2 = true;
        if ((packet[1] & 0x7f) == 96) saw_pt96 = true;
        const uint32_t ts =
            (static_cast<uint32_t>(packet[4]) << 24)
            | (static_cast<uint32_t>(packet[5]) << 16)
            | (static_cast<uint32_t>(packet[6]) << 8)
            | packet[7];
        saw_ts = true;
        if (have_prev) {
            const uint32_t diff = ts - prev_ts;
            if (diff == 0) continue;
            if (diff > 0x80000000u) ts_increasing = false;
            if (diff >= 8500 && diff <= 9500) saw_gap = true;
            if (diff > 2500) {
                std::fprintf(stderr, "ts diff %u\n", diff);
            }
        }
        prev_ts = ts;
        have_prev = true;
    }
    sender.stop();

    check(rtp_packets > 0, "rtp packets received");
    check(saw_v2, "rtp version 2 on wire");
    check(saw_pt96, "payload type 96 on wire");
    check(mtu_ok, "datagrams within mtu");
    check(saw_ts, "rtp timestamps parsed");
    check(ts_increasing, "rtp timestamps monotonic");
    check(saw_gap, "dropped-frame timestamp gap present");
    check(sender.access_unit_count() > 0,
          "au probe counted encoded units");
}

void test_push_input_validation()
{
    if (!gst_has("x264enc") || !gst_has("udpsink")) {
        std::fprintf(stderr, "SKIP push validation\n");
        return;
    }
    UdpReceiver receiver;
    check(receiver.open(), "validation receiver binds");

    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = receiver.port();
    config.width = 64;
    config.height = 64;
    config.bitrate_kbps = 300;

    GstVideoSender sender(config);
    bool started = false;
    try { sender.start(); started = true; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "start failed: %s\n", e.what());
        check(false, "sender start with available factories");
    }
    if (!started) return;

    const auto frame = make_bgr(64, 64);
    const auto now = Clock::now();

    check(!sender.push(nullptr, 192, now, 33000000),
          "null buffer rejected");
    check(!sender.push(frame.data(), 10, now, 33000000),
          "short stride rejected");
    check(sender.push(frame.data(), 192, now, 33000000),
          "valid push accepted");

    sender.stop();
    check(!sender.push(frame.data(), 192, now, 33000000),
          "push after stop returns false");
}

void test_decoded_output()
{
    const char* decoder = nullptr;
    for (const char* candidate :
         {"avdec_h264", "openh264dec", "omxh264dec"}) {
        if (gst_has(candidate)) {
            decoder = candidate;
            break;
        }
    }
    if (!decoder || !gst_has("x264enc")) {
        std::fprintf(stderr, "SKIP decode: no h264 decoder\n");
        return;
    }

    UdpReceiver receiver;
    check(receiver.open(), "decode receiver binds");
    const int port = receiver.port();
    receiver.release();

    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = port;
    config.width = 64;
    config.height = 64;
    config.bitrate_kbps = 400;

    GstVideoSender sender(config);
    bool started = false;
    try { sender.start(); started = true; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "start failed: %s\n", e.what());
        check(false, "sender start with available factories");
    }
    if (!started) return;

    const std::string launch =
        "udpsrc address=127.0.0.1 port="
        + std::to_string(port)
        + " caps=\"application/x-rtp,media=video,"
          "encoding-name=H264,clock-rate=90000,payload=96\" ! "
          "rtph264depay ! h264parse ! " + decoder
        + " ! videoconvert ! video/x-raw,format=BGR ! "
          "appsink name=sink sync=false";

    GError* error = nullptr;
    GstElement* pipeline = gst_parse_launch(
        launch.c_str(), &error
    );
    check(pipeline != nullptr, "decode pipeline parses");
    if (error) g_error_free(error);
    if (!pipeline) {
        sender.stop();
        return;
    }
    GstElement* sink = gst_bin_get_by_name(
        GST_BIN(pipeline), "sink"
    );
    if (gst_element_set_state(pipeline, GST_STATE_PLAYING)
            == GST_STATE_CHANGE_FAILURE) {
        std::fprintf(
            stderr, "SKIP decode: %s failed to play\n", decoder
        );
        gst_object_unref(sink);
        gst_object_unref(pipeline);
        sender.stop();
        return;
    }

    const auto frame = make_bgr(64, 64);
    const auto t0 = Clock::now();
    int decoded = 0;
    const auto deadline =
        std::chrono::steady_clock::now()
        + std::chrono::seconds(12);
    int pushes = 0;
    while (std::chrono::steady_clock::now() < deadline
           && decoded < 2) {
        if (pushes < 40) {
            sender.push(
                frame.data(), 192,
                t0 + std::chrono::milliseconds(pushes * 33),
                33000000
            );
            ++pushes;
        }
        GstSample* sample = gst_app_sink_try_pull_sample(
            GST_APP_SINK(sink), 400 * GST_MSECOND
        );
        if (!sample) continue;
        GstCaps* caps = gst_sample_get_caps(sample);
        GstStructure* s = caps
            ? gst_caps_get_structure(caps, 0)
            : nullptr;
        const gchar* fmt = nullptr;
        int w = 0, h = 0;
        if (s) {
            fmt = gst_structure_get_string(s, "format");
            gst_structure_get_int(s, "width", &w);
            gst_structure_get_int(s, "height", &h);
        }
        check(fmt && std::strcmp(fmt, "BGR") == 0,
              "decoded format BGR");
        check(w == 64 && h == 64, "decoded dimensions");
        GstBuffer* buf = gst_sample_get_buffer(sample);
        check(buf && gst_buffer_get_size(buf) == 64u * 64u * 3u,
              "decoded buffer byte size");
        gst_sample_unref(sample);
        ++decoded;
    }
    if (decoded == 0) {
        std::fprintf(
            stderr,
            "NOTE: %s produced no decoded buffers\n",
            decoder
        );
    }
    check(decoded > 0, "raw decoded buffers received");

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    sender.stop();
}

void test_padded_width_and_ownership()
{
    if (!gst_has("x264enc") || !gst_has("udpsink")) {
        std::fprintf(stderr, "SKIP padded width\n");
        return;
    }
    UdpReceiver receiver;
    check(receiver.open(), "padded receiver binds");

    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = receiver.port();
    config.width = 318;
    config.height = 64;
    config.bitrate_kbps = 300;

    GstVideoSender sender(config);
    bool started = false;
    try { sender.start(); started = true; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "start failed: %s\n", e.what());
        check(false, "sender start with available factories");
    }
    if (!started) return;

    const std::size_t stride = 318u * 3u;
    std::vector<uint8_t> frame(stride * 64u, 0xAB);
    check(sender.push(frame.data(), stride,
                      Clock::now(), 33000000),
          "non-aligned-width push accepted");

    frame.assign(frame.size(), 0xCD);
    check(sender.push(frame.data(), stride,
                      Clock::now(), 33000000),
          "mutated source push accepted");

    sender.stop();
}

void test_backpressure_and_stop()
{
    if (!gst_has("x264enc") || !gst_has("udpsink")) {
        std::fprintf(stderr, "SKIP backpressure\n");
        return;
    }
    UdpReceiver receiver;
    check(receiver.open(), "backpressure receiver binds");

    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = receiver.port();
    config.width = 320;
    config.height = 240;
    config.bitrate_kbps = 200;

    GstVideoSender sender(config);
    bool started = false;
    try { sender.start(); started = true; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "start failed: %s\n", e.what());
        check(false, "sender start with available factories");
    }
    if (!started) return;

    const auto frame = make_bgr(320, 240);
    std::atomic<int> failed{0};
    std::atomic<bool> done{false};

    std::thread pusher([&] {
        for (int i = 0; i < 500 && !done.load(); ++i) {
            if (!sender.push(
                    frame.data(), 960,
                    Clock::now(), 33000000)) {
                ++failed;
            }
        }
    });
    std::this_thread::sleep_for(
        std::chrono::milliseconds(50)
    );
    sender.stop();
    done.store(true);
    pusher.join();
    check(true, "concurrent stop/push completed");
    check(!sender.push(frame.data(), 960,
                       Clock::now(), 33000000),
          "push after stop rejected");
}

void test_redis_publish()
{
    FakeRedis server;
    check(server.open(), "redis fixture binds");

    std::string error;
    const auto candidates = resolve_tcp_addresses(
        "127.0.0.1", server.port(), error
    );
    check(!candidates.empty(), "redis candidates resolved");

    std::thread server_thread([&] {
        server.serve(40);
    });

    RedisPublisher publisher;
    never_stop.flag.store(false);

    const auto try_publish = [&](const std::string& payload,
            const std::string& channel = "kv260_test")
            -> int64_t {
        if (!publisher.connected()) {
            if (!publisher.connect_deadline(
                    candidates, 2.0, never_stop, error)) {
                std::fprintf(
                    stderr, "connect: %s\n", error.c_str()
                );
                return -1;
            }
        }
        const int64_t rc = publisher.publish(
            channel, payload, 2.0, never_stop, error
        );
        if (rc < 0) {
            std::fprintf(
                stderr, "publish: %s\n", error.c_str()
            );
        }
        return rc;
    };

    server.set_reply(":3\r\n");
    const int64_t r1 = try_publish("{\"face_detected\":true}");
    check(r1 == 3, "integer reply value returned");

    server.set_fragmented(":5\r\n");
    const int64_t r2 = try_publish("{\"face_detected\":false}");
    check(r2 == 5, "fragmented reply assembled");

    server.set_reply("-ERR nope\r\n");
    const int64_t r3 = try_publish("x");
    check(r3 < 0, "error reply fails");
    check(!publisher.connected(),
          "error reply closes connection");

    server.set_reply("+OK\r\n");
    const int64_t r4 = try_publish("x");
    check(r4 < 0, "unexpected type fails");

    server.set_reply(":99999999999999999999999\r\n");
    const int64_t r5 = try_publish("x");
    check(r5 < 0, "overflow integer fails");

    server.set_reply(":+5\r\n");
    const int64_t r6 = try_publish("x");
    check(r6 < 0, "signed reply rejected");

    server.set_reply(":1\n");
    const int64_t r7 = try_publish("x");
    check(r7 < 0, "bare LF rejected");

    server.set_reply(
        ":9223372036854775807\r\n"
    );
    const int64_t r_max = try_publish("x");
    check(r_max == INT64_MAX, "INT64_MAX reply ok");

    server.set_reply(":0\r\n");
    const int64_t r_zero = try_publish("x");
    check(r_zero == 0, "zero reply ok");

    {
        std::string big = ":" + std::string(4095, '7') + "\r\n";
        server.set_reply(big);
        const int64_t r_big = try_publish("x");
        check(r_big < 0, "over-4096 reply rejected");
    }

    server.set_reply(":1\r\n:2\r\n");
    const int64_t r_extra = try_publish("x");
    check(r_extra < 0, "extra reply line rejected");

    server.set_reply("-ERR bad\rtext\r\n");
    const int64_t r_cr = try_publish("x");
    check(r_cr < 0, "embedded CR in error rejected");

    const char channel_bytes[] = "kv260_\xC3\xA9\r\n\0channel";
    const char payload_bytes[] = "\xC3\xA9\0\xFF\xD7\x80";
    const std::string binary_channel(
        channel_bytes, sizeof(channel_bytes) - 1
    );
    const std::string binary_payload(
        payload_bytes, sizeof(payload_bytes) - 1
    );
    server.set_reply(":2\r\n");
    const int64_t r_uni = try_publish(
        binary_payload, binary_channel
    );
    check(r_uni == 2, "binary payload argv safe");

    server.set_silent();
    RedisPublisher publisher2;
    check(publisher2.connect_deadline(
              candidates, 2.0, never_stop, error),
          "connect to silent server ok");
    const int64_t r8 = publisher2.publish(
        "kv260_test", "x", 0.8, never_stop, error
    );
    check(r8 < 0, "read timeout fails");
    check(!publisher2.connected(),
          "timeout closes connection");
    publisher2.close();

    RedisPublisher publisher3;
    TestStop stopped;
    stopped.request();
    check(!publisher3.connect_deadline(
              candidates, 5.0, stopped, error),
          "pre-stopped connect refused");

    publisher.close();
    server.stop();
    server_thread.join();

    check(server.publishes().size() >= 10,
          "server received commands");
    if (!server.publishes().empty()) {
        check(server.publishes()[0].size() == 3
                  && server.publishes()[0][0] == "PUBLISH"
                  && server.publishes()[0][1] == "kv260_test",
              "framed PUBLISH argv");
        check(server.publishes()[0][2]
                  == "{\"face_detected\":true}",
              "payload bytes intact");
        bool saw_binary = false;
        for (const auto& args : server.publishes()) {
            if (args.size() == 3
                && args[1] == binary_channel
                && args[2] == binary_payload) {
                saw_binary = true;
            }
        }
        check(saw_binary,
              "binary argv reached server intact");
    }
}

void test_closed_port()
{
    std::string error;
    const auto candidates = resolve_tcp_addresses(
        "127.0.0.1", 1, error
    );
    check(!candidates.empty(), "closed-port candidates");

    RedisPublisher publisher;
    check(!publisher.connect_deadline(
              candidates, 1.0, never_stop, error),
          "closed port fails connect");
    check(!error.empty(), "connect error reported");
}

void test_publisher_loop()
{
    FakeRedis server;
    check(server.open(), "loop server binds");

    std::string error;
    PublisherParams params;
    params.candidates = resolve_tcp_addresses(
        "127.0.0.1", server.port(), error
    );
    params.channel = "kv260_test";
    params.interval_seconds = 0.15;
    params.stale_seconds = 1.0;
    params.timeout_seconds = 2.0;
    check(!params.candidates.empty(), "loop candidates");

    TestStop stop;
    LatestFaceState state;

    RedisPublisher real_pub;
    PublisherHooks hooks;
    hooks.connect =
        [&](const std::vector<ResolvedAddress>& c,
            double t, const StopToken& s, std::string& e) {
            return real_pub.connect_deadline(c, t, s, e);
        };
    hooks.publish =
        [&](const std::string& ch, const std::string& p,
            double t, const StopToken& s, std::string& e) {
            return real_pub.publish(ch, p, t, s, e);
        };
    hooks.close = [&] { real_pub.close(); };
    hooks.connected = [&] { return real_pub.connected(); };

    server.set_reply(":1\r\n");
    std::thread server_thread([&] {
        server.serve(30);
    });

    std::thread worker([&] {
        run_face_publisher(params, state, stop, hooks);
    });

    std::this_thread::sleep_for(
        std::chrono::milliseconds(120)
    );
    check(server.accept_count() == 0,
          "no connect before state exists");

    state.update(false, Clock::now());
    bool first_seen = false;
    for (int i = 0; i < 30; ++i) {
        if (!server.publishes().empty()) {
            first_seen = true;
            break;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100)
        );
    }
    check(first_seen, "publish happened after state");
    if (!server.publishes().empty()) {
        check(server.publishes()[0].size() == 3
                  && server.publishes()[0][2]
                      == "{\"face_detected\":false}",
              "first publish false");
    }

    state.update(true, Clock::now());
    const size_t before = server.publishes().size();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(800)
    );
    const auto all = server.publishes();
    check(all.size() > before, "further publishes");
    bool saw_true = false;
    for (const auto& p : all) {
        if (p.size() == 3 && p[2] == "{\"face_detected\":true}") {
            saw_true = true;
        }
    }
    check(saw_true, "latest true state published");

    state.update(false, Clock::now());
    std::this_thread::sleep_for(
        std::chrono::milliseconds(500)
    );
    const auto later = server.publishes();
    bool last_is_false = false;
    if (!later.empty()) {
        last_is_false =
            later.back()[2] == "{\"face_detected\":false}";
    }
    check(last_is_false,
          "latest false not OR-combined with true");

    stop.request();
    worker.join();
    server.stop();
    server_thread.join();
    check(true, "publisher loop exits on stop");
}

void test_publisher_hooks_backoff()
{
    TestStop stop;
    LatestFaceState state;
    state.update(true, Clock::now());

    PublisherParams params;
    params.channel = "ch";
    params.interval_seconds = 0.15;
    params.stale_seconds = 60.0;
    params.timeout_seconds = 0.3;

    std::vector<double> attempt_times;
    std::mutex attempt_mutex;
    std::atomic<int> connect_calls{0};
    std::atomic<bool> allow_connect{false};
    std::atomic<int> publish_calls{0};
    std::atomic<bool> allow_publish{false};
    std::atomic<bool> conn_state{false};
    const auto t0 = Clock::now();

    PublisherHooks hooks;
    hooks.connect =
        [&](const std::vector<ResolvedAddress>&,
            double, const StopToken&, std::string& e) {
            ++connect_calls;
            {
                std::lock_guard<std::mutex> g(attempt_mutex);
                attempt_times.push_back(
                    std::chrono::duration<double>(
                        Clock::now() - t0
                    ).count()
                );
            }
            conn_state = allow_connect.load();
            if (!conn_state) e = "refused";
            return conn_state.load();
        };
    hooks.publish =
        [&](const std::string&, const std::string&,
            double, const StopToken&, std::string& e) {
            ++publish_calls;
            if (!allow_publish.load()) {
                conn_state = false;
                e = "write fail";
                return static_cast<int64_t>(-1);
            }
            return static_cast<int64_t>(1);
        };
    hooks.close = [&] { conn_state = false; };
    hooks.connected = [&] { return conn_state.load(); };
    std::mutex log_mutex;
    std::vector<std::string> log_lines;
    hooks.log_error = [&](const std::string& m) {
        std::lock_guard<std::mutex> g(log_mutex);
        log_lines.push_back("E " + m);
    };
    hooks.log_info = [&](const std::string& m) {
        std::lock_guard<std::mutex> g(log_mutex);
        log_lines.push_back("I " + m);
    };

    std::thread worker([&] {
        run_face_publisher(params, state, stop, hooks);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    check(connect_calls == 1,
          "first backoff suppresses retry for ~1s");
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    check(connect_calls >= 2,
          "connect retried on failure");
    {
        std::lock_guard<std::mutex> g(attempt_mutex);
        if (attempt_times.size() >= 2) {
            check(attempt_times[1] - attempt_times[0] >= 0.9,
                  "first backoff ~1s");
        }
    }

    allow_connect = true;
    for (int i = 0; i < 50 && connect_calls < 3; ++i) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100)
        );
    }
    check(connect_calls >= 3, "reconnect after backoff");

    for (int i = 0; i < 30 && publish_calls < 1; ++i) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100)
        );
    }
    check(publish_calls >= 1, "publish attempted after connect");

    allow_publish = true;
    const int pubs_before = publish_calls;
    for (int i = 0; i < 120 && publish_calls <= pubs_before; ++i) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(100)
        );
    }
    check(publish_calls > pubs_before,
          "successful publish after failures");
    if (publish_calls <= pubs_before) {
        std::lock_guard<std::mutex> g(log_mutex);
        for (const auto& l : log_lines) {
            std::fprintf(stderr, "LOG %s\n", l.c_str());
        }
        std::fprintf(stderr, "connect_calls=%d publish_calls=%d\n",
                     connect_calls.load(), publish_calls.load());
    }
    stop.request();
    worker.join();

    check(true, "hooks loop exited");
}

void test_concurrent_stop_poll_push()
{
    if (!gst_has("x264enc") || !gst_has("udpsink")) {
        std::fprintf(stderr, "SKIP concurrent stop\n");
        return;
    }
    UdpReceiver receiver;
    check(receiver.open(), "concurrent receiver binds");

    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = receiver.port();
    config.width = 64;
    config.height = 64;
    config.bitrate_kbps = 200;
    config.count_access_units = true;

    GstVideoSender sender(config);
    bool started = false;
    try { sender.start(); started = true; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "start failed: %s\n", e.what());
        check(false, "sender start with available factories");
    }
    if (!started) return;

    const auto frame = make_bgr(64, 64);
    std::atomic<bool> go{false};

    std::thread pusher([&] {
        go.store(true);
        for (int i = 0; i < 300; ++i) {
            sender.push(frame.data(), 192,
                        Clock::now(), 33000000);
        }
    });
    std::thread poller([&] {
        std::string err;
        for (int i = 0; i < 300; ++i) {
            sender.poll_bus_error(err);
        }
    });
    std::thread stopper([&] {
        while (!go.load()) {}
        sender.stop();
    });

    pusher.join();
    poller.join();
    stopper.join();
    sender.stop();
    check(!sender.running(), "sender stopped");
    check(true, "concurrent stop/poll/push survived");
}

void test_preflight_fake_gstlaunch()
{
    const char* tmpdir = std::getenv("FAKEBIN");
    if (!tmpdir) {
        std::fprintf(
            stderr,
            "SKIP preflight: FAKEBIN fixture dir unset\n"
        );
        return;
    }

    const std::string bindir = tmpdir;
    const std::string log = bindir + "/exec.log";

    const auto write_script = [&](const char* body) {
        const std::string path = bindir + "/gst-launch-1.0";
        FILE* f = std::fopen(path.c_str(), "w");
        if (f) {
            std::fputs("#!/bin/sh\n", f);
            std::fprintf(
                f, "echo EXEC >> %s\n", log.c_str()
            );
            std::fputs(body, f);
            std::fclose(f);
        }
        ::chmod(path.c_str(), 0755);
    };

    const std::string old_path = std::getenv("PATH")
        ? std::getenv("PATH") : "";
    const std::string scoped =
        bindir + ":" + old_path;
    setenv("PATH", scoped.c_str(), 1);

    {
        std::remove(log.c_str());
        write_script(
            "echo verbose-preflight-output\nexit 0\n"
        );
        const auto result = smoke_encoder(
            "x264", 64, 64, 30, 300, never_stop
        );
        check(result.ok, "preflight success passes");
        check(result.detail.empty()
                  || result.detail.find("verbose")
                      == std::string::npos,
              "success has no error detail");
    }

    {
        std::remove(log.c_str());
        write_script("exit 3\n");
        const auto result = smoke_encoder(
            "x264", 64, 64, 30, 300, never_stop
        );
        check(!result.ok, "nonzero exit fails");
        check(!result.detail.empty(),
              "failure has detail");
    }

    {
        std::remove(log.c_str());
        const std::string pidfile = bindir + "/child.pid";
        std::remove(pidfile.c_str());
        const std::string body =
            "echo $$ > " + pidfile + "\n"
            "trap '' TERM\n"
            "while :; do :; done\n";
        write_script(body.c_str());
        const auto begin = Clock::now();
        const auto result = smoke_encoder(
            "x264", 64, 64, 30, 300, never_stop
        );
        const double elapsed =
            std::chrono::duration<double>(
                Clock::now() - begin
            ).count();
        check(!result.ok, "term-ignorer killed");
        check(elapsed < 20.0, "bounded kill wait");
        FILE* f = std::fopen(pidfile.c_str(), "r");
        check(f != nullptr, "child pidfile written");
        if (f) {
            int child = -1;
            if (std::fscanf(f, "%d", &child) == 1 && child > 0) {
                check(::kill(child, 0) < 0 && errno == ESRCH,
                      "owned child reaped");
            }
            std::fclose(f);
        }
    }

    {
        std::remove(log.c_str());
        write_script("exit 0\n");
        TestStop stopped;
        stopped.request();
        const auto result = smoke_encoder(
            "x264", 64, 64, 30, 300, stopped
        );
        check(!result.ok, "pre-stopped returns fail");
        FILE* f = std::fopen(log.c_str(), "r");
        check(!f, "no exec when pre-stopped");
        if (f) std::fclose(f);
    }

    setenv("PATH", old_path.c_str(), 1);
}

void test_telemetry_disabled_no_probe()
{
    if (!gst_has("x264enc") || !gst_has("udpsink")) {
        std::fprintf(stderr, "SKIP telemetry-off\n");
        return;
    }
    UdpReceiver receiver;
    check(receiver.open(), "telemetry-off binds");

    VideoSenderConfig config;
    config.host = "127.0.0.1";
    config.port = receiver.port();
    config.width = 64;
    config.height = 64;
    config.bitrate_kbps = 300;
    config.count_access_units = false;

    GstVideoSender sender(config);
    bool started = false;
    try { sender.start(); started = true; }
    catch (const std::exception& e) {
        std::fprintf(stderr, "start failed: %s\n", e.what());
        check(false, "sender start with available factories");
    }
    if (!started) return;

    const auto frame = make_bgr(64, 64);
    sender.push(frame.data(), 192, Clock::now(), 33000000);
    std::this_thread::sleep_for(
        std::chrono::milliseconds(300)
    );
    sender.stop();
    check(sender.access_unit_count() == 0,
          "no au counting when disabled");
}

}

int main(int argc, char** argv)
{
    gst_init(&argc, &argv);
    test_resolution();
    test_sender_config();
    test_encoded_stream();
    test_push_input_validation();
    test_decoded_output();
    test_backpressure_and_stop();
    test_padded_width_and_ownership();
    test_redis_publish();
    test_closed_port();
    test_publisher_loop();
    test_publisher_hooks_backoff();
    test_concurrent_stop_poll_push();
    test_preflight_fake_gstlaunch();
    test_telemetry_disabled_no_probe();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
