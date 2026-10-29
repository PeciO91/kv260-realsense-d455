#include "yolov26_rpi_transport.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <thread>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <hiredis/hiredis.h>

extern char** environ;

namespace yolov26 {
namespace rpi {

namespace {

constexpr double PREFLIGHT_TIMEOUT_SECONDS = 5.0;
constexpr std::size_t REPLY_LIMIT = 4096;
constexpr std::size_t PROCESS_OUTPUT_TAIL = 2000;

Clock::time_point seconds_later(double seconds)
{
    return Clock::now()
        + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(seconds)
        );
}

bool wait_poll_slice(
    Clock::time_point deadline,
    const StopToken& stop)
{
    while (!stop.requested()) {
        const auto now = Clock::now();

        if (now >= deadline) {
            return true;
        }

        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now
            );
        std::this_thread::sleep_for(
            std::min(
                remaining,
                std::chrono::milliseconds(POLL_SLICE_MS)
            )
        );
    }

    return false;
}

void sanitize_error(std::string& error)
{
    for (char& c : error) {
        if (c == '\r' || c == '\n') {
            c = ' ';
        }
    }

    if (error.size() > 200) {
        error.resize(200);
    }
}

int run_bounded(
    const std::vector<std::string>& argv,
    double timeout_seconds,
    const StopToken& stop,
    std::string& tail)
{
    tail.clear();

    if (argv.empty()) {
        throw std::runtime_error("empty argv for managed process");
    }

    if (stop.requested()) {
        return -1;
    }

    std::vector<std::string> storage(argv.begin(), argv.end());
    std::vector<char*> args;
    args.reserve(storage.size() + 1);

    for (std::string& item : storage) {
        args.push_back(item.data());
    }
    args.push_back(nullptr);

    int output_pipe[2];

    if (pipe(output_pipe) < 0) {
        throw std::runtime_error(
            std::string("pipe() failed: ") + std::strerror(errno)
        );
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(
        &actions, output_pipe[1], STDOUT_FILENO
    );
    posix_spawn_file_actions_adddup2(
        &actions, output_pipe[1], STDERR_FILENO
    );
    posix_spawn_file_actions_addclose(&actions, output_pipe[0]);

    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    sigset_t empty_mask;
    sigemptyset(&empty_mask);
    posix_spawnattr_setsigmask(&attributes, &empty_mask);
    posix_spawnattr_setflags(
        &attributes, POSIX_SPAWN_SETSIGMASK
    );

    pid_t pid = -1;
    const int spawn_rc = posix_spawnp(
        &pid,
        args[0],
        &actions,
        &attributes,
        args.data(),
        environ
    );
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);

    if (spawn_rc != 0) {
        close(output_pipe[0]);
        close(output_pipe[1]);
        throw std::runtime_error(
            std::string("posix_spawnp(") + args[0]
            + ") failed: " + std::strerror(spawn_rc)
        );
    }

    close(output_pipe[1]);
    if (fcntl(output_pipe[0], F_SETFL, O_NONBLOCK) < 0) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        close(output_pipe[0]);
        throw std::runtime_error("fcntl(O_NONBLOCK) failed");
    }

    const auto deadline = seconds_later(timeout_seconds);
    int status = 0;
    bool exited = false;
    bool killed = false;

    while (true) {
        char buffer[4096];
        const ssize_t count =
            read(output_pipe[0], buffer, sizeof(buffer));

        if (count > 0) {
            tail.append(buffer, static_cast<std::size_t>(count));

            if (tail.size() > PROCESS_OUTPUT_TAIL) {
                tail.erase(0, tail.size() - PROCESS_OUTPUT_TAIL);
            }
        }

        pid_t result;
        do {
            result = waitpid(pid, &status, WNOHANG);
        } while (result < 0 && errno == EINTR);

        if (result == pid) {
            exited = true;
            break;
        }

        if (stop.requested() || Clock::now() >= deadline) {
            killed = true;
            kill(pid, SIGTERM);

            const auto kill_deadline = seconds_later(2.0);
            bool reaped = false;

            while (Clock::now() < kill_deadline) {
                do {
                    result = waitpid(pid, &status, WNOHANG);
                } while (result < 0 && errno == EINTR);

                if (result == pid) {
                    reaped = true;
                    break;
                }
                poll(nullptr, 0, POLL_SLICE_MS);
            }

            if (!reaped) {
                kill(pid, SIGKILL);
                do {
                    result = waitpid(pid, &status, 0);
                } while (result < 0 && errno == EINTR);
            }

            break;
        }

        poll(nullptr, 0, POLL_SLICE_MS);
    }

    const auto drain_deadline = seconds_later(0.5);
    char buffer[4096];

    while (Clock::now() < drain_deadline) {
        const ssize_t count =
            read(output_pipe[0], buffer, sizeof(buffer));
        if (count <= 0) {
            if (count < 0
                && (errno == EAGAIN || errno == EWOULDBLOCK
                    || errno == EINTR)) {
                poll(nullptr, 0, 5);
                continue;
            }
            break;
        }
        tail.append(buffer, static_cast<std::size_t>(count));

        if (tail.size() > PROCESS_OUTPUT_TAIL) {
            tail.erase(0, tail.size() - PROCESS_OUTPUT_TAIL);
        }
    }

    close(output_pipe[0]);

    if (!exited || killed) {
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    return -1;
}

GstElement* make_checked(
    const char* factory,
    const char* name)
{
    GstElement* element = gst_element_factory_make(factory, name);

    if (!element) {
        throw std::runtime_error(
            std::string("missing GStreamer factory: ") + factory
        );
    }

    return element;
}

GstPadProbeReturn count_access_units(
    GstPad*,
    GstPadProbeInfo* info,
    gpointer user_data)
{
    auto* counter = static_cast<std::atomic<uint64_t>*>(user_data);
    const GstPadProbeType type =
        static_cast<GstPadProbeType>(info->type);

    if (type & GST_PAD_PROBE_TYPE_BUFFER) {
        GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);

        if (buffer && gst_buffer_get_size(buffer) > 0) {
            counter->fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (type & GST_PAD_PROBE_TYPE_BUFFER_LIST) {
        GstBufferList* list =
            GST_PAD_PROBE_INFO_BUFFER_LIST(info);

        if (list) {
            const guint count = gst_buffer_list_length(list);

            for (guint i = 0; i < count; ++i) {
                GstBuffer* buffer = gst_buffer_list_get(list, i);

                if (buffer && gst_buffer_get_size(buffer) > 0) {
                    counter->fetch_add(
                        1,
                        std::memory_order_relaxed
                    );
                }
            }
        }
    }

    return GST_PAD_PROBE_OK;
}

std::string unbracket_ipv6(const std::string& host)
{
    if (host.size() > 2 && host.front() == '['
        && host.back() == ']') {
        return host.substr(1, host.size() - 2);
    }
    return host;
}

}

std::vector<ResolvedAddress> resolve_tcp_addresses(
    const std::string& host,
    int port,
    std::string& error)
{
    std::vector<ResolvedAddress> results;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    const std::string normalized = unbracket_ipv6(host);
    const std::string service = std::to_string(port);
    addrinfo* list = nullptr;
    const int rc = getaddrinfo(
        normalized.c_str(),
        service.c_str(),
        &hints,
        &list
    );

    if (rc != 0) {
        error = "cannot resolve '" + host
            + "': " + gai_strerror(rc);
        return results;
    }

    for (addrinfo* item = list; item; item = item->ai_next) {
        if (item->ai_family != AF_INET
            && item->ai_family != AF_INET6) {
            continue;
        }

        if (item->ai_addrlen > sizeof(sockaddr_storage)) {
            continue;
        }

        ResolvedAddress resolved;
        std::memcpy(&resolved.addr, item->ai_addr, item->ai_addrlen);
        resolved.length =
            static_cast<socklen_t>(item->ai_addrlen);
        results.push_back(resolved);
    }

    freeaddrinfo(list);

    if (results.empty()) {
        error = "no IPv4/IPv6 address for '" + host + "'";
    }

    return results;
}

bool resolve_tcp_address(
    const std::string& host,
    int port,
    ResolvedAddress& result,
    std::string& error)
{
    const std::vector<ResolvedAddress> all =
        resolve_tcp_addresses(host, port, error);

    if (all.empty()) {
        return false;
    }

    result = all.front();
    return true;
}

bool resolve_numeric_host(
    const std::string& host,
    std::string& numeric,
    std::string& error)
{
    const std::string normalized = unbracket_ipv6(host);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    addrinfo* list = nullptr;
    const int rc = getaddrinfo(
        normalized.c_str(),
        nullptr,
        &hints,
        &list
    );

    if (rc != 0) {
        error = "cannot resolve video host '" + host
            + "': " + gai_strerror(rc);
        return false;
    }

    char buffer[NI_MAXHOST];
    bool ok = false;

    for (addrinfo* item = list; item; item = item->ai_next) {
        if (item->ai_family != AF_INET
            && item->ai_family != AF_INET6) {
            continue;
        }

        if (getnameinfo(
                item->ai_addr,
                static_cast<socklen_t>(item->ai_addrlen),
                buffer,
                sizeof(buffer),
                nullptr,
                0,
                NI_NUMERICHOST) == 0) {
            numeric = buffer;
            ok = true;
            break;
        }
    }

    freeaddrinfo(list);

    if (!ok) {
        error = "no IPv4/IPv6 address for video host '"
            + host + "'";
    }

    return ok;
}

RedisPublisher::RedisPublisher() = default;

RedisPublisher::~RedisPublisher()
{
    close();
}

bool RedisPublisher::connected() const
{
    return fd_ >= 0;
}

void RedisPublisher::close()
{
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }

    if (reader_) {
        redisReaderFree(
            static_cast<redisReader*>(reader_)
        );
        reader_ = nullptr;
    }

    pending_.clear();
}

bool RedisPublisher::connect_deadline(
    const std::vector<ResolvedAddress>& candidates,
    double timeout_seconds,
    const StopToken& stop,
    std::string& error)
{
    close();

    if (stop.requested()) {
        error = "cancelled";
        return false;
    }

    const auto deadline = seconds_later(timeout_seconds);

    for (const ResolvedAddress& address : candidates) {
        if (stop.requested()) {
            error = "cancelled";
            return false;
        }

        const auto now = Clock::now();

        if (now >= deadline) {
            error = "connect timed out";
            return false;
        }

        const int fd =
            socket(address.addr.ss_family, SOCK_STREAM, 0);

        if (fd < 0) {
            error = std::string("socket() failed: ")
                + std::strerror(errno);
            continue;
        }

        const int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0
            || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
            error = "fcntl(O_NONBLOCK) failed";
            ::close(fd);
            continue;
        }

        const int fd_flags = fcntl(fd, F_GETFD, 0);
        if (fd_flags < 0
            || fcntl(fd, F_SETFD, fd_flags | FD_CLOEXEC) < 0) {
            error = "fcntl(FD_CLOEXEC) failed";
            ::close(fd);
            continue;
        }

        const int rc = connect(
            fd,
            reinterpret_cast<const sockaddr*>(&address.addr),
            address.length
        );

        if (rc < 0 && errno != EINPROGRESS) {
            error = std::string("connect() failed: ")
                + std::strerror(errno);
            ::close(fd);
            continue;
        }

        bool established = (rc == 0);

        while (!established) {
            if (stop.requested()) {
                ::close(fd);
                error = "cancelled";
                return false;
            }

            const auto now2 = Clock::now();

            if (now2 >= deadline) {
                ::close(fd);
                error = "connect timed out";
                return false;
            }

            const int remaining_ms = static_cast<int>(
                std::min<int64_t>(
                    POLL_SLICE_MS,
                    std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        deadline - now2
                    ).count()
                )
            );
            pollfd descriptor{};
            descriptor.fd = fd;
            descriptor.events = POLLOUT;
            const int ready =
                poll(&descriptor, 1, std::max(1, remaining_ms));

            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                error = std::string("poll() failed: ")
                    + std::strerror(errno);
                break;
            }

            if (ready == 0) {
                continue;
            }

            int socket_error = 0;
            socklen_t size = sizeof(socket_error);

            if (getsockopt(fd, SOL_SOCKET, SO_ERROR,
                           &socket_error, &size) < 0) {
                error = std::string("getsockopt() failed: ")
                    + std::strerror(errno);
                break;
            }

            if (socket_error != 0) {
                error = std::string("connect() failed: ")
                    + std::strerror(socket_error);
                break;
            }

            established = true;
        }

        if (established) {
            fd_ = fd;
            reader_ = redisReaderCreate();

            if (!reader_) {
                error = "redisReaderCreate failed";
                close();
                return false;
            }

            return true;
        }

        ::close(fd);
    }

    if (error.empty()) {
        error = "no usable address";
    }

    return false;
}

int64_t RedisPublisher::publish(
    const std::string& channel,
    const std::string& payload,
    double timeout_seconds,
    const StopToken& stop,
    std::string& error)
{
    if (fd_ < 0 || !reader_) {
        error = "not connected";
        return -1;
    }

    if (stop.requested()) {
        error = "cancelled";
        close();
        return -1;
    }

    const char* args[3] = {
        "PUBLISH",
        channel.data(),
        payload.data()
    };
    const std::size_t lengths[3] = {
        std::strlen(args[0]),
        channel.size(),
        payload.size()
    };
    char* command = nullptr;
    const int command_length = redisFormatCommandArgv(
        &command,
        3,
        args,
        lengths
    );

    if (command_length < 0 || !command) {
        error = "failed to format PUBLISH command";
        return -1;
    }

    struct CommandGuard {
        char* ptr;
        ~CommandGuard() { if (ptr) hi_free(ptr); }
    } guard{command};

    const auto deadline = seconds_later(timeout_seconds);
    std::size_t sent = 0;
    const std::size_t total =
        static_cast<std::size_t>(command_length);

    while (sent < total) {
        if (stop.requested()) {
            error = "cancelled";
            close();
            return -1;
        }

        const auto now = Clock::now();

        if (now >= deadline) {
            error = "write timed out";
            close();
            return -1;
        }

        const int remaining_ms = static_cast<int>(
            std::min<int64_t>(
                POLL_SLICE_MS,
                std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    deadline - now
                ).count()
            )
        );
        pollfd descriptor{};
        descriptor.fd = fd_;
        descriptor.events = POLLOUT;
        const int ready =
            poll(&descriptor, 1, std::max(1, remaining_ms));

        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = std::string("poll() failed: ")
                + std::strerror(errno);
            close();
            return -1;
        }

        if (ready == 0) {
            continue;
        }

        const ssize_t written = send(
            fd_,
            command + sent,
            total - sent,
            MSG_NOSIGNAL
        );

        if (written <= 0) {
            if (written < 0
                && (errno == EAGAIN || errno == EWOULDBLOCK
                    || errno == EINTR)) {
                continue;
            }
            error = std::string("send() failed: ")
                + (written < 0
                    ? std::strerror(errno)
                    : "connection closed");
            close();
            return -1;
        }

        sent += static_cast<std::size_t>(written);
    }

    while (true) {
        if (stop.requested()) {
            error = "cancelled";
            close();
            return -1;
        }

        const auto now = Clock::now();

        if (now >= deadline) {
            error = "read timed out";
            close();
            return -1;
        }

        if (pending_.size() > REPLY_LIMIT) {
            error = "oversized Redis reply";
            close();
            return -1;
        }

        const std::size_t newline = pending_.find('\n');

        if (newline != std::string::npos) {
            std::string line = pending_.substr(0, newline);
            const std::string rest =
                pending_.substr(newline + 1);

            if (line.empty() || line.back() != '\r'
                || !rest.empty() || line.size() > REPLY_LIMIT) {
                error = "malformed Redis reply";
                close();
                return -1;
            }

            const char kind = line[0];
            const std::string body = line.substr(
                1,
                line.size() - 2
            );

            if (kind == ':') {
                if (body.empty()
                    || !std::all_of(
                        body.begin(), body.end(),
                        [](char c) {
                            return std::isdigit(
                                static_cast<unsigned char>(c)
                            );
                        })) {
                    error = "malformed Redis integer reply";
                    close();
                    return -1;
                }

                errno = 0;
                char* end = nullptr;
                std::strtoll(body.c_str(), &end, 10);

                if (end != body.c_str() + body.size()
                    || errno == ERANGE) {
                    error = "Redis integer overflow";
                    close();
                    return -1;
                }

                const std::string wire =
                    pending_.substr(0, newline + 1);
                redisReader* reader =
                    static_cast<redisReader*>(reader_);
                void* reply = nullptr;

                if (redisReaderFeed(
                        reader, wire.data(), wire.size()) != REDIS_OK
                    || redisReaderGetReply(reader, &reply)
                        != REDIS_OK
                    || !reply
                    || static_cast<redisReply*>(reply)->type
                        != REDIS_REPLY_INTEGER) {
                    if (reply) {
                        freeReplyObject(reply);
                    }
                    error = "hiredis reader rejected reply";
                    close();
                    return -1;
                }

                const int64_t subscribers =
                    static_cast<redisReply*>(reply)->integer;
                freeReplyObject(reply);

                if (subscribers < 0) {
                    error = "unexpected negative subscriber count";
                    close();
                    return -1;
                }

                pending_.clear();
                return subscribers;
            }

            if (kind == '-') {
                if (body.find('\r') != std::string::npos) {
                    error = "malformed Redis error reply";
                    close();
                    return -1;
                }
                const std::string wire =
                    pending_.substr(0, newline + 1);
                redisReader* reader =
                    static_cast<redisReader*>(reader_);
                void* reply = nullptr;

                if (redisReaderFeed(
                        reader, wire.data(), wire.size())
                        != REDIS_OK
                    || redisReaderGetReply(reader, &reply)
                        != REDIS_OK
                    || !reply
                    || static_cast<redisReply*>(reply)->type
                        != REDIS_REPLY_ERROR) {
                    if (reply) {
                        freeReplyObject(reply);
                    }
                    error = "hiredis reader rejected reply";
                    close();
                    return -1;
                }

                error = "Redis error: " + body;
                freeReplyObject(reply);
                sanitize_error(error);
                pending_.clear();
                close();
                return -1;
            }

            error = "unexpected Redis reply type";
            close();
            return -1;
        }

        const int remaining_ms = static_cast<int>(
            std::min<int64_t>(
                POLL_SLICE_MS,
                std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    deadline - now
                ).count()
            )
        );
        pollfd descriptor{};
        descriptor.fd = fd_;
        descriptor.events = POLLIN;
        const int ready =
            poll(&descriptor, 1, std::max(1, remaining_ms));

        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            error = std::string("poll() failed: ")
                + std::strerror(errno);
            close();
            return -1;
        }

        if (ready == 0) {
            continue;
        }

        char buffer[4096];
        const ssize_t count =
            recv(fd_, buffer, sizeof(buffer), 0);

        if (count <= 0) {
            if (count < 0
                && (errno == EAGAIN || errno == EWOULDBLOCK
                    || errno == EINTR)) {
                continue;
            }
            error = count == 0
                ? "connection closed by server"
                : std::string("recv() failed: ")
                    + std::strerror(errno);
            close();
            return -1;
        }

        pending_.append(buffer, static_cast<std::size_t>(count));
    }
}

void run_face_publisher(
    const PublisherParams& params,
    const LatestFaceState& state,
    const StopToken& stop,
    PublisherHooks hooks)
{
    const Clock::time_point schedule_origin = Clock::now();
    Clock::time_point deadline = next_report_deadline(
        schedule_origin,
        params.interval_seconds,
        Clock::now()
    );
    int failures = 0;
    Clock::time_point next_attempt = Clock::now();
    Clock::time_point last_error_log = Clock::time_point::min();

    const auto log_error = [&](const std::string& message) {
        if (!hooks.log_error) {
            return;
        }
        const auto now = Clock::now();

        if (last_error_log == Clock::time_point::min()
            || elapsed_ms(last_error_log, now) >= 5000.0) {
            hooks.log_error(message);
            last_error_log = now;
        }
    };

    while (!stop.requested()) {
        if (!wait_poll_slice(deadline, stop)) {
            break;
        }

        const auto current = state.snapshot();

        if (!current
            || !state_fresh(
                current->second,
                Clock::now(),
                params.stale_seconds
            )) {
            deadline = next_report_deadline(
                schedule_origin,
                params.interval_seconds,
                Clock::now()
            );
            continue;
        }

        if (!hooks.connected()) {
            if (!wait_poll_slice(next_attempt, stop)) {
                break;
            }

            const auto eligible = state.snapshot();

            if (!eligible
                || !state_fresh(
                    eligible->second,
                    Clock::now(),
                    params.stale_seconds
                )) {
                deadline = next_report_deadline(
                    schedule_origin,
                    params.interval_seconds,
                    Clock::now()
                );
                continue;
            }

            std::string error;

            if (hooks.connect(
                    params.candidates,
                    params.timeout_seconds,
                    stop,
                    error)) {
                deadline = next_report_deadline(
                    schedule_origin,
                    params.interval_seconds,
                    Clock::now()
                );
                if (hooks.log_info) {
                    hooks.log_info("Redis connected");
                }
            } else {
                if (stop.requested()) {
                    break;
                }
                log_error("connect failed: " + error);
                const double backoff =
                    reconnect_backoff_seconds(failures);
                if (failures < RECONNECT_BACKOFF_STEPS - 1) {
                    ++failures;
                }
                next_attempt = seconds_later(backoff);
                deadline = next_report_deadline(
                    schedule_origin,
                    params.interval_seconds,
                    Clock::now()
                );
            }
            continue;
        }

        const auto latest = state.snapshot();

        if (latest
            && state_fresh(
                latest->second,
                Clock::now(),
                params.stale_seconds
            )) {
            std::string error;
            const int64_t subscribers = hooks.publish(
                params.channel,
                face_payload(latest->first),
                params.timeout_seconds,
                stop,
                error
            );

            if (subscribers < 0) {
                if (stop.requested()) {
                    break;
                }
                log_error("publish failed: " + error);
                hooks.close();
                const double backoff =
                    reconnect_backoff_seconds(failures);
                if (failures < RECONNECT_BACKOFF_STEPS - 1) {
                    ++failures;
                }
                next_attempt = seconds_later(backoff);
            } else {
                failures = 0;
                if (hooks.log_info) {
                    hooks.log_info(
                        "Redis PUBLISH " + params.channel
                        + " " + face_payload(latest->first)
                        + " (subscribers="
                        + std::to_string(subscribers) + ")"
                    );
                }
            }
        }

        deadline = next_report_deadline(
            schedule_origin,
            params.interval_seconds,
            Clock::now()
        );
    }

    hooks.close();
}

bool gstreamer_element_available(const std::string& element)
{
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    GstRegistry* registry = gst_registry_get();
    GstPluginFeature* feature =
        gst_registry_find_feature(
            registry,
            element.c_str(),
            GST_TYPE_ELEMENT_FACTORY
        );

    if (!feature) {
        return false;
    }

    gst_object_unref(feature);
    return true;
}

EncoderPreflightResult smoke_encoder(
    const std::string& encoder,
    int width,
    int height,
    int fps,
    int bitrate_kbps,
    const StopToken& stop)
{
    if (stop.requested()) {
        return {false, "cancelled"};
    }

    EncoderPreflightResult result;

    for (const char* factory : {
             "videotestsrc", "queue", "videoconvert",
             "capsfilter", "h264parse", "rtph264pay",
             "appsrc", "udpsink", "fakesink"}) {
        if (!gstreamer_element_available(factory)) {
            result.detail =
                std::string("missing GStreamer factory: ")
                + factory;
            return result;
        }
    }

    const std::string element =
        encoder == "x264" ? "x264enc" : "omxh264enc";

    if (!gstreamer_element_available(element)) {
        result.detail = element + " not present";
        return result;
    }

    std::vector<std::string> argv = {
        "gst-launch-1.0", "-q", "-e",
        "videotestsrc", "num-buffers=1", "pattern=snow",
        "!", "video/x-raw,format=BGR,width="
            + std::to_string(width)
            + ",height=" + std::to_string(height)
            + ",framerate=" + std::to_string(fps) + "/1",
        "!", "queue",
        "max-size-buffers=1",
        "max-size-bytes=0",
        "max-size-time=0",
        "leaky=downstream",
        "!", "videoconvert",
        "!", "video/x-raw,format=NV12",
        "!"
    };

    if (encoder == "x264") {
        argv.insert(argv.end(), {
            "x264enc",
            "tune=zerolatency",
            "speed-preset=ultrafast",
            "bitrate=" + std::to_string(bitrate_kbps),
            "key-int-max=30",
            "bframes=0",
            "rc-lookahead=0"
        });
    } else {
        argv.insert(argv.end(), {
            "omxh264enc",
            "target-bitrate=" + std::to_string(bitrate_kbps),
            "control-rate=low-latency",
            "b-frames=0",
            "look-ahead=0",
            "gop-mode=basic",
            "gop-length=30",
            "periodicity-idr=30"
        });
    }

    argv.insert(argv.end(), {
        "!", "video/x-h264,profile=baseline",
        "!", "h264parse",
        "!", "video/x-h264,alignment=au,stream-format=byte-stream",
        "!", "rtph264pay",
        "pt=96",
        "config-interval=1",
        "mtu=1200",
        "!", "fakesink", "sync=false"
    });

    std::string tail;
    const int code = run_bounded(
        argv,
        PREFLIGHT_TIMEOUT_SECONDS,
        stop,
        tail
    );

    if (code != 0) {
        if (stop.requested()) {
            result.detail = "cancelled";
            return result;
        }
        result.detail = element + " smoke encode failed (rc="
            + std::to_string(code) + "): " + tail;
        return result;
    }

    result.ok = true;
    return result;
}

struct GstVideoSender::Impl {
    explicit Impl(const VideoSenderConfig& c) : config(c) {}

    VideoSenderConfig config;
    std::mutex mutex;
    GstElement* pipeline = nullptr;
    GstAppSrc* appsrc = nullptr;
    GstElement* parser = nullptr;
    gulong probe_id = 0;
    GstVideoInfo video_info{};
    GstClockTime base_running_time = 0;
    Clock::time_point host_anchor{};
    uint64_t last_pts = 0;
    bool have_last_pts = false;
    std::atomic<uint64_t> access_units{0};
    bool running = false;
};

GstVideoSender::GstVideoSender(const VideoSenderConfig& config)
    : impl_(std::make_unique<Impl>(config))
{
    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
    }

    if (config.width <= 0 || config.width > 8192
        || config.height <= 0 || config.height > 8192
        || config.width % 2 != 0 || config.height % 2 != 0) {
        throw std::invalid_argument(
            "invalid frame dimensions "
            + std::to_string(config.width) + "x"
            + std::to_string(config.height)
        );
    }

    if (config.fps < 1 || config.fps > 240) {
        throw std::invalid_argument("invalid frame rate");
    }

    if (config.bitrate_kbps < 1) {
        throw std::invalid_argument("invalid bitrate");
    }

    if (config.encoder != "x264" && config.encoder != "omx") {
        throw std::invalid_argument(
            "invalid encoder '" + config.encoder + "'"
        );
    }

    if (config.port < 1 || config.port > 65535) {
        throw std::invalid_argument("invalid video port");
    }

    if (config.host.empty()) {
        throw std::invalid_argument("empty video host");
    }

    if (!gst_video_info_set_format(
            &impl_->video_info,
            GST_VIDEO_FORMAT_BGR,
            config.width,
            config.height)) {
        throw std::invalid_argument(
            "cannot describe BGR frame layout"
        );
    }
}

GstVideoSender::~GstVideoSender()
{
    stop();
}

void GstVideoSender::start()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);

    if (impl_->running) {
        return;
    }

    const VideoSenderConfig& c = impl_->config;
    GstElement* pipeline = gst_pipeline_new("yolov26-sender");

    if (!pipeline) {
        throw std::runtime_error("cannot create GStreamer pipeline");
    }

    std::vector<GstElement*> pending;
    GstElement* parser = nullptr;
    gulong probe = 0;

    try {
        const auto take =
            [&pending](GstElement* element) -> GstElement* {
                pending.push_back(element);
                return element;
            };

        GstElement* appsrc = take(make_checked("appsrc", "source"));
        GstElement* queue = take(make_checked("queue", "raw-queue"));
        GstElement* convert =
            take(make_checked("videoconvert", "convert"));
        GstElement* nv12 =
            take(make_checked("capsfilter", "nv12-caps"));
        GstElement* encoder = nullptr;

        if (c.encoder == "x264") {
            encoder = take(make_checked("x264enc", "encoder"));
            gst_util_set_object_arg(
                G_OBJECT(encoder),
                "tune",
                "zerolatency"
            );
            gst_util_set_object_arg(
                G_OBJECT(encoder),
                "speed-preset",
                "ultrafast"
            );
            g_object_set(
                encoder,
                "bitrate", c.bitrate_kbps,
                "key-int-max", 30,
                "bframes", 0,
                "rc-lookahead", 0,
                nullptr
            );
        } else {
            encoder = take(make_checked("omxh264enc", "encoder"));
            gst_util_set_object_arg(
                G_OBJECT(encoder),
                "control-rate",
                "low-latency"
            );
            gst_util_set_object_arg(
                G_OBJECT(encoder),
                "gop-mode",
                "basic"
            );
            g_object_set(
                encoder,
                "target-bitrate", c.bitrate_kbps,
                "b-frames", 0u,
                "look-ahead", 0u,
                "gop-length", 30u,
                "periodicity-idr", 30u,
                nullptr
            );
        }

        GstElement* baseline =
            take(make_checked("capsfilter", "baseline-caps"));
        parser = take(make_checked("h264parse", "parse"));
        GstElement* au_caps =
            take(make_checked("capsfilter", "au-caps"));
        GstElement* pay = take(make_checked("rtph264pay", "pay"));
        GstElement* sink = take(make_checked("udpsink", "sink"));

        GstCaps* input_caps = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, c.width,
            "height", G_TYPE_INT, c.height,
            "framerate", GST_TYPE_FRACTION, c.fps, 1,
            nullptr
        );

        g_object_set(
            appsrc,
            "caps", input_caps,
            "format", GST_FORMAT_TIME,
            "is-live", TRUE,
            "block", TRUE,
            "max-bytes",
            static_cast<guint64>(impl_->video_info.size),
            nullptr
        );
        gst_caps_unref(input_caps);

        gst_util_set_object_arg(
            G_OBJECT(queue),
            "leaky",
            "downstream"
        );
        g_object_set(
            queue,
            "max-size-buffers", 1u,
            "max-size-bytes", 0u,
            "max-size-time", G_GUINT64_CONSTANT(0),
            nullptr
        );

        GstCaps* nv12_caps = gst_caps_new_simple(
            "video/x-raw",
            "format", G_TYPE_STRING, "NV12",
            nullptr
        );
        g_object_set(nv12, "caps", nv12_caps, nullptr);
        gst_caps_unref(nv12_caps);

        GstCaps* baseline_caps = gst_caps_new_simple(
            "video/x-h264",
            "profile", G_TYPE_STRING, "baseline",
            nullptr
        );
        g_object_set(baseline, "caps", baseline_caps, nullptr);
        gst_caps_unref(baseline_caps);

        GstCaps* byte_stream_caps = gst_caps_new_simple(
            "video/x-h264",
            "alignment", G_TYPE_STRING, "au",
            "stream-format", G_TYPE_STRING, "byte-stream",
            nullptr
        );
        g_object_set(au_caps, "caps", byte_stream_caps, nullptr);
        gst_caps_unref(byte_stream_caps);

        g_object_set(
            pay,
            "pt", 96u,
            "config-interval", 1,
            "mtu", RTP_MTU,
            "perfect-rtptime", FALSE,
            nullptr
        );

        g_object_set(
            sink,
            "host", c.host.c_str(),
            "port", c.port,
            "sync", FALSE,
            "async", FALSE,
            nullptr
        );

        gst_bin_add_many(
            GST_BIN(pipeline),
            appsrc,
            queue,
            convert,
            nv12,
            encoder,
            baseline,
            parser,
            au_caps,
            pay,
            sink,
            nullptr
        );
        pending.clear();

        if (!gst_element_link_many(
                appsrc,
                queue,
                convert,
                nv12,
                encoder,
                baseline,
                parser,
                au_caps,
                pay,
                sink,
                nullptr)) {
            throw std::runtime_error(
                "failed to link GStreamer pipeline"
            );
        }

        if (c.count_access_units) {
            GstPad* pad =
                gst_element_get_static_pad(parser, "src");

            if (!pad) {
                throw std::runtime_error(
                    "missing h264parse src pad"
                );
            }

            probe = gst_pad_add_probe(
                pad,
                static_cast<GstPadProbeType>(
                    GST_PAD_PROBE_TYPE_BUFFER
                    | GST_PAD_PROBE_TYPE_BUFFER_LIST
                ),
                count_access_units,
                &impl_->access_units,
                nullptr
            );
            gst_object_unref(pad);
        }

        const GstStateChangeReturn change =
            gst_element_set_state(pipeline, GST_STATE_PLAYING);

        if (change == GST_STATE_CHANGE_FAILURE) {
            throw std::runtime_error(
                "GStreamer pipeline failed to start"
            );
        }

        GstClock* clock =
            gst_element_get_clock(pipeline);
        GstClockTime base_time =
            gst_element_get_base_time(pipeline);
        GstClockTime running = GST_CLOCK_TIME_NONE;

        if (clock) {
            running = gst_clock_get_time(clock) - base_time;
            gst_object_unref(clock);
        }

        impl_->base_running_time =
            running == GST_CLOCK_TIME_NONE ? 0 : running;
        impl_->host_anchor = Clock::now();
        impl_->last_pts = 0;
        impl_->have_last_pts = false;
        impl_->probe_id = probe;
        probe = 0;
        impl_->parser = parser;
        impl_->pipeline = pipeline;
        impl_->appsrc = GST_APP_SRC(appsrc);
        impl_->running = true;
    } catch (...) {
        if (probe && parser) {
            GstPad* pad =
                gst_element_get_static_pad(parser, "src");
            if (pad) {
                gst_pad_remove_probe(pad, probe);
                gst_object_unref(pad);
            }
        }
        gst_element_set_state(pipeline, GST_STATE_NULL);
        for (GstElement* element : pending) {
            gst_object_unref(element);
        }
        gst_object_unref(pipeline);
        throw;
    }
}

bool GstVideoSender::push(
    const void* bgr_data,
    std::size_t row_stride_bytes,
    Clock::time_point acquired_at,
    uint64_t duration_ns)
{
    GstAppSrc* appsrc = nullptr;

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        if (!impl_->running || !impl_->appsrc) {
            return false;
        }

        appsrc = GST_APP_SRC(
            gst_object_ref(impl_->appsrc)
        );
    }

    const GstVideoInfo& info = impl_->video_info;
    const std::size_t row_bytes =
        static_cast<std::size_t>(info.width) * 3;

    if (!bgr_data
        || row_stride_bytes < row_bytes) {
        gst_object_unref(appsrc);
        return false;
    }

    GstBuffer* buffer = gst_buffer_new_allocate(
        nullptr,
        info.size,
        nullptr
    );

    if (!buffer) {
        gst_object_unref(appsrc);
        return false;
    }

    GstMapInfo map;

    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        gst_object_unref(appsrc);
        return false;
    }

    const auto* source =
        static_cast<const uint8_t*>(bgr_data);
    uint8_t* destination =
        map.data + info.offset[0];
    const std::size_t gst_stride =
        static_cast<std::size_t>(info.stride[0]);

    for (int row = 0; row < info.height; ++row) {
        std::memcpy(
            destination + row * gst_stride,
            source + row * row_stride_bytes,
            row_bytes
        );

        if (gst_stride > row_bytes) {
            std::memset(
                destination + row * gst_stride + row_bytes,
                0,
                gst_stride - row_bytes
            );
        }
    }

    gst_buffer_unmap(buffer, &map);

    uint64_t pts;

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        const int64_t host_delta =
            std::chrono::duration_cast<
                std::chrono::nanoseconds>(
                acquired_at - impl_->host_anchor
            ).count();
        const int64_t mapped =
            static_cast<int64_t>(impl_->base_running_time)
            + host_delta;
        pts = static_cast<uint64_t>(
            std::max<int64_t>(0, mapped)
        );

        if (impl_->have_last_pts && pts <= impl_->last_pts) {
            pts = impl_->last_pts + 1;
        }

        impl_->last_pts = pts;
        impl_->have_last_pts = true;
    }

    GST_BUFFER_PTS(buffer) = pts;
    GST_BUFFER_DTS(buffer) = pts;
    GST_BUFFER_DURATION(buffer) = duration_ns;

    const GstFlowReturn flow =
        gst_app_src_push_buffer(appsrc, buffer);
    gst_object_unref(appsrc);

    return flow == GST_FLOW_OK;
}

void GstVideoSender::stop()
{
    GstElement* pipeline;
    GstElement* parser;
    gulong probe_id;

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);

        impl_->running = false;
        impl_->appsrc = nullptr;
        pipeline = impl_->pipeline;
        parser = impl_->parser;
        probe_id = impl_->probe_id;
        impl_->pipeline = nullptr;
        impl_->parser = nullptr;
        impl_->probe_id = 0;
    }

    if (!pipeline && !parser) {
        return;
    }

    if (pipeline) {
        gst_element_set_state(pipeline, GST_STATE_NULL);
    }

    if (probe_id && parser) {
        GstPad* pad =
            gst_element_get_static_pad(parser, "src");

        if (pad) {
            gst_pad_remove_probe(pad, probe_id);
            gst_object_unref(pad);
        }
    }

    if (pipeline) {
        gst_object_unref(pipeline);
    }
}

bool GstVideoSender::running() const
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->running;
}

bool GstVideoSender::poll_bus_error(std::string& error)
{
    GstElement* pipeline;

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        pipeline = impl_->pipeline
            ? GST_ELEMENT(gst_object_ref(impl_->pipeline))
            : nullptr;
    }

    if (!pipeline) {
        return false;
    }

    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* message = gst_bus_timed_pop_filtered(
        bus,
        0,
        static_cast<GstMessageType>(
            GST_MESSAGE_ERROR | GST_MESSAGE_EOS
        )
    );
    gst_object_unref(bus);
    gst_object_unref(pipeline);

    if (!message) {
        return false;
    }

    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
        GError* detail = nullptr;
        gchar* debug = nullptr;
        gst_message_parse_error(message, &detail, &debug);
        error = detail && detail->message
            ? detail->message
            : "unknown GStreamer bus error";

        if (detail) {
            g_error_free(detail);
        }

        g_free(debug);
    } else {
        error = "GStreamer pipeline reached EOS";
    }

    gst_message_unref(message);
    return true;
}

uint64_t GstVideoSender::access_unit_count() const
{
    return impl_->access_units.load(std::memory_order_relaxed);
}

}
}
