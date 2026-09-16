import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
CAMERA_FPS = 30


class FrameStore:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.sequence = 0

    def update(self, frame):
        with self.condition:
            self.frame = frame
            self.sequence += 1
            self.condition.notify_all()

    def wait_for_frame(self, last_sequence, timeout=2.0):
        with self.condition:
            if self.sequence <= last_sequence:
                self.condition.wait(timeout)

            return self.frame, self.sequence


class StreamHandler(BaseHTTPRequestHandler):
    frame_store = None

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()

            html = """
<!DOCTYPE html>
<html>
<head>
    <title>KV260 RealSense D455</title>
    <style>
        body {
            font-family: sans-serif;
            background: #111;
            color: #eee;
            text-align: center;
        }

        img {
            max-width: 95vw;
            max-height: 85vh;
        }
    </style>
</head>
<body>
    <h1>KV260 RealSense D455</h1>
    <img src="/stream.mjpg">
</body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            last_sequence = -1

            try:
                while True:
                    frame, sequence = self.frame_store.wait_for_frame(
                        last_sequence
                    )

                    if frame is None or sequence == last_sequence:
                        continue

                    last_sequence = sequence

                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(
                        f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    )
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            except (
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ):
                pass

            return

        self.send_error(404)

    def log_message(self, format, *args):
        return


def capture_frames(frame_store, stop_event, stream_fps, jpeg_quality):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        CAMERA_FPS,
    )

    print("Starting RealSense D455...")

    profile = pipeline.start(config)
    device = profile.get_device()

    print("Camera:", device.get_info(rs.camera_info.name))
    print("Serial:", device.get_info(rs.camera_info.serial_number))

    frame_interval = 1.0 / stream_fps
    next_frame_time = 0.0

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            now = time.monotonic()

            # Camera can run at 30 FPS while network preview is slower.
            if now < next_frame_time:
                continue

            next_frame_time = now + frame_interval

            image = np.asanyarray(color_frame.get_data())

            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )

            if not ok:
                continue

            frame_store.update(encoded.tobytes())

    finally:
        pipeline.stop()
        print("Camera stopped")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
    )

    parser.add_argument(
        "--stream-fps",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
    )

    args = parser.parse_args()

    frame_store = FrameStore()
    stop_event = threading.Event()

    StreamHandler.frame_store = frame_store

    capture_thread = threading.Thread(
        target=capture_frames,
        args=(
            frame_store,
            stop_event,
            args.stream_fps,
            args.jpeg_quality,
        ),
        daemon=True,
    )

    capture_thread.start()

    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port),
        StreamHandler,
    )

    print()
    print(f"HTTP server listening on port {args.port}")
    print("Open http://<KV260-IP>:%d/ on your PC" % args.port)
    print("Press Ctrl+C to stop.")
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        server.shutdown()
        server.server_close()

        stop_event.set()
        capture_thread.join(timeout=3)

        print("Server stopped")


if __name__ == "__main__":
    main()
