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

# Half-size of the square used for distance measurement.
# 5 -> 11x11 pixel region.
DEPTH_RADIUS = 5


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
    <p>RGB + aligned depth</p>
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


def get_region_distance(depth_frame, depth_scale, x, y, radius):
    """
    Return median distance in metres from a square region
    around (x, y). Zero depth values are ignored.
    """
    depth_image = np.asanyarray(depth_frame.get_data())

    x1 = max(0, x - radius)
    x2 = min(depth_image.shape[1], x + radius + 1)

    y1 = max(0, y - radius)
    y2 = min(depth_image.shape[0], y + radius + 1)

    region = depth_image[y1:y2, x1:x2]

    valid = region[region > 0]

    if valid.size == 0:
        return None

    return float(np.median(valid)) * depth_scale


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

    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        CAMERA_FPS,
    )

    align = rs.align(rs.stream.color)

    print("Starting RealSense D455 RGB + depth...")

    profile = pipeline.start(config)
    device = profile.get_device()

    print("Camera:", device.get_info(rs.camera_info.name))
    print("Serial:", device.get_info(rs.camera_info.serial_number))

    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"Depth scale: {depth_scale} m/unit")

    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    frame_interval = 1.0 / stream_fps
    next_frame_time = 0.0

    # Let auto exposure and depth settle.
    print("Warming up...")

    for _ in range(10):
        frames = pipeline.wait_for_frames()
        align.process(frames)

    print("Streaming started.")

    try:
        while not stop_event.is_set():
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            now = time.monotonic()

            # Camera remains at 30 FPS, but network preview may be slower.
            if now < next_frame_time:
                continue

            next_frame_time = now + frame_interval

            image = np.asanyarray(color_frame.get_data()).copy()

            distance = get_region_distance(
                depth_frame,
                depth_scale,
                center_x,
                center_y,
                DEPTH_RADIUS,
            )

            # Draw measurement region.
            cv2.rectangle(
                image,
                (
                    center_x - DEPTH_RADIUS,
                    center_y - DEPTH_RADIUS,
                ),
                (
                    center_x + DEPTH_RADIUS,
                    center_y + DEPTH_RADIUS,
                ),
                (0, 255, 0),
                1,
            )

            # Draw crosshair.
            cv2.line(
                image,
                (center_x - 15, center_y),
                (center_x + 15, center_y),
                (0, 255, 0),
                2,
            )

            cv2.line(
                image,
                (center_x, center_y - 15),
                (center_x, center_y + 15),
                (0, 255, 0),
                2,
            )

            if distance is None:
                text = "Distance: N/A"
            else:
                text = f"Distance: {distance:.2f} m"

            cv2.putText(
                image,
                text,
                (center_x - 100, center_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )

            if not ok:
                continue

            frame_store.update(encoded.tobytes())

    finally:
        print("Stopping camera...")
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
    print(f"Open http://<KV260-IP>:{args.port}/")
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
