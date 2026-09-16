import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    print("Starting Intel RealSense D455...")

    profile = pipeline.start(config)
    device = profile.get_device()

    print("Camera:", device.get_info(rs.camera_info.name))
    print("Serial:", device.get_info(rs.camera_info.serial_number))

    last_frame = None

    try:
        # Skip initial frames so auto-exposure can settle
        for i in range(30):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            last_frame = np.asanyarray(
                color_frame.get_data()
            )

            print(
                f"Frame {i + 1:02d}: "
                f"{last_frame.shape[1]}x{last_frame.shape[0]}"
            )

        if last_frame is None:
            raise RuntimeError("No RGB frame received")

        output_path = "d455_test.jpg"

        if not cv2.imwrite(output_path, last_frame):
            raise RuntimeError("Failed to save image")

        print(f"Saved test frame to: {output_path}")

    finally:
        pipeline.stop()
        print("Camera stopped")


if __name__ == "__main__":
    main()
