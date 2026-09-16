import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30

WARMUP_FRAMES = 10
TEST_FRAMES = 30


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

    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS,
    )

    # Transform depth coordinates into the RGB camera coordinate system
    align = rs.align(rs.stream.color)

    print("Starting Intel RealSense D455 RGB + aligned depth...")

    profile = pipeline.start(config)
    device = profile.get_device()

    print("Camera:", device.get_info(rs.camera_info.name))
    print("Serial:", device.get_info(rs.camera_info.serial_number))

    depth_sensor = device.first_depth_sensor()
    print(f"Depth scale: {depth_sensor.get_depth_scale()} m/unit")

    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    try:
        print(f"Warming up ({WARMUP_FRAMES} frames)...")

        for _ in range(WARMUP_FRAMES):
            frames = pipeline.wait_for_frames()
            align.process(frames)

        print("Starting aligned depth measurements...")

        for i in range(TEST_FRAMES):
            frames = pipeline.wait_for_frames()

            # Align depth to RGB
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print(f"Frame {i + 1:02d}: incomplete frameset")
                continue

            distance = depth_frame.get_distance(
                center_x,
                center_y,
            )

            print(
                f"Frame {i + 1:02d}: "
                f"RGB {color_frame.get_width()}x{color_frame.get_height()}, "
                f"Aligned depth "
                f"{depth_frame.get_width()}x{depth_frame.get_height()}, "
                f"RGB center ({center_x}, {center_y}) = "
                f"{distance:.3f} m"
            )

    finally:
        print("Stopping camera...")
        pipeline.stop()
        print("Camera stopped")


if __name__ == "__main__":
    main()
