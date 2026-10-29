import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30
NUM_FRAMES = 30


def main():
    pipeline = rs.pipeline()
    config = rs.config()

    # RGB stream
    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    # Depth stream
    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS,
    )

    print("Starting Intel RealSense D455 RGB + depth...")

    profile = pipeline.start(config)
    device = profile.get_device()

    print("Camera:", device.get_info(rs.camera_info.name))
    print("Serial:", device.get_info(rs.camera_info.serial_number))

    # Depth sensor information
    depth_sensor = device.first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    print(f"Depth scale: {depth_scale} m/unit")

    center_x = WIDTH // 2
    center_y = HEIGHT // 2

    try:
        for i in range(NUM_FRAMES):
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print(f"Frame {i + 1:02d}: incomplete frameset")
                continue

            # get_distance() returns metres
            distance = depth_frame.get_distance(center_x, center_y)

            print(
                f"Frame {i + 1:02d}: "
                f"RGB {color_frame.get_width()}x{color_frame.get_height()}, "
                f"Depth {depth_frame.get_width()}x{depth_frame.get_height()}, "
                f"center distance = {distance:.3f} m"
            )

    finally:
        print("Stopping camera...")
        pipeline.stop()
        print("Camera stopped")


if __name__ == "__main__":
    main()
