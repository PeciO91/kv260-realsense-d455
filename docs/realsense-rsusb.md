# RealSense D455 on KV260

## Platform

- AMD/Xilinx Kria KV260
- Xilinx/PetaLinux 2022.2
- Linux kernel 5.15.36-xilinx-v2022.2
- Intel RealSense D455
- Python 3.9

## Native backend issue

The standard pyrealsense2 package used the Linux V4L2/UVC backend.

RGB streaming itself worked, but stopping the RealSense pipeline caused
a kernel Oops inside:

uvc_stop_streaming()
-> vb2_core_streamoff()
-> v4l2_ioctl()

A plain V4L2 RGB capture using v4l2-ctl worked correctly, indicating
that the issue was specific to the librealsense native V4L2/UVC path.

## Solution

librealsense 2.58.4 was built with the official RSUSB backend:

    -DFORCE_RSUSB_BACKEND=ON
    -DBUILD_PYTHON_BINDINGS=ON

The resulting librealsense library is installed under:

    /root/librealsense-rsusb

Python bindings are installed under:

    /usr/lib/python3.9/site-packages/pyrealsense2

The runtime library path is configured to include:

    /root/librealsense-rsusb/lib64

With the RSUSB backend, the following sequence works correctly:

    pipeline.start()
    wait_for_frames()
    pipeline.stop()

without crashing the kernel.

## USB connection

The D455 is connected to a USB 3 port (`lsusb -t` shows 5000M for the camera).
The early bring-up ran over USB 2.0, which limits 640x480 RGB + depth; the current
848x480 colour + 424x240 depth at 30 FPS configuration was measured on USB 3.
