# Tests

Host-side tests. They need no camera, DPU or board: control logic, option
parsing, geometry, launchers, wire contracts and build scripts are exercised
against fixtures. Real camera, DPU and monitor behaviour is verified manually on the
KV260.

```
tests/yolov26/
├── hdmi/    CLI + fit geometry, KMS frame composition/scaling, launcher, build script
├── rpi/     Pi sender control logic and transport, launcher, mock sender
└── mjpeg/   browser option parsing and overlay drawing (extracted from the real sources)
```

```sh
python3 -m unittest discover -s tests                 # everything
python3 -m unittest discover -s tests/yolov26/hdmi    # one mode
```

Requirements: Python 3.9+ and `g++` (the `.cpp` fixtures are compiled on the fly).
Two fixtures are skipped on a PC without the board's development files: the Pi
transport fixture (GStreamer + hiredis headers) and the KMS composition fixture
(OpenCV + libdrm headers). They are meant to run on the KV260. On a development PC
without those libraries: 110 tests, 2 skipped.
