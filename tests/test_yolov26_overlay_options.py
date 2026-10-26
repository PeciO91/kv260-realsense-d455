import os
import re
import shutil
import subprocess
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFERENCE = os.path.join(REPO, "inference")

BROWSER_SOURCE = os.path.join(INFERENCE, "live_yolov26_realsense.cpp")
CORE_SOURCE = os.path.join(INFERENCE, "yolov26_realsense_core.cpp")


def compile_binary(source, output):
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-I", INFERENCE,
        "-x", "c++", "-",
        "-o", output,
    ]
    result = subprocess.run(
        cmd, input=source, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "compile failed:\n%s\n%s" % (result.stdout, result.stderr)
        )


def extract_browser_parser():
    with open(BROWSER_SOURCE) as handle:
        text = handle.read()
    match = re.search(
        r"constexpr int DEFAULT_PORT.*?\nclass PerformanceMonitor",
        text,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("browser parser section not found")
    return match.group(0)[: -len("class PerformanceMonitor")]


def extract_draw_annotations():
    with open(CORE_SOURCE) as handle:
        text = handle.read()
    start = text.index("void draw_detection_annotations(")
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


BROWSER_PARSER_DRIVER = r"""
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "yolov26_common.hpp"

using namespace yolov26;

%s

int main(int argc, char* argv[])
{
    try {
        const Options options = parse_options(argc, argv);
        std::printf(
            "conf=%%.4f max=%%d depth_roi=%%d conf_vis=%%d telem=%%d\n",
            options.confidence_threshold,
            options.max_detections,
            options.show_depth_roi ? 1 : 0,
            options.show_confidence ? 1 : 0,
            options.telemetry ? 1 : 0
        );
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERR: %%s\n", e.what());
        return 2;
    }
}
"""

DRAW_DRIVER = r"""
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "yolov26_common.hpp"

using namespace yolov26;

struct RectCall {
    double color[3];
};

static std::vector<RectCall> g_rect_calls;
static std::vector<std::string> g_text_calls;

namespace cv {

struct Scalar {
    double v[4];
    Scalar(double a = 0, double b = 0, double c = 0, double d = 0)
    {
        v[0] = a;
        v[1] = b;
        v[2] = c;
        v[3] = d;
    }
};

struct Point {
    int x;
    int y;
    Point(int a = 0, int b = 0) : x(a), y(b) {}
};

struct Rect {
    int x;
    int y;
    int width;
    int height;
    Rect(int a = 0, int b = 0, int w = 0, int h = 0)
        : x(a), y(b), width(w), height(h)
    {
    }
};

struct Mat {
    int cols = 0;
    int rows = 0;
};

constexpr int FONT_HERSHEY_SIMPLEX = 0;
constexpr int LINE_AA = 16;

inline void rectangle(
    Mat&,
    const Rect&,
    const Scalar& color,
    int)
{
    g_rect_calls.push_back({{color.v[0], color.v[1], color.v[2]}});
}

inline void rectangle(
    Mat&,
    Point,
    Point,
    const Scalar& color,
    int)
{
    g_rect_calls.push_back({{color.v[0], color.v[1], color.v[2]}});
}

inline void putText(
    Mat&,
    const char* text,
    Point,
    int,
    double,
    const Scalar&,
    int,
    int)
{
    g_text_calls.emplace_back(text);
}

}

%s

static int failures = 0;

static void check(bool condition, const char* name)
{
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %%s\n", name);
    }
}

static bool has_blue_rect()
{
    for (const RectCall& call : g_rect_calls) {
        if (call.color[0] == 255.0
            && call.color[1] == 0.0
            && call.color[2] == 0.0) {
            return true;
        }
    }
    return false;
}

static bool any_text_contains(const char* needle)
{
    for (const std::string& text : g_text_calls) {
        if (text.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static void reset()
{
    g_rect_calls.clear();
    g_text_calls.clear();
}

static DetectionAnnotation person_box()
{
    DetectionAnnotation a;
    a.class_id = 0;
    a.score = 0.85f;
    a.x = 10;
    a.y = 30;
    a.width = 40;
    a.height = 60;
    a.depth_x1 = 15;
    a.depth_y1 = 40;
    a.depth_x2 = 25;
    a.depth_y2 = 50;
    a.distance = 1.25f;
    return a;
}

int main()
{
    cv::Mat image;
    image.cols = 848;
    image.rows = 480;
    std::vector<DetectionAnnotation> annotations = {person_box()};
    InferenceOptions options;

    reset();
    draw_detection_annotations(image, annotations, options);
    check(!has_blue_rect(), "depth ROI rect hidden by default");
    check(any_text_contains("0.85"), "score shown by default");
    check(any_text_contains("Person"), "class label default");
    check(any_text_contains("1.25 m"), "distance shown default");

    reset();
    options.show_depth_roi = true;
    draw_detection_annotations(image, annotations, options);
    check(has_blue_rect(), "depth ROI rect shown when enabled");
    check(any_text_contains("0.85"), "score still shown with roi");

    reset();
    options.show_confidence = false;
    draw_detection_annotations(image, annotations, options);
    check(has_blue_rect(), "depth ROI rect shown without score");
    check(!any_text_contains("0.85"), "score hidden when disabled");
    check(any_text_contains("Person"), "class label without score");
    check(any_text_contains("1.25 m"), "distance kept without score");

    reset();
    options = InferenceOptions();
    options.show_confidence = false;
    annotations[0].distance = 0.0f;
    draw_detection_annotations(image, annotations, options);
    check(!has_blue_rect(), "depth ROI hidden without score");
    check(!any_text_contains("0.85"), "no score in N/A label");
    check(any_text_contains("Person"), "class label N/A case");
    check(any_text_contains("N/A"), "N/A kept without score");

    reset();
    options.show_confidence = true;
    draw_detection_annotations(image, annotations, options);
    check(any_text_contains("0.85"), "score shown in N/A label");
    check(any_text_contains("N/A"), "N/A kept with score");

    std::printf("draw fixture done, %%d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
"""


@unittest.skipUnless(shutil.which("g++"), "g++ unavailable")
class BrowserParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmp.cleanup)
        binary = os.path.join(cls._tmp.name, "browser_options")
        source = BROWSER_PARSER_DRIVER % extract_browser_parser()
        compile_binary(source, binary)
        cls.binary = binary

    def run_cli(self, *args):
        return subprocess.run(
            [self.binary, *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def parse_fields(self, result):
        self.assertEqual(0, result.returncode, result.stderr)
        fields = {}
        for pair in result.stdout.strip().split():
            key, _, value = pair.partition("=")
            fields[key] = value
        return fields

    def test_defaults(self):
        fields = self.parse_fields(self.run_cli())
        self.assertAlmostEqual(0.30, float(fields["conf"]), places=3)
        self.assertEqual("50", fields["max"])
        self.assertEqual("0", fields["depth_roi"])
        self.assertEqual("1", fields["conf_vis"])
        self.assertEqual("0", fields["telem"])

    def test_bool_overrides(self):
        fields = self.parse_fields(
            self.run_cli(
                "--show-depth-roi", "true",
                "--show-confidence", "false",
                "--telemetry", "true",
            )
        )
        self.assertEqual("1", fields["depth_roi"])
        self.assertEqual("0", fields["conf_vis"])
        self.assertEqual("1", fields["telem"])

        fields = self.parse_fields(
            self.run_cli(
                "--show-depth-roi", "false",
                "--show-confidence", "true",
                "--telemetry", "false",
            )
        )
        self.assertEqual("0", fields["depth_roi"])
        self.assertEqual("1", fields["conf_vis"])
        self.assertEqual("0", fields["telem"])

    def test_invalid_bool_rejected(self):
        for flag in (
            "--show-depth-roi",
            "--show-confidence",
            "--telemetry",
        ):
            for value in ("yes", "1", "TRUE", ""):
                result = self.run_cli(flag, value)
                self.assertNotEqual(
                    0,
                    result.returncode,
                    "%s %r accepted" % (flag, value),
                )

    def test_missing_bool_value_rejected(self):
        for flag in (
            "--show-depth-roi",
            "--show-confidence",
            "--telemetry",
        ):
            result = self.run_cli(flag)
            self.assertNotEqual(
                0, result.returncode, "bare %s accepted" % flag
            )

    def test_confidence_override(self):
        fields = self.parse_fields(self.run_cli("--confidence", "0.25"))
        self.assertAlmostEqual(0.25, float(fields["conf"]), places=3)


@unittest.skipUnless(shutil.which("g++"), "g++ unavailable")
class DrawAnnotationsTest(unittest.TestCase):
    def test_draw_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "draw_fixture")
            source = DRAW_DRIVER % extract_draw_annotations()
            compile_binary(source, binary)
            result = subprocess.run(
                [binary], capture_output=True, text=True, timeout=30
            )
            self.assertEqual(
                0,
                result.returncode,
                "%s\n%s" % (result.stdout, result.stderr),
            )


if __name__ == "__main__":
    unittest.main()
