#!/usr/bin/env python3
"""Summarise saved `[telemetry]` logs into a markdown table.

Usage:
    python3 scripts/summarize_telemetry.py [--skip N] LOG [LOG ...]

Each LOG is the stdout of a run with `--telemetry true`: either the HDMI
app's `[telemetry]` lines or the MJPEG app's `Performance:` lines. The first N
one-second windows (default 5) are dropped as warm-up. Per log the script
prints the mean of the per-window values (fps, e2e mean, stage means, cpu)
and the largest `e2e_max` seen. Standard library only.
"""

import argparse
import os
import re
import sys

PAIR = re.compile(r"([A-Za-z0-9_]+)=([-+0-9.eE]+)")
LINE_MARKERS = ("[telemetry]", "Performance:")
# MJPEG app names -> HDMI app names
ALIASES = {"stream": "fps", "e2e_latency": "e2e_mean", "CPU": "cpu"}
FIELDS = (
    "fps", "e2e_mean", "e2e_max", "acquire", "preprocess", "dpu",
    "depth", "wait", "draw", "show", "cpu",
)


def parse_log(path, skip):
    """Return a list of {field: float} dicts, one per telemetry window."""
    windows = []
    with open(path, errors="replace") as handle:
        for line in handle:
            if not any(marker in line for marker in LINE_MARKERS):
                continue
            windows.append({
                ALIASES.get(key, key): float(value)
                for key, value in PAIR.findall(line)
            })
    return windows[skip:]


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(path, skip):
    windows = parse_log(path, skip)
    row = {"windows": len(windows)}
    for field in FIELDS:
        values = [w[field] for w in windows if field in w]
        row[field] = max(values) if field == "e2e_max" and values \
            else mean(values)
    return row


def fmt(value, digits=1):
    return "-" if value is None else "%.*f" % (digits, value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--skip", type=int, default=5,
                        help="warm-up windows to drop (default 5)")
    args = parser.parse_args(argv)

    print("| Run | windows | FPS | e2e mean (ms) | e2e max (ms) "
          "| wait | draw | show | CPU % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for path in args.logs:
        row = summarize(path, args.skip)
        name = os.path.splitext(os.path.basename(path))[0]
        print("| %s | %d | %s | %s | %s | %s | %s | %s | %s |" % (
            name, row["windows"], fmt(row["fps"]),
            fmt(row["e2e_mean"]), fmt(row["e2e_max"], 0),
            fmt(row["wait"]), fmt(row["draw"]), fmt(row["show"]),
            fmt(row["cpu"], 0),
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
