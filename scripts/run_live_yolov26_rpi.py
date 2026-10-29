import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mock_yolov26_sender import (
    DependencyError,
    build_child_env,
    resolve_deps_prefix,
)

LAUNCHER_OPTIONS = ("--deps-prefix", "--binary")


def split_launcher_args(argv):
    launcher_args = []
    forwarded = []
    i = 0
    while i < len(argv):
        if argv[i] == "--":
            forwarded.extend(argv[i + 1:])
            break
        if argv[i] in LAUNCHER_OPTIONS:
            if i + 1 >= len(argv):
                raise DependencyError(
                    "missing value for %s" % argv[i]
                )
            launcher_args.extend(argv[i:i + 2])
            i += 2
            continue
        if any(argv[i].startswith(o + "=") for o in LAUNCHER_OPTIONS):
            launcher_args.append(argv[i])
            i += 1
            continue
        forwarded.extend(argv[i:])
        break
    return launcher_args, forwarded


def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="run_live_yolov26_rpi.py",
        description="Launch the native live_yolov26_rpi sender"
        " with an isolated dependency prefix",
    )
    parser.add_argument("--deps-prefix", type=str, default=None)
    parser.add_argument(
        "--binary",
        type=str,
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ),
            "build",
            "live_yolov26_rpi",
        ),
    )
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    try:
        launcher_args, forwarded = split_launcher_args(list(argv))
        ns = build_arg_parser().parse_args(launcher_args)
    except DependencyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    try:
        prefix = resolve_deps_prefix(ns.deps_prefix)
        env = build_child_env(os.environ, prefix)
    except DependencyError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    if not os.path.isfile(ns.binary):
        print("error: binary not found: %s" % ns.binary,
              file=sys.stderr)
        return 2
    argv_exec = [ns.binary] + forwarded
    try:
        os.execvpe(ns.binary, argv_exec, env)
    except OSError as exc:
        print("error: cannot exec %s: %s" % (ns.binary, exc),
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
